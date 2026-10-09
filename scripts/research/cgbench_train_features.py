"""Scale Head A's training data: features for every released CG-Bench video except dev/test.

Head A needs only frame features, the question text and the human clue intervals (no subtitles),
so all 770 videos in the release archives can provide training questions, except the 14 videos of
our dev/test split. To avoid storing ~160 GB of video, each video is fetched by byte range from its
remote archive, turned into MobileCLIP features every 2 s, and deleted. Downloads are prefetched one
video ahead so network and GPU work overlap. Resumable: existing feature files are skipped.

    data/cgbench/features/<video_uid>.npz      (same format as scripts/v3_features.py)
    data/cgbench/train_scale_log.json          per video: archive, bytes, seconds, frames

    python scripts/research/cgbench_train_features.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fetch_cgbench_videos import REPO, central_directory, extract_member  # noqa: E402

DATA = Path("data/cgbench")
TMP = DATA / "_tmp"


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import get_token, hf_hub_url  # noqa: PLC0415

    from v3_features import video_features  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    token = get_token()
    TMP.mkdir(parents=True, exist_ok=True)
    (DATA / "features").mkdir(parents=True, exist_ok=True)
    held_out = {json.loads(x)["video_id"] for x in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip() and json.loads(x)["experiment_split"] in ("dev", "test")}
    with_questions = {q["video_uid"] for q in json.loads(Path("data/cgbench_raw/cgbench.json").read_text(encoding="utf-8"))}
    jobs = []
    for k in range(1, 33):
        url = hf_hub_url(REPO, f"video_chunk_{k:02d}.zip", repo_type="dataset")
        _, members = central_directory(url, token)
        for m in members:
            vid = Path(m["name"]).stem
            if (m["name"].endswith("/") or "__MACOSX" in m["name"] or vid in held_out or vid not in with_questions
                    or (DATA / "features" / f"{vid}.npz").exists()):
                continue
            jobs.append((url, m, vid))
    print(f"{len(jobs)} videos to process (excluding {len(held_out)} dev/test videos)", flush=True)
    log_path = DATA / "train_scale_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)

    def fetch(job):
        url, m, vid = job
        dest = TMP / f"{vid}.mp4"
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                extract_member(url, token, m, dest)
                return job, dest, time.perf_counter() - t0, None
            except Exception as e:  # noqa: BLE001 - network errors are retried, then recorded
                err = str(e)[:200]
                time.sleep(10 * (attempt + 1))
        return job, dest, time.perf_counter() - t0, err

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(fetch, jobs[0]) if jobs else None
        for n in range(len(jobs)):
            job, dest, dl_s, err = pending.result()
            pending = pool.submit(fetch, jobs[n + 1]) if n + 1 < len(jobs) else None
            vid = job[2]
            if err:
                log[vid] = {"error": err}
            else:
                t0 = time.perf_counter()
                try:
                    times, emb, dur = video_features(enc, dest, 2.0)
                    np.savez_compressed(DATA / "features" / f"{vid}.npz", times=times, emb=emb,
                                        duration=np.float32(dur), step=np.float32(2.0))
                    log[vid] = {"bytes": dest.stat().st_size, "download_s": round(dl_s, 1),
                                "feature_s": round(time.perf_counter() - t0, 1), "frames": len(times)}
                except Exception as e:  # noqa: BLE001 - a broken video is recorded, not fatal
                    log[vid] = {"error": f"features: {str(e)[:200]}"}
                dest.unlink(missing_ok=True)
            log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
            print(f"{n + 1}/{len(jobs)} {vid} {log[vid]}", flush=True)


if __name__ == "__main__":
    main()
