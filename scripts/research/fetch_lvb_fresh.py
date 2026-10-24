"""Fresh confirmation set: every LongVideoBench validation question on a video never used before.

v1 and v2 used 440 questions on 252 LongVideoBench validation videos (data/longvideobench_full). This takes the
remaining validation videos (all question categories) so v3 can be checked on data that played no part in any
decision. Videos are read by HTTP range from the dataset's tar archive (same helpers and catalog as
download_longvideobench_subset.py); subtitles come from the dataset's subtitle archive.

Output (local layout, every row experiment_split = "test"):
    data/lvb_fresh/videos/<video_id>.mp4, data/lvb_fresh/transcripts/<video_id>.json
    data/lvb_fresh/qa.jsonl, data/lvb_fresh/manifest.json

    python scripts/research/fetch_lvb_fresh.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download_longvideobench_subset import (  # noqa: E402
    DATASET_ID,
    LICENSE_NOTE,
    PART_NAMES,
    archive_name,
    convert_item,
    subtitles_to_transcript,
    to_jsonable,
)
from huggingface_hub import hf_hub_download  # noqa: E402

from videoqa.sources.tar_range import RemoteMultipartTar, build_catalog, hf_dataset_tar_parts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/lvb_fresh")
    ap.add_argument("--used", default="data/longvideobench_full/qa.jsonl", help="questions used by v1/v2")
    ap.add_argument("--cache", default="cache/lvb_meta")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    token = a.token or (Path.home() / ".cache" / "huggingface" / "token").read_text(encoding="utf-8").strip()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)

    val = json.loads(Path(hf_hub_download(DATASET_ID, "lvb_val.json", repo_type="dataset", token=token,
                                          local_dir=a.cache)).read_text(encoding="utf-8"))
    subs = tarfile.open(hf_hub_download(DATASET_ID, "subtitles.tar", repo_type="dataset", token=token, local_dir=a.cache))
    names = set(subs.getnames())
    used_videos = {json.loads(x)["video_id"] for x in Path(a.used).read_text(encoding="utf-8").splitlines() if x.strip()}
    by_video: dict[str, list[dict]] = {}
    for d in val:
        by_video.setdefault(d["video_id"], []).append(d)
    fresh = sorted(v for v in by_video if v not in used_videos and f"subtitles/{by_video[v][0]['subtitle_path']}" in names)
    base, parts = hf_dataset_tar_parts(DATASET_ID, PART_NAMES, token=token)
    tar = RemoteMultipartTar(base, parts, token=token, max_retries=6)
    catalog = build_catalog(tar, Path(a.cache) / "lvb_video_catalog.json")
    fresh = [v for v in fresh if archive_name(by_video[v][0]) in catalog.entries]
    n_q = sum(len(by_video[v]) for v in fresh)
    gb = sum(catalog.entries[archive_name(by_video[v][0])].size for v in fresh) / 1e9
    print(f"fresh videos {len(fresh)} (used before: {len(used_videos)}), questions {n_q}, {gb:.1f} GB", flush=True)
    if a.dry_run:
        return
    transcripts, found, errors = {}, [], {}
    t0 = time.perf_counter()
    for k, vid in enumerate(fresh, 1):
        rec0 = by_video[vid][0]
        raw = json.loads(subs.extractfile(f"subtitles/{rec0['subtitle_path']}").read().decode("utf-8"))
        t = subtitles_to_transcript(vid, raw, offset_s=rec0["starting_timestamp_for_subtitles"], clip_duration_s=rec0["duration"])
        transcripts[vid] = t
        (out / "transcripts" / f"{vid}.json").write_text(
            json.dumps([{"start": s.start_s, "end": s.end_s, "text": s.text} for s in t.segments]), encoding="utf-8")
        entry = catalog.entries[archive_name(rec0)]
        dest = out / "videos" / f"{vid}.mp4"
        if dest.exists() and dest.stat().st_size == entry.size:
            found.append(vid)
            continue
        try:
            tar.extract_to(entry, dest)
            found.append(vid)
            if k % 20 == 0:
                print(f"  [{k}/{len(fresh)}] {time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - record and continue
            errors[vid] = f"{type(e).__name__}: {e}"
    rows = []
    for vid in found:
        for record in by_video[vid]:
            q = to_jsonable(convert_item(record, transcripts[vid], LICENSE_NOTE).qa)
            q["experiment_split"] = "test"
            q["source_split"] = "lvb_validation_fresh"
            rows.append(json.dumps(q))
    (out / "qa.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {"dataset": DATASET_ID, "license": LICENSE_NOTE, "rule": "LongVideoBench validation videos not used by v1/v2",
                "videos": len(found), "questions": len(rows), "errors": errors, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "errors"}, indent=1), "errors:", len(errors))


if __name__ == "__main__":
    main()
