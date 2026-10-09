"""Frozen MobileCLIP-S2 features every STEP seconds for whole videos (input to Head A and baselines).

    data/<dataset>/features/<video_id>.npz   times (float32, s), emb (float16, N x 512, unit norm)

Frames are decoded with the project's decoder (640 px, the answerer's size) and embedded in batches.
Resumable: existing feature files are skipped. Text embeddings are computed where needed, not here.

    python scripts/v3_features.py --dataset cgbench --step 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def video_features(enc, path: Path, step: float, chunk: int = 120):
    """Decode every ``step`` s in chunks (bounded memory) and embed each chunk."""
    from videoqa.frames import decode_at, probe  # noqa: PLC0415

    dur = probe(str(path)).duration_s
    grid = list(np.arange(0.5, max(dur - 0.05, 0.5), step))
    times, embs = [], []
    for i in range(0, len(grid), chunk):
        frames = decode_at(str(path), grid[i:i + chunk], max_side=640, video_id=path.stem).frames
        if frames:
            embs.append(enc.embed_images([f.image for f in frames]))
            times.extend(f.decoded_pts_s for f in frames)
    emb = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    return np.asarray(times, np.float32), emb.astype(np.float16), dur


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cgbench")
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--split", default=None, help="only videos with questions in this experiment split")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    root = Path("data") / a.dataset
    out = root / "features"
    out.mkdir(parents=True, exist_ok=True)
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    videos = sorted((root / "videos").glob("*.mp4"))
    if a.split:
        rows = [json.loads(x) for x in (root / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        keep = {r["video_id"] for r in rows if r.get("experiment_split") == a.split}
        videos = [v for v in videos if v.stem in keep]
    if a.limit:
        videos = videos[: a.limit]
    for n, v in enumerate(videos):
        dest = out / f"{v.stem}.npz"
        if dest.exists():
            continue
        t0 = time.perf_counter()
        times, emb, dur = video_features(enc, v, a.step)
        np.savez_compressed(dest, times=times, emb=emb, duration=np.float32(dur), step=np.float32(a.step))
        print(f"{n + 1}/{len(videos)} {v.stem}: {len(times)} frames, {dur / 60:.1f} min video, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
