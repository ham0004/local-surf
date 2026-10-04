"""Evidence-loss audit for Path B's stable-run collapse (CPU only, no model calls).

Path B keeps ONE frame per run of visually similar scan frames: the LAST frame,
judged with an 8x8 average hash (distance to the run's first frame <= 6).
The hash is global; a small edit (a minus sign, an exponent) or an erasure can
leave it unchanged. This script re-decodes the exact scan frames of every
question, replays the collapse rule and, for each frame it drops, measures what
the dropped frame shows that the kept frame does not.

Local change is measured on a 16x9 grid of 40x40-pixel tiles (640x360 frames):
  changed tile   mean absolute grey difference > 12
  ink            share of pixels brighter than the tile median + 35 (chalk on a
                 dark board)
  lost ink       a changed tile where the dropped frame has more ink than the
                 kept frame by > 0.02 (content that was visible, then gone)
Small edits touch 1-3 tiles; the lecturer walking past touches many. These are
pixel measurements, not proof that the lost content answers any question.

    python scripts/v2_dedup_audit.py --data data/mit_lectures --qa-file qa_v2.jsonl \
        --run runs/v2_balanced --out reports/v2_evidence_loss
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from videoqa.datasets import load_local_dataset
from videoqa.frames import decode_at, hamming_hex, probe
from videoqa.retrieval import bm25_rank, build_windows
from videoqa.transcript import build_units
from videoqa.v2.candidates import PoolConfig, scan_timestamps

TILE = 40
CHANGED_MAD, INK_DELTA, INK_LOST = 12.0, 35.0, 0.02


def tiles(gray: np.ndarray) -> np.ndarray:
    """(rows, cols, TILE, TILE) view of a greyscale frame, cropped to whole tiles."""
    h, w = (gray.shape[0] // TILE) * TILE, (gray.shape[1] // TILE) * TILE
    g = gray[:h, :w]
    return g.reshape(h // TILE, TILE, w // TILE, TILE).swapaxes(1, 2)


def ink(t: np.ndarray) -> np.ndarray:
    med = np.median(t, axis=(2, 3), keepdims=True)
    return (t > med + INK_DELTA).mean(axis=(2, 3))


def compare(dropped: Image.Image, kept: Image.Image) -> dict:
    a = tiles(np.asarray(dropped.convert("L"), dtype=np.float32))
    b = tiles(np.asarray(kept.convert("L"), dtype=np.float32))
    mad = np.abs(a - b).mean(axis=(2, 3))
    changed = mad > CHANGED_MAD
    lost = changed & (ink(a) - ink(b) > INK_LOST)
    return {"changed_tiles": int(changed.sum()), "lost_ink_tiles": int(lost.sum()),
            "max_tile_mad": float(mad.max()), "global_mad": float(np.abs(a - b).mean()),
            "lost_cells": np.argwhere(lost).tolist()}


def collapse_runs(frames, threshold: int):
    """Replay _stable_representatives, returning each run's frames (kept = last)."""
    runs, cur = [], []
    for f in frames:
        if cur and hamming_hex(cur[0].phash, f.phash) > threshold:
            runs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    return runs


def draw_example(dropped, kept, cells, path: Path) -> None:
    pair = Image.new("RGB", (dropped.width, dropped.height * 2))
    pair.paste(dropped, (0, 0))
    pair.paste(kept, (0, dropped.height))
    d = ImageDraw.Draw(pair)
    for r, c in cells:
        for y0 in (0, dropped.height):
            d.rectangle([c * TILE, y0 + r * TILE, c * TILE + TILE - 1, y0 + r * TILE + TILE - 1],
                        outline=(255, 0, 0), width=2)
    pair.convert("RGB").save(path, quality=80)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/mit_lectures")
    p.add_argument("--qa-file", default="qa_v2.jsonl")
    p.add_argument("--run", default="runs/v2_balanced", help="its pool_config fixes the scan policy")
    p.add_argument("--out", default="reports/v2_evidence_loss")
    p.add_argument("--examples", type=int, default=6)
    a = p.parse_args()
    out = Path(a.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.run) / "pools.json").read_text(encoding="utf-8"))
    cfg = PoolConfig(**meta["pool_config"]) if "pool_config" in meta else PoolConfig(scan_step_s=5.0, scan_cap=24)
    items, _ = load_local_dataset(a.data, splits=("train", "dev", "calibration", "test"), qa_file=a.qa_file)
    items = {it.qa.qa_id: it for it in items if it.qa.qa_id in set(meta["qa_ids"])}

    rows, cases, t0 = [], [], time.perf_counter()
    for q in sorted(items):
        it = items[q]
        segs = it.transcript.segments
        duration = probe(it.video_path).duration_s
        units = build_units(segs, cfg.unit_seconds, 5.0)
        query = it.qa.question + " " + " ".join(it.qa.options or [])
        windows = build_windows(bm25_rank(query, units, 64), units, cfg.max_windows, cfg.neighbour_expansion, duration)
        times = scan_timestamps(windows, duration, cfg.scan_step_s, cfg.scan_cap, cfg.scan_policy)
        frames = sorted(decode_at(it.video_path, times, max_side=cfg.max_side, video_id=it.qa.video_id).frames,
                        key=lambda f: f.decoded_pts_s)
        runs = collapse_runs(frames, cfg.stable_hamming)
        q_drop = q_small = q_lost = 0
        for run in runs:
            kept = run[-1]
            for f in run[:-1]:
                if f.digest == kept.digest:
                    continue
                c = compare(f.image, kept.image)
                q_drop += 1
                small = 1 <= c["changed_tiles"] <= 3
                q_small += small
                q_lost += c["lost_ink_tiles"] > 0
                cases.append({"qa_id": q, "dropped_s": f.decoded_pts_s, "kept_s": kept.decoded_pts_s,
                              "hash_distance": hamming_hex(f.phash, kept.phash), "small_edit": small, **c,
                              "_images": (f.image, kept.image)})
        rows.append({"qa_id": q, "scan_frames": len(frames), "runs": len(runs), "dropped": q_drop,
                     "dropped_small_edit": q_small, "dropped_with_lost_ink": q_lost})

    # Examples: small, local ink losses are the cases a global hash cannot see.
    ranked = sorted((c for c in cases if c["small_edit"] and c["lost_ink_tiles"]),
                    key=lambda c: -c["max_tile_mad"])
    for n, c in enumerate(ranked[: a.examples]):
        d, k = c["_images"]
        c["example"] = f"examples/{n:02d}_{c['qa_id'].replace(':', '_')}_{c['dropped_s']:.0f}s.jpg"
        draw_example(d, k, c["lost_cells"], out / c["example"])
    for c in cases:
        c.pop("_images")
    n_drop = sum(r["dropped"] for r in rows)
    summary = {
        "questions": len(rows), "scan_frames": sum(r["scan_frames"] for r in rows),
        "kept_representatives": sum(r["runs"] for r in rows), "dropped_frames": n_drop,
        "dropped_with_any_changed_tile": sum(c["changed_tiles"] > 0 for c in cases),
        "dropped_small_local_edit_1to3_tiles": sum(c["small_edit"] for c in cases),
        "dropped_with_lost_ink": sum(c["lost_ink_tiles"] > 0 for c in cases),
        "dropped_small_edit_with_lost_ink": len(ranked),
        "questions_with_small_edit_lost_ink": len({c["qa_id"] for c in ranked}),
        "hash_distance_of_small_edits": np.bincount([c["hash_distance"] for c in cases if c["small_edit"]]).tolist(),
        "thresholds": {"tile_px": TILE, "changed_mad": CHANGED_MAD, "ink_delta": INK_DELTA, "ink_lost": INK_LOST},
        "scan_policy": cfg.scan_policy, "cpu_seconds": time.perf_counter() - t0,
        "examples": [c["example"] for c in ranked[: a.examples]],
        "notes": ["Pixel evidence of lost content, not evidence that it answers the question.",
                  "Merge-step drops (within 1 s or hash <= 3 of a Path A frame) are not replayed here."],
    }
    (out / "dedup_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "dedup_cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    (out / "dedup_per_question.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
