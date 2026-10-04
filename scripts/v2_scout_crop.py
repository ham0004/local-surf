"""Scout coverage ablation: does MobileCLIP's centre crop hide board content?

The installed MobileCLIP-S2 transform resizes the short side to 256 and centre
crops 256x256, so a 640x360 lecture frame keeps only 56.25% of its width.
The final answerer always receives the full frame; only the SELECTION feature
is cropped. Arms (same pools, same question embedding, K = 4, frozen answerer):

  crop     the current transform (reproduces the stored MobileCLIP ranking)
  pad      whole frame letterboxed to a square with its mean colour (same cost:
           one image encoding per frame; writing becomes smaller)
  tiles3   left / centre / right square tiles, score = best tile (3 encodings
           per frame: charged as 3x the scout image cost)

Only the reranking changes. The Path B pool itself was built with the cropped
scout, so this is a lower bound on what a padded scout could do end to end.

    python scripts/v2_scout_crop.py --runs runs/v2_main runs/v2_balanced --out reports/v2_scout_crop
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from videoqa.config import load_config
from videoqa.v2.candidates import load_pool
from videoqa.v2.teacher import CachedTeacher, answerer_identity

K = 4


def letterbox(im: Image.Image) -> Image.Image:
    """Whole frame on a square canvas filled with the frame's mean colour."""
    side = max(im.size)
    fill = tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def tiles3(im: Image.Image) -> list[Image.Image]:
    """Left, centre and right square tiles that together cover the full width."""
    s = min(im.size)
    xs = [0, (im.width - s) // 2, im.width - s]
    return [im.crop((x, 0, x + s, s)) for x in xs]


def lecture_bootstrap(diffs: dict, videos: dict, repeats: int = 5000) -> dict:
    by: dict[str, list[float]] = {}
    for q, d in diffs.items():
        by.setdefault(videos[q], []).append(d)
    sums = np.array([sum(x) for x in by.values()])
    counts = np.array([len(x) for x in by.values()])
    ids = np.random.default_rng(0).integers(0, len(sums), (repeats, len(sums)))
    boot = sums[ids].sum(axis=1) / counts[ids].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95_lecture_bootstrap": np.percentile(boot, [2.5, 97.5]).tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["runs/v2_main", "runs/v2_balanced"])
    p.add_argument("--out", default="reports/v2_scout_crop")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from videoqa.answerer import make_answerer  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    cfg = load_config(a.config)
    ident = answerer_identity(cfg)
    teacher_id = f"{cfg['answerer']['model_id']}@{cfg['answerer'].get('revision')}"
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    answerer = None
    summary = {}
    for run in map(Path, a.runs):
        name = run.name
        meta = json.loads((run / "pools.json").read_text(encoding="utf-8"))
        pools = {q: load_pool(run / "pools", q) for q in meta["qa_ids"]}
        # Legacy records are reused only from caches built on THIS run's pool files.
        own = {"v2_balanced": Path("reports/v2_balanced/teacher_cache_balanced.jsonl")}.get(name, run / "teacher_cache.jsonl")
        teacher = CachedTeacher(None, teacher_id, out / f"teacher_cache_{name}.jsonl", identity=ident,
                                legacy_paths=(own,) if own.exists() else ())
        per_q, timing = {}, {"pad": [], "tiles3": [], "crop_check_max_abs": 0.0}
        for q in sorted(pools):
            pool = pools[q]
            ims = [c.image for c in pool.candidates]
            qe = pool.question_emb
            t0 = time.perf_counter()
            crop = enc.embed_images(ims) @ qe
            timing["crop_check_max_abs"] = max(timing["crop_check_max_abs"],
                                               float(np.max(np.abs(crop - [c.clip_sim for c in pool.candidates]))))
            t0 = time.perf_counter()
            pad = enc.embed_images([letterbox(im) for im in ims]) @ qe
            timing["pad"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            tl = (enc.embed_images([t for im in ims for t in tiles3(im)]) @ qe).reshape(len(ims), 3).max(axis=1)
            timing["tiles3"].append(time.perf_counter() - t0)
            row = {"video_id": pool.video_id}
            for arm, s in (("crop", crop), ("pad", pad), ("tiles3", tl)):
                chosen = [pool.candidates[i] for i in sorted(range(len(s)), key=lambda i: (-s[i], i))[:K]]
                if teacher.cached_prediction(pool, chosen) is None and teacher.answerer is None:
                    answerer = answerer or make_answerer(cfg)
                    teacher.answerer = answerer
                row[arm] = teacher.quality(pool, chosen)
                row[f"{arm}_ids"] = sorted(c.id for c in chosen)
            per_q[q] = row
        videos = {q: r["video_id"] for q, r in per_q.items()}
        acc = {arm: float(np.mean([r[arm] for r in per_q.values()])) for arm in ("crop", "pad", "tiles3")}
        summary[name] = {
            "questions": len(per_q), "accuracy": acc,
            "selection_changed_vs_crop": {arm: float(np.mean([r[f"{arm}_ids"] != r["crop_ids"] for r in per_q.values()]))
                                          for arm in ("pad", "tiles3")},
            "paired": {f"{arm}_minus_crop": lecture_bootstrap({q: r[arm] - r["crop"] for q, r in per_q.items()}, videos)
                       for arm in ("pad", "tiles3")},
            "scout_seconds_median_per_question": {k: float(np.median(v)) for k, v in timing.items() if isinstance(v, list)},
            "stored_vs_recomputed_crop_similarity_max_abs_diff": timing["crop_check_max_abs"],
            "fresh_calls": teacher.calls, "answer_seconds": teacher.seconds,
        }
        (out / f"per_question_{name}.json").write_text(json.dumps(per_q, indent=1), encoding="utf-8")
        print(json.dumps({name: summary[name]}, indent=1), flush=True)
    summary["notes"] = ["Only the reranking feature changes; pools were built with the cropped scout.",
                        "Scout seconds are measured warm on GPU for all candidates of a question.",
                        "Development data (197 synthetic MIT questions)."]
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
