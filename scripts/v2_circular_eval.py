"""Option-order debiased evaluation (circular evaluation) for the MIT question set.

Why: the generator never shuffled options and placed the answer early (gold is
A or B in 171/197 questions), and the "drop if answerable without evidence"
filter removed C/D-gold questions more often (61% vs 39%), because the 2B
answerer prefers C/D. The kept set is therefore anti-aligned with the
answerer's letter preference, which distorts absolute accuracies (transcript
only scores 15.2%, below the 25% chance level; "always B" would score 46.7%).

Circular evaluation (as in MMBench's CircularEval) asks every question under
all 4 cyclic rotations of its options. The per-question score is the mean over
rotations (position bias cancels in expectation); "strict" requires all 4.
Frames, retained transcript and selections are unchanged; only option order
varies. Rotation 0 is the original prompt (served from the run's cache).

    python scripts/v2_circular_eval.py --run runs/v2_main --arms transcript_only A B F \
        --out reports/v2_circular
    python scripts/v2_circular_eval.py --run runs/v2_balanced --arms A F --out reports/v2_circular
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from videoqa.config import load_config
from videoqa.v2.candidates import load_pool
from videoqa.v2.selectors import MMRSelector, SimilarityTopK
from videoqa.v2.teacher import CachedTeacher, answerer_identity

K = 4
SELECT = {
    "transcript_only": lambda p: [],
    "A": lambda p: SimilarityTopK().select(p, K),
    "B": lambda p: MMRSelector().select(p, K),
    "F": lambda p: sorted(p.candidates, key=lambda c: -c.head_a_visual)[:K],
}
OWN_CACHE = {"v2_balanced": Path("reports/v2_balanced/teacher_cache_balanced.jsonl")}


def rotate(pool, r: int):
    """Same pool with options cyclically rotated left by r; gold index follows its option."""
    n = len(pool.options)
    return dataclasses.replace(pool, options=pool.options[r:] + pool.options[:r],
                               gold_option_index=(pool.gold_option_index - r) % n)


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
    p.add_argument("--run", default="runs/v2_main")
    p.add_argument("--arms", nargs="+", default=["transcript_only", "A", "B", "F"])
    p.add_argument("--out", default="reports/v2_circular")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from videoqa.answerer import make_answerer  # noqa: PLC0415

    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    ident = answerer_identity(cfg)
    teacher_id = f"{cfg['answerer']['model_id']}@{cfg['answerer'].get('revision')}"
    own = OWN_CACHE.get(run.name, run / "teacher_cache.jsonl")
    teacher = CachedTeacher(make_answerer(cfg), teacher_id, out / f"teacher_cache_{run.name}.jsonl",
                            identity=ident, legacy_paths=(own,) if own.exists() else ())
    meta = json.loads((run / "pools.json").read_text(encoding="utf-8"))
    pools = {q: load_pool(run / "pools", q) for q in meta["qa_ids"]}

    per_q, letters, t0 = {}, collections.Counter(), time.perf_counter()
    for n, q in enumerate(sorted(pools)):
        pool = pools[q]
        row = {"video_id": pool.video_id}
        for arm in a.arms:
            frames = SELECT[arm](pool)
            scores = []
            for r in range(len(pool.options)):
                rp = rotate(pool, r)
                scores.append(teacher.quality(rp, frames))
                rec = teacher.cached_prediction(rp, frames)
                letters[(arm, rec["option"])] += 1
            row[arm] = scores
        per_q[q] = row
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(pools)} calls={teacher.calls} {teacher.seconds / max(teacher.calls, 1):.2f}s/call",
                  flush=True)

    videos = {q: r["video_id"] for q, r in per_q.items()}
    res = {"run": str(run), "questions": len(per_q), "arms": {}, "paired_circular_mean": {},
           "predicted_option_position": {arm: [letters[(arm, i)] for i in (0, 1, 2, 3, None)] for arm in a.arms},
           "fresh_calls": teacher.calls, "answer_seconds": teacher.seconds,
           "wall_seconds": time.perf_counter() - t0}
    for arm in a.arms:
        s = np.array([per_q[q][arm] for q in sorted(per_q)])
        res["arms"][arm] = {"original_rotation0": float(s[:, 0].mean()), "circular_mean": float(s.mean()),
                            "circular_strict_all4": float(s.min(axis=1).mean()),
                            "per_rotation": s.mean(axis=0).tolist()}
    for x in a.arms:
        for y in a.arms:
            if x < y:
                res["paired_circular_mean"][f"{x}_minus_{y}"] = lecture_bootstrap(
                    {q: float(np.mean(per_q[q][x]) - np.mean(per_q[q][y])) for q in per_q}, videos)
    (out / f"summary_{run.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / f"per_question_{run.name}.json").write_text(json.dumps(per_q, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("arms", "paired_circular_mean", "predicted_option_position")}, indent=1))


if __name__ == "__main__":
    main()
