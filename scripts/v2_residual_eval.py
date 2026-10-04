"""Evaluate exported out-of-fold selections with the SAME frozen four-frame VLM.

Default is cache-only. Positive --max-new-calls explicitly enables bounded local
inference; no training, model download, transcript change or candidate rebuilding.
Original caches and scores are read-only. A resumable new cache stores fresh calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from videoqa.config import load_config
from videoqa.v2.candidates import load_pool
from videoqa.v2.teacher import CachedTeacher, answerer_identity


ARMS = ("clip", "mmr", "relevance", "residual_base", "residual_context")


def paired(rows, arm, baseline, repeats=5000):
    by_video = {}
    for row in rows:
        if row[arm] is not None and row[baseline] is not None:
            by_video.setdefault(row["video_id"], []).append(row[arm] - row[baseline])
    if not by_video:
        return None
    sums = np.array([sum(ds) for ds in by_video.values()])
    counts = np.array([len(ds) for ds in by_video.values()])
    ids = np.random.default_rng(0).integers(0, len(sums), (repeats, len(sums)))
    bs = sums[ids].sum(axis=1) / counts[ids].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95_video_bootstrap": np.percentile(bs, [2.5, 97.5]).tolist(),
            "paired_questions": int(counts.sum()), "videos": len(counts)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("runs/v2_main"))
    p.add_argument("--predictions", type=Path, default=Path("reports/v2_review/residual_probe_predictions.jsonl"))
    p.add_argument("--output", type=Path, default=Path("reports/v2_review/residual_qa"))
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    p.add_argument("--max-new-calls", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=900)
    a = p.parse_args()
    if a.max_new_calls < 0 or a.max_seconds <= 0:
        p.error("call budget must be nonnegative and time budget positive")
    # All weights must already exist locally. Even an enabled inference run cannot download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cfg = load_config(a.config)
    answer_cfg = cfg["answerer"]
    teacher_id = f"{answer_cfg['model_id']}@{answer_cfg.get('revision')}"
    predictions_bytes = a.predictions.read_bytes()
    manifest = {"predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                "answerer_config": answer_cfg, "run": str(a.run.resolve()),
                "k": 4, "arms": ARMS, "protocol": "frozen original pools and transcripts; nested OOF selections"}
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output / "manifest.json"
    canonical = json.loads(json.dumps(manifest))
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != canonical:
        raise ValueError("Output belongs to a different experiment; use a new --output")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    new = CachedTeacher(None, teacher_id, a.output / "new_teacher_cache.jsonl", identity=answerer_identity(cfg),
                        legacy_paths=(a.run / "teacher_cache.jsonl",))
    own_cache = a.output / "new_teacher_cache.jsonl"
    # Every line in this output's own cache is a fresh call made by an earlier resume of this run.
    previous_new_calls = (sum(1 for line in own_cache.read_text(encoding="utf-8").splitlines() if line.strip())
                          if own_cache.exists() else 0)
    records = [json.loads(s) for s in predictions_bytes.decode("utf-8").splitlines() if s.strip()]
    if len({r["qa_id"] for r in records}) != len(records):
        raise ValueError("Duplicate question predictions")
    records.sort(key=lambda r: hashlib.sha256(r["qa_id"].encode()).hexdigest())
    rows, missing = [], []
    started = time.perf_counter()
    budget_file = a.output / "budget.json"
    budget = json.loads(budget_file.read_text()) if budget_file.exists() else {"answer_seconds": 0.0}
    prior_seconds = budget["answer_seconds"]
    with (a.output / "eval_rows.jsonl").open("w", encoding="utf-8") as out:
        for n, rec in enumerate(records):
            pool = load_pool(a.run / "pools", rec["qa_id"])
            if pool.video_id != rec["video_id"]:
                raise ValueError("Prediction/video mismatch")
            candidates = {c.id: c for c in pool.candidates}
            row = {"qa_id": pool.qa_id, "video_id": pool.video_id}
            for arm in ARMS:
                ids = rec["full_pool_selected_ids"][arm]
                if len(ids) != min(4, len(candidates)) or len(set(ids)) != len(ids):
                    raise ValueError("Invalid selected frame set")
                chosen = [candidates[i] for i in ids]
                key = f"{pool.qa_id}|{','.join(sorted(ids))}"
                if new.cached_prediction(pool, chosen) is None:
                    if (new.calls + previous_new_calls >= a.max_new_calls or
                            new.seconds + prior_seconds >= a.max_seconds):
                        row[arm] = None
                        missing.append({"qa_id": pool.qa_id, "arm": arm, "key": key})
                        continue
                    if new.answerer is None:
                        from videoqa.answerer import make_answerer
                        new.answerer = make_answerer(cfg)
                row[arm] = new.quality(pool, chosen)
                budget_file.write_text(json.dumps({"answer_seconds": prior_seconds + new.seconds,
                                                   "fresh_calls": previous_new_calls + new.calls}), encoding="utf-8")
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if (n + 1) % 20 == 0:
                print(f"QA {n+1}/{len(records)}, fresh calls={new.calls}, answer seconds={new.seconds:.1f}", flush=True)
    complete = not missing
    summary = {"complete": complete, "questions": len(rows), "videos": len({r["video_id"] for r in rows}),
               "accuracy": {}, "paired": {}, "fresh_calls_this_run": new.calls,
               "fresh_calls_total": previous_new_calls + new.calls,
               "answer_seconds_total": prior_seconds + new.seconds,
               "wall_seconds_this_run": time.perf_counter() - started,
               "missing_evaluations": len(missing),
               "warning": "Exploratory on reused synthetic MIT development data; not an independent benchmark or novelty claim."}
    for arm in ARMS:
        observed = [r[arm] for r in rows if r[arm] is not None]
        summary["accuracy"][arm] = {"correct": int(sum(observed)), "evaluated": len(observed),
                                    "accuracy": float(np.mean(observed)) if observed else None}
    for arm, baseline in (("residual_base", "clip"), ("residual_base", "relevance"),
                          ("residual_base", "mmr"), ("residual_context", "clip"),
                          ("residual_context", "residual_base")):
        summary["paired"][f"{arm}_minus_{baseline}"] = paired(rows, arm, baseline)
    if not complete:
        summary["warning"] += " Incomplete cache coverage is selection-biased; do not report subset accuracy as the full result."
    (a.output / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.output / "missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
