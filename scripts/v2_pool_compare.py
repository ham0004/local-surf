"""Candidate-generation experiment: legacy vs balanced visual scan, same selectors.

Only the pool changes. Questions, retained transcript, answerer, K = 4 and the
three label-free selectors (MobileCLIP top-k, MMR, zero-shot relevance) are
identical, so a difference is attributable to which candidates exist. No
learned model is retrained; that would mix two changes.

Per policy it reports
  - four-frame QA accuracy for each selector, with lecture-bootstrap CIs
    (balanced - legacy, paired by question)
  - evidence recall: share of questions with a candidate (pool recall) or a
    selected frame (selection recall) inside the question's evidence interval
  - pool size and the stage costs that a label-free selector actually pays
    (retrieval, Path A scoring, decoding, MobileCLIP; OCR is not needed)

    python -m videoqa.v2.experiment pools --data data/mit_lectures --qa-file qa_v2.jsonl \
        --scan-policy balanced --run runs/v2_balanced
    python scripts/v2_pool_compare.py --legacy runs/v2_main --balanced runs/v2_balanced \
        --out reports/v2_balanced
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from videoqa.config import load_config
from videoqa.v2.candidates import load_pool
from videoqa.v2.selectors import MMRSelector, ScoreTopK, SimilarityTopK
from videoqa.v2.teacher import CachedTeacher, answerer_identity

K = 4
SELECTORS = [SimilarityTopK(), MMRSelector(),
             ScoreTopK(lambda p: {c.id: c.head_a_visual for c in p.candidates}, "F_relevance")]
LABEL_FREE_STAGES = ("retrieval", "head_a", "decode", "clip")


def _pools(run: Path) -> dict:
    meta = json.loads((run / "pools.json").read_text(encoding="utf-8"))
    return {q: load_pool(run / "pools", q) for q in meta["qa_ids"]}, meta


def _evidence(meta: dict) -> dict:
    path = Path(meta["data"]) / meta["qa_file"]
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    return {r["qa_id"]: r.get("evidence_intervals_s") or [] for r in rows}


def _inside(t: float, intervals) -> bool:
    return any(a <= t <= b for a, b in intervals)


def _bootstrap(diffs: dict, videos: dict, repeats: int = 5000) -> dict:
    by: dict[str, list[float]] = {}
    for q, d in diffs.items():
        by.setdefault(videos[q], []).append(d)
    sums = np.array([sum(x) for x in by.values()])
    counts = np.array([len(x) for x in by.values()])
    ids = np.random.default_rng(0).integers(0, len(sums), (repeats, len(sums)))
    boot = sums[ids].sum(axis=1) / counts[ids].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95_lecture_bootstrap": np.percentile(boot, [2.5, 97.5]).tolist()}


def check_controlled(legacy: dict, balanced: dict, allow_path_a_change: bool = False) -> dict:
    """Prove that only the visual scan differs between the two pools of each question.

    Raises if a non-experimental factor differs. Path A frames do not depend on
    the scan, so a Path A difference is also an error unless explicitly allowed
    (it would mean a second factor changed).
    """
    path_a_identical = 0
    for q in legacy:
        a, b = legacy[q], balanced[q]
        fixed = ("video_id", "question", "options", "gold_option_index", "duration_s")
        for f in fixed:
            if getattr(a, f) != getattr(b, f):
                raise ValueError(f"{q}: {f} differs between pools; the comparison would be confounded")
        if [(s.id, s.start_s, s.end_s, s.text) for s in a.transcript] !=                 [(s.id, s.start_s, s.end_s, s.text) for s in b.transcript]:
            raise ValueError(f"{q}: retained transcript differs between pools")
        pa = lambda p: sorted((c.digest, c.time_s) for c in p.candidates if "A" in c.paths)  # noqa: E731
        same = pa(a) == pa(b)
        if not same and not allow_path_a_change:
            raise ValueError(f"{q}: Path A frames differ between pools; a second factor changed")
        path_a_identical += same
    return {"questions": len(legacy), "fixed_fields_identical": True,
            "path_a_frames_identical_questions": path_a_identical,
            "changed_factor": "Path B scan policy (and therefore the merged pool)"}


def evaluate(run: Path, teacher: CachedTeacher, evidence: dict, pools: dict) -> tuple[dict, dict]:
    """Per-question outcomes and a policy summary for one pool directory."""
    per_q, recall_pool, recall_sel = {}, [], {s.name: [] for s in SELECTORS}
    for q in sorted(pools):
        pool, ev = pools[q], evidence.get(q, [])
        row = {"video_id": pool.video_id, "pool_size": len(pool.candidates),
               "pool_has_evidence": any(_inside(c.time_s, ev) for c in pool.candidates)}
        recall_pool.append(row["pool_has_evidence"])
        for s in SELECTORS:
            chosen = s.select(pool, K)
            row[s.name] = teacher.quality(pool, chosen)
            hit = any(_inside(c.time_s, ev) for c in chosen)
            row[f"{s.name}_evidence"] = hit
            recall_sel[s.name].append(hit)
        per_q[q] = row
    stage = {k: float(np.median([p.timings.get(k, 0.0) for p in pools.values()])) for k in LABEL_FREE_STAGES}
    summary = {
        "questions": len(per_q),
        "accuracy": {s.name: float(np.mean([r[s.name] for r in per_q.values()])) for s in SELECTORS},
        "pool_evidence_recall": float(np.mean(recall_pool)),
        "selection_evidence_recall": {k: float(np.mean(v)) for k, v in recall_sel.items()},
        "pool_size_median": float(np.median([r["pool_size"] for r in per_q.values()])),
        "stage_seconds_median": stage,
        "label_free_selector_cost_s": sum(stage.values()),
        "source_run": str(run),
    }
    return per_q, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy", default="runs/v2_main")
    p.add_argument("--balanced", default="runs/v2_balanced")
    p.add_argument("--out", default="reports/v2_balanced")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from videoqa.answerer import make_answerer  # noqa: PLC0415 - GPU import only when needed

    cfg = load_config(a.config)
    ans = cfg["answerer"]
    teacher_id = f"{ans['model_id']}@{ans.get('revision')}"
    legacy_pools, legacy_meta = _pools(Path(a.legacy))
    balanced_pools, balanced_meta = _pools(Path(a.balanced))
    common = sorted(set(legacy_pools) & set(balanced_pools))
    if len(common) < len(legacy_pools):
        print(f"WARNING: {len(common)}/{len(legacy_pools)} questions have both pools; comparing those only")
    evidence = _evidence(legacy_meta)
    controls = check_controlled({q: legacy_pools[q] for q in common}, {q: balanced_pools[q] for q in common})

    results, per_policy = {}, {}
    t0 = time.perf_counter()
    calls = 0
    for name, run, pools in (("legacy", Path(a.legacy), legacy_pools), ("balanced", Path(a.balanced), balanced_pools)):
        # Read the run's own cache; new calls go to a cache in the report's work area only.
        # The run's own cache was built from these exact pools, so it is attached read-only.
        teacher = CachedTeacher(None, teacher_id, out / f"teacher_cache_{name}.jsonl",
                                identity=answerer_identity(cfg), legacy_paths=(run / "teacher_cache.jsonl",))
        pools = {q: pools[q] for q in common}
        missing = any(teacher.cached_prediction(pl, s.select(pl, K)) is None
                      for pl in pools.values() for s in SELECTORS)
        if missing and teacher.answerer is None:
            teacher.answerer = make_answerer(cfg)
        per_policy[name], results[name] = evaluate(run, teacher, evidence, pools)
        results[name]["fresh_calls"] = teacher.calls
        results[name]["answer_seconds"] = teacher.seconds
        calls += teacher.calls

    videos = {q: per_policy["legacy"][q]["video_id"] for q in common}
    results["balanced_minus_legacy"] = {
        s.name: _bootstrap({q: per_policy["balanced"][q][s.name] - per_policy["legacy"][q][s.name]
                            for q in common}, videos) for s in SELECTORS}
    results["balanced_minus_legacy"]["pool_evidence_recall"] = _bootstrap(
        {q: float(per_policy["balanced"][q]["pool_has_evidence"]) - float(per_policy["legacy"][q]["pool_has_evidence"])
         for q in common}, videos)
    results["pool_configs"] = {"legacy": legacy_meta.get("pool_config", "legacy (pre-flag run)"),
                               "balanced": balanced_meta.get("pool_config")}
    results["controls"] = controls
    results["answerer_identity"] = answerer_identity(cfg)
    results["total_fresh_calls"] = calls
    results["wall_seconds"] = time.perf_counter() - t0
    results["notes"] = ["Pool stage costs are medians of per-question stage timings, summed: a composed estimate.",
                        "Evidence intervals are the generator's 30 s mining windows, a coarse proxy for evidence.",
                        "Development data (197 synthetic MIT questions), not an external benchmark."]
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out / "per_question.json").write_text(json.dumps(per_policy, indent=1), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("legacy", "balanced", "balanced_minus_legacy")}, indent=1))


if __name__ == "__main__":
    main()
