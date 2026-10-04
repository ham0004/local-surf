"""Fourth-frame completion experiment (see src/videoqa/v2/completion.py).

Declared before any label was collected (2026-10-04):

  anchor     top-3 frames by zero-shot transcript relevance (the strongest simple
             baseline, 38.6% at K=4 on these 197 questions)
  labels     R(T, anchor + {c}) for EVERY other pool candidate c, plus R(T, anchor)
             as a reference; exhaustive, so every fourth-frame policy is scored
             exactly from the table
  budget     1,697 answerer calls (197 + 1,500), vs 1,770 spent on the existing
             independent single-frame labels: an equal-or-smaller call budget
  primary    completion head - relevance rank-4, four-frame accuracy,
             95% lecture-bootstrap CI (20 lectures)
  secondary  completion head - residual scorer trained on single-frame labels
             (same pools, same anchor, larger label budget)
  reported   oracle fourth frame (headroom), CLIP / MMR / expected-random picks,
             and how often the fourth frame breaks or rescues the anchor's answer

Pools, transcripts and the answerer are the frozen ones of runs/v2_main. New
calls go to a NEW run directory; the main run's cache is read, never written.

    python scripts/v2_completion.py label   --run runs/v2_completion
    python scripts/v2_completion.py analyze --run runs/v2_completion --out reports/v2_completion
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
from videoqa.v2.completion import (ANCHOR_SIZE, anchor_and_rest, build_questions, completion_feature_names,
                                   nested_cv, validate_question_rows)
from videoqa.v2.pilot import Budget
from videoqa.v2.teacher import CachedTeacher, answerer_identity

SOURCE_RUN = Path("runs/v2_main")
GPU_BUDGET_S = 2 * 3600
K_FRAMES = ANCHOR_SIZE + 1


def _pools(run: Path) -> dict:
    meta = json.loads((run / "pools.json").read_text(encoding="utf-8"))
    return {q: load_pool(run / "pools", q) for q in meta["qa_ids"]}


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["qa_id"], []).append(r)
    return by


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


# ---------------------------------------------------------------------------
# Stage: label (GPU)
# ---------------------------------------------------------------------------


def stage_label(a) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"source_run": str(SOURCE_RUN), "anchor": f"top-{ANCHOR_SIZE} zero-shot relevance",
                "k_frames": K_FRAMES, "config": a.config}
    mpath = run / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise ValueError("run directory belongs to a different completion experiment")
    mpath.write_text(json.dumps(manifest, indent=1))

    from videoqa.answerer import make_answerer  # noqa: PLC0415 - GPU import only for this stage

    cfg = load_config(a.config)
    ans = cfg["answerer"]
    teacher_id = f"{ans['model_id']}@{ans.get('revision')}"
    # Same pools as the source run: its cache is attached read-only; new records go to this run.
    teacher = CachedTeacher(make_answerer(cfg), teacher_id, run / "teacher_cache.jsonl",
                            identity=answerer_identity(cfg), legacy_paths=(SOURCE_RUN / "teacher_cache.jsonl",))

    pools, budget = _pools(SOURCE_RUN), Budget(run, GPU_BUDGET_S)
    # Resolved identities (not just a config path), appended once per labelling session.
    with open(run / "resolved_sessions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "answerer": answerer_identity(cfg),
                             "source_pools_json_sha256": _sha256(SOURCE_RUN / "pools.json"),
                             "source_cache_sha256": _sha256(SOURCE_RUN / "teacher_cache.jsonl")}) + "\n")
    out = run / "rows_completion.jsonl"
    # A question counts as done only if its rows are complete and consistent;
    # a partial question (interrupted write) stops the run instead of being skipped.
    done = set()
    for q, rows in _group(_read_jsonl(out)).items():
        validate_question_rows(pools[q], rows)
        done.add(q)
    t0 = time.perf_counter()
    for n, q in enumerate(sorted(pools)):
        if q in done:
            continue
        if budget.remaining - (time.perf_counter() - t0) < 0:
            print("budget exhausted; progress saved")
            break
        pool = pools[q]
        anchor, rest = anchor_and_rest(pool)
        s3 = [pool.candidates[i] for i in anchor]
        before = teacher.quality(pool, s3)
        rows = [{"qa_id": q, "video_id": pool.video_id, "anchor_ids": [c.id for c in s3],
                 "candidate_id": pool.candidates[i].id, "baseline_rank": r,
                 "before": before, "after": teacher.quality(pool, s3 + [pool.candidates[i]])}
                for r, i in enumerate(rest)]
        for r in rows:
            r["gain"] = r["after"] - r["before"]
        with open(out, "a", encoding="utf-8") as fh:      # whole question at once: resumable
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(pools)} calls={teacher.calls} hits={teacher.hits} "
                  f"{teacher.seconds / max(teacher.calls, 1):.2f}s/call", flush=True)
    budget.charge("label", time.perf_counter() - t0)
    print(json.dumps({"fresh_calls": teacher.calls, "cache_hits": teacher.hits,
                      "answer_seconds": teacher.seconds, "budget": budget.state}, indent=1))


# ---------------------------------------------------------------------------
# Stage: analyze (CPU, no teacher calls)
# ---------------------------------------------------------------------------


def paired(per_q: dict, arm: str, ref: str, videos: dict, repeats: int = 5000) -> dict:
    """Paired difference with a lecture (video) bootstrap."""
    by: dict[str, list[float]] = {}
    for q, v in videos.items():
        by.setdefault(v, []).append(per_q[arm][q] - per_q[ref][q])
    sums = np.array([sum(x) for x in by.values()])
    counts = np.array([len(x) for x in by.values()])
    ids = np.random.default_rng(0).integers(0, len(sums), (repeats, len(sums)))
    boot = sums[ids].sum(axis=1) / counts[ids].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95_lecture_bootstrap": np.percentile(boot, [2.5, 97.5]).tolist()}


def stage_analyze(a) -> None:
    run, outdir = Path(a.run), Path(a.out)
    rows = _read_jsonl(run / "rows_completion.jsonl")
    pools = _pools(SOURCE_RUN)
    questions = build_questions(pools, rows)
    if len(questions) != len(pools):
        print(f"WARNING: only {len(questions)}/{len(pools)} questions labelled; results are partial")
    picks, folds = nested_cv(questions)

    # The residual scorer trained on single-frame labels (exported out-of-fold scores).
    residual = {r["qa_id"]: r["full_pool_scores"]["residual_base"]
                for r in _read_jsonl(Path(a.residual_predictions))}

    per_q: dict[str, dict[str, float]] = {k: {} for k in (
        "relevance_rank4", "clip_best", "mmr_next", "expected_random", "single_frame_residual",
        "completion_head", "oracle", "three_frame_reference")}
    videos, outcomes = {}, []
    for q in questions:
        pool = pools[q.qa_id]
        anchor, rest = anchor_and_rest(pool)
        cands = [pool.candidates[i] for i in rest]
        clip = [c.clip_sim for c in cands]
        red = [max(float(np.asarray(c.emb) @ np.asarray(pool.candidates[j].emb)) for j in anchor) for c in cands]
        mmr = [0.7 * s - 0.3 * r for s, r in zip(clip, red, strict=True)]
        res = [residual[q.qa_id][c.id] for c in cands]
        best = lambda s: int(min(range(len(s)), key=lambda i: (-s[i], i)))  # noqa: E731
        y = q.correct
        per_q["relevance_rank4"][q.qa_id] = y[0]
        per_q["clip_best"][q.qa_id] = y[best(clip)]
        per_q["mmr_next"][q.qa_id] = y[best(mmr)]
        per_q["expected_random"][q.qa_id] = float(y.mean())
        per_q["single_frame_residual"][q.qa_id] = y[best(res)]
        per_q["completion_head"][q.qa_id] = y[picks[q.qa_id]]
        per_q["oracle"][q.qa_id] = float(y.max())
        per_q["three_frame_reference"][q.qa_id] = q.three_frame
        videos[q.qa_id] = q.video_id
        outcomes.append({"qa_id": q.qa_id, "video_id": q.video_id, "three_frame": q.three_frame,
                         "completions": dict(zip(q.rest_ids, y.tolist(), strict=True)),
                         "picked": {"completion_head": q.rest_ids[picks[q.qa_id]],
                                    "relevance_rank4": q.rest_ids[0]}})

    s3_right = [q for q in questions if q.three_frame == 1.0]
    s3_wrong = [q for q in questions if q.three_frame == 0.0]
    summary = {
        "questions": len(questions), "lectures": len(set(videos.values())),
        "label_rows": len(rows), "k_frames": K_FRAMES,
        "accuracy": {k: float(np.mean(list(v.values()))) for k, v in per_q.items()},
        "correct": {k: float(np.sum(list(v.values()))) for k, v in per_q.items()},
        "paired": {f"{a_}_minus_{b_}": paired(per_q, a_, b_, videos) for a_, b_ in (
            ("completion_head", "relevance_rank4"), ("completion_head", "single_frame_residual"),
            ("single_frame_residual", "relevance_rank4"), ("oracle", "relevance_rank4"),
            ("mmr_next", "relevance_rank4"), ("clip_best", "relevance_rank4"))},
        "fourth_frame_effects": {
            "anchor_correct": len(s3_right),
            "anchor_correct_and_some_fourth_frame_breaks_it": sum(bool((q.correct == 0).any()) for q in s3_right),
            "anchor_correct_and_every_fourth_frame_breaks_it": sum(bool((q.correct == 0).all()) for q in s3_right),
            "anchor_wrong": len(s3_wrong),
            "anchor_wrong_and_some_fourth_frame_rescues_it": sum(bool((q.correct == 1).any()) for q in s3_wrong),
            "questions_where_the_choice_matters": sum(bool(q.correct.min() != q.correct.max()) for q in questions),
        },
        "ridge_selected_per_fold": [f["ridge"] for f in folds],
        "features": list(completion_feature_names()),
        "label_rows_sha256": hashlib.sha256((run / "rows_completion.jsonl").read_bytes()).hexdigest(),
        "notes": ["expected_random is the exact mean over completions, not a sampled run.",
                  "three_frame_reference uses only 3 frames; it is not a K=4 arm.",
                  "single_frame_residual was trained on 1,770 single-frame calls; the completion head on 1,697.",
                  "Development data: 197 synthetic MIT questions already studied; not an external test."],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (outdir / "folds.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")
    (outdir / "per_question.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outcomes), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("accuracy", "paired", "fourth_frame_effects")}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("label", "analyze"))
    p.add_argument("--run", default="runs/v2_completion")
    p.add_argument("--out", default="reports/v2_completion")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    p.add_argument("--residual-predictions", default="reports/v2_review/residual_probe_predictions.jsonl")
    a = p.parse_args()
    {"label": stage_label, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
