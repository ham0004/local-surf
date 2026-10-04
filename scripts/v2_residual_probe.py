"""Nested video CV on existing random single-frame labels; zero teacher calls.

Produces out-of-fold full-pool selections for a later exact four-frame teacher
evaluation. Reported label metrics are single-frame proxies, NOT four-frame QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from videoqa.schemas import TranscriptSegment
from videoqa.v2.records import FrameCandidate, QuestionPool, UtilityRow
from videoqa.v2.residual_selector import (RIDGES, ResidualSelector, design, feature_names,
                                        mmr_order, pair_blocks, ranked_indices)


def load_saved(run):
    """Load only cached JSON and NPZ; never import encoders or decode images."""
    pools = {}
    meta = json.loads((run / "pools.json").read_text(encoding="utf-8"))
    for q in meta["qa_ids"]:
        directory = run / "pools" / q.replace(":", "_").replace("/", "_")
        value = json.loads((directory / "pool.json").read_text(encoding="utf-8"))
        with np.load(directory / "emb.npz", allow_pickle=False) as cache:
            candidates = []
            for cm in value.pop("candidates"):
                cm["paths"] = tuple(cm["paths"])
                c = FrameCandidate(**cm)
                c.emb, c.ocr_emb, c.near_emb = (cache[f"{c.id}|{key}"].copy()
                                               for key in ("emb", "ocr_emb", "near_emb"))
                candidates.append(c)
            value["transcript"] = [TranscriptSegment(**s) for s in value["transcript"]]
            pool = QuestionPool(**value, candidates=candidates)
            pool.question_emb = cache["question"].copy()
            pools[q] = pool
    rows = [UtilityRow(**json.loads(s)) for s in (run / "rows_independent.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    return pools, rows


def _by_question(rows):
    by = {}
    for row in rows:
        by.setdefault(row.qa_id, []).append(row)
    return by


def subset_metric(pool, rows, scores=None, order=None, k=4):
    """Selection is restricted to the random labeled subset for proxy metrics."""
    lookup = {c.id: i for i, c in enumerate(pool.candidates)}
    labels = {lookup[r.candidate_id]: float(r.after) for r in rows}
    if order is None:
        order = ranked_indices(pool, scores, labels)
    selected = [i for i in order if i in labels][:k]
    if not selected:
        raise ValueError("Empty labeled subset")
    # For MMR, the declared sequential selection ordering provides its rank.
    rank_scores = scores if scores is not None else {i: -n for n, i in enumerate(order)}
    concordance = []
    indices = sorted(labels)
    for n, i in enumerate(indices):
        for j in indices[n + 1:]:
            if labels[i] != labels[j]:
                delta = (rank_scores[i] - rank_scores[j]) * (labels[i] - labels[j])
                concordance.append(float(delta > 0) + 0.5 * float(delta == 0))
    return {"topk_single_frame_mean": float(np.mean([labels[i] for i in selected])),
            "pairwise_concordance": float(np.mean(concordance)) if concordance else None,
            "unequal_label_pairs": len(concordance), "labeled_candidates": len(labels),
            "selected_count": len(selected),
            "selected_ids": [pool.candidates[i].id for i in selected],
            "oracle_topk_single_frame_mean": float(np.mean(sorted(labels.values(), reverse=True)[:k]))}


def choose_ridge(pools, train_rows, blocks, context, k=4):
    """Inner leave-one-video-out using only the outer training data.

    Equal-video average of equal-question top-k single-frame label means chooses
    the hyperparameter. Exact ties prefer CLIP, then stronger regularization.
    """
    videos = sorted({r.video_id for r in train_rows})
    candidates = (None,) + tuple(sorted(RIDGES, reverse=True))
    values = {str(r): [] for r in candidates}
    if len(videos) < 2:
        return None, {"reason": "too_few_training_videos", "videos": videos}
    grouped = _by_question(train_rows)
    cached_designs = {q: design(pools[q], context) for q in grouped}
    for held in videos:
        inner_blocks = [b for b in blocks if b.video_id != held]
        for ridge in candidates:
            model = ResidualSelector.fit(inner_blocks, context, ridge)
            qvalues = [subset_metric(pools[q], rr, cached_designs[q][1] + cached_designs[q][0] @ model.weights, k=k)["topk_single_frame_mean"]
                       for q, rr in grouped.items() if pools[q].video_id == held]
            values[str(ridge)].append(float(np.mean(qvalues)))
    means = {key: float(np.mean(value)) for key, value in values.items()}
    # Explicit stable preference order, with a tolerance only for floating error.
    best = candidates[0]
    for ridge in candidates[1:]:
        if means[str(ridge)] > means[str(best)] + 1e-12:
            best = ridge
    return best, {"videos": videos, "mean_by_candidate": means,
                  "inner_video_scores": values, "objective": "equal-video top-k single-frame mean",
                  "tie_preference": ["MobileCLIP fallback", "100", "10", "1", "0.1"]}


def fit_outer(pools, rows, held, context, k=4, prepared_blocks=None):
    train = [r for r in rows if r.video_id != held]
    blocks = (pair_blocks(pools, train, context) if prepared_blocks is None else
              [b for b in prepared_blocks if b.video_id != held])
    ridge, audit = choose_ridge(pools, train, blocks, context, k)
    model = ResidualSelector.fit(blocks, context, ridge)
    return model, {"held_video": held, "train_videos": sorted({r.video_id for r in train}),
                   "selected_ridge": ridge, "inner_cv": audit, "model": model.to_dict()}


def bootstrap_differences(records, arm, baseline, metric, rng, repetitions):
    groups = {}
    for row in records:
        a, b = row["subset_metrics"][arm][metric], row["subset_metrics"][baseline][metric]
        if a is not None and b is not None:
            groups.setdefault(row["video_id"], []).append(a - b)
    if not groups:
        return None
    videos = sorted(groups)
    sums = np.array([sum(groups[v]) for v in videos])
    counts = np.array([len(groups[v]) for v in videos])
    samples = rng.integers(0, len(videos), size=(repetitions, len(videos)))
    diffs = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95_video_bootstrap": np.percentile(diffs, [2.5, 97.5]).tolist(),
            "questions": int(counts.sum()), "videos": len(videos), "replicates": repetitions}


def run_probe(pools, rows, k=4, bootstraps=2000, progress=None):
    if k < 1 or bootstraps < 1:
        raise ValueError("k and bootstrap count must be positive")
    grouped = _by_question(rows)
    blocks = {c: pair_blocks(pools, rows, c) for c in (False, True)}
    videos = sorted({r.video_id for r in rows})
    records, folds = [], []
    for held in videos:
        models = {}
        fold = {"held_video": held, "arms": {}}
        for context, arm in ((False, "residual_base"), (True, "residual_context")):
            models[arm], audit = fit_outer(pools, rows, held, context, k, blocks[context])
            fold["arms"][arm] = audit
        folds.append(fold)
        for q in sorted(q for q in grouped if pools[q].video_id == held):
            pool, rr = pools[q], grouped[q]
            scores = {"clip": design(pool)[1], "relevance": np.array([c.head_a_visual for c in pool.candidates])}
            scores.update({arm: m.scores(pool) for arm, m in models.items()})
            metrics = {arm: subset_metric(pool, rr, ss, k=k) for arm, ss in scores.items()}
            subset_indices = [i for i, c in enumerate(pool.candidates) if c.id in {r.candidate_id for r in rr}]
            metrics["mmr"] = subset_metric(pool, rr, order=mmr_order(pool, subset_indices), k=k)
            full_order = {arm: ranked_indices(pool, ss) for arm, ss in scores.items()}
            full_order["mmr"] = mmr_order(pool)
            records.append({"qa_id": q, "video_id": held, "subset_metrics": metrics,
                            "full_pool_selected_ids": {arm: [pool.candidates[i].id for i in order[:k]]
                                                       for arm, order in full_order.items()},
                            "full_pool_scores": {arm: {c.id: float(s) for c, s in zip(pool.candidates, ss, strict=True)}
                                                 for arm, ss in scores.items()},
                            "candidate_count": len(pool.candidates)})
        if progress:
            progress(fold, len(folds), len(videos))
    arms = ("clip", "mmr", "relevance", "residual_base", "residual_context")
    metrics = ("topk_single_frame_mean", "pairwise_concordance")
    aggregate = {}
    for arm in arms:
        aggregate[arm] = {}
        for metric in metrics:
            observed = [r["subset_metrics"][arm][metric] for r in records if r["subset_metrics"][arm][metric] is not None]
            aggregate[arm][metric] = {"mean": float(np.mean(observed)) if observed else None,
                                      "questions": len(observed)}
    rng = np.random.default_rng(0)
    pairs = [(a, "clip") for a in arms if a != "clip"] + [("residual_context", "residual_base")]
    paired = {f"{a}_minus_{b}": {m: bootstrap_differences(records, a, b, m, rng, bootstraps) for m in metrics}
              for a, b in pairs}
    summary = {"status": "exploratory engineering prototype; no novelty or performance guarantee",
               "protocol": "nested leave-one-video-out; fixed ridge grid and CLIP fallback",
               "metric_warning": "Random labeled subset only. Mean correctness of separately evaluated single frames, NOT four-frame QA accuracy.",
               "k": k, "questions": len(records), "videos": len(videos), "label_rows": len(rows),
               "unlabeled_pool_questions_not_scored": sorted(set(pools) - set(grouped)),
               "random_subset_provenance": "runs rows_independent.jsonl; original independent_labels shuffles candidates per question with seed 0",
               "features": {"base": list(feature_names(False)), "context": list(feature_names(True))},
               "pair_weighting": "all unordered pairs; mean within question then mean across questions; L2 residual norm",
               "oracle_topk_single_frame_mean": float(np.mean([r["subset_metrics"]["clip"]["oracle_topk_single_frame_mean"] for r in records])),
               "metrics": aggregate, "paired_differences": paired,
               "teacher_calls": 0, "fresh_encoder_calls": 0,
               "limitations": ["Cached synthetic MIT questions; prior exploration means nested CV is not a new external test.",
                               "Single-frame utility does not measure four-frame interactions.",
                               "Lexical/numeral differences are coverage signals, not verified contradictions.",
                               "Bootstrap intervals describe clustered out-of-fold differences and do not refit nested models."]}
    return summary, records, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/v2_main"))
    parser.add_argument("--output", type=Path, default=Path("reports/v2_review"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    notes = args.output / "residual_probe_notes.md"
    notes.write_text("# Residual probe\n\nCPU-only nested video CV on existing random single-frame labels. "
                     "These are proxy ranking metrics, not four-frame QA accuracy. No teacher or encoder calls. "
                     "Full-pool selections are exported for later exact evaluation.\n", encoding="utf-8")
    started = time.perf_counter()
    pools, rows = load_saved(args.run)
    def progress(fold, done, total):
        print(f"Outer video {done}/{total}: {fold['held_video']}", flush=True)
    summary, records, folds = run_probe(pools, rows, args.k, args.bootstraps, progress)
    summary["elapsed_cpu_wall_seconds"] = time.perf_counter() - started
    summary["input_run"] = str(args.run.resolve())
    summary["input_labels_sha256"] = hashlib.sha256((args.run / "rows_independent.jsonl").read_bytes()).hexdigest()
    for name, value in (("summary", summary), ("folds", folds)):
        (args.output / f"residual_probe_{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "residual_probe_predictions.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    notes.write_text(notes.read_text(encoding="utf-8") + "\nCompleted nested CV. "
                     f"{summary['questions']} questions, {summary['videos']} videos, {summary['label_rows']} single-frame labels.\n\n"
                     + "\n".join(f"- {a}: mean selected single-frame correctness "
                                  f"{m['topk_single_frame_mean']['mean']:.4f}; pairwise concordance "
                                  f"{m['pairwise_concordance']['mean']:.4f}" for a, m in summary["metrics"].items())
                     + "\n\nThese exploratory results carry no novelty or performance promise.\n", encoding="utf-8")
    print(json.dumps({"questions": summary["questions"], "videos": summary["videos"],
                      "metrics": summary["metrics"], "seconds": summary["elapsed_cpu_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
