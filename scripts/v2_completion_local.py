"""Do LOCAL board features help choose the fourth frame?  (no new answerer calls)

Uses the exhaustive fourth-frame completion table (runs/v2_completion): every
policy's four-frame QA accuracy is looked up exactly. Only the representation
changes; labels, loss, ridge grid, nested leave-one-lecture-out and fallback are
those of the global completion head (src/videoqa/v2/completion.py).

Declared before the analysis was run (docs/v2/novelty_check.md):
  primary      global+local head - relevance rank-4 (lecture-bootstrap 95% CI)
  attribution  global+local head - global head (same labels, same loss)
  reported     local-only head; zero-shot local rules (exploratory)

    python scripts/v2_completion_local.py features --out runs/v2_completion/local_features.json
    python scripts/v2_completion_local.py analyze  --features runs/v2_completion/local_features.json \
        --out reports/v2_completion_local
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from videoqa.v2.candidates import load_pool
from videoqa.v2.completion import anchor_and_rest, build_questions, nested_cv
from videoqa.v2.local_features import LOCAL_FEATURES, local_features, tile_images

SOURCE_RUN = Path("runs/v2_main")


def _pools() -> dict:
    meta = json.loads((SOURCE_RUN / "pools.json").read_text(encoding="utf-8"))
    return {q: load_pool(SOURCE_RUN / "pools", q) for q in meta["qa_ids"]}


def stage_features(a) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415 - GPU import only here

    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    out, enc_s, feat_s = {}, [], []
    for q, pool in sorted(_pools().items()):
        anchor, rest = anchor_and_rest(pool)
        t0 = time.perf_counter()
        tiles = {c.id: enc.embed_images(tile_images(c.image)) for c in pool.candidates}
        enc_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        anc = [pool.candidates[i] for i in anchor]
        out[q] = [local_features(pool.candidates[i].image, tiles[pool.candidates[i].id], [c.image for c in anc],
                                 [tiles[c.id] for c in anc], pool.question_emb) for i in rest]
        feat_s.append(time.perf_counter() - t0)
    Path(a.out).write_text(json.dumps({"features": list(LOCAL_FEATURES), "rows": out,
                                       "tile_encode_seconds_median_per_question": float(np.median(enc_s)),
                                       "ink_feature_seconds_median_per_question": float(np.median(feat_s))}),
                           encoding="utf-8")
    print(f"{len(out)} questions; tile encoding median {np.median(enc_s):.3f}s, ink features {np.median(feat_s):.3f}s")


def _boot(per_q: dict, a: str, b: str, videos: dict, repeats: int = 5000) -> dict:
    by: dict[str, list[float]] = {}
    for q, v in videos.items():
        by.setdefault(v, []).append(per_q[a][q] - per_q[b][q])
    sums = np.array([sum(x) for x in by.values()])
    counts = np.array([len(x) for x in by.values()])
    ids = np.random.default_rng(0).integers(0, len(sums), (repeats, len(sums)))
    boot = sums[ids].sum(axis=1) / counts[ids].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95_lecture_bootstrap": np.percentile(boot, [2.5, 97.5]).tolist()}


def stage_analyze(a) -> None:
    rows = [json.loads(s) for s in Path(a.rows).read_text(encoding="utf-8").splitlines() if s.strip()]
    feats = json.loads(Path(a.features).read_text(encoding="utf-8"))
    local = {q: np.asarray(v, dtype=float) for q, v in feats["rows"].items()}
    pools = _pools()
    designs = {
        "global": build_questions(pools, rows),
        "global_local": build_questions(pools, rows, local),
    }
    # Local-only: replace the global block by zeros so only local columns can move the ranking.
    lo = build_questions(pools, rows, local)
    ng = designs["global"][0].z.shape[1]
    for q in lo:
        q.z = q.z.copy()
        q.z[:, :ng] = 0.0
    designs["local_only"] = lo

    per_q: dict[str, dict[str, float]] = {}
    folds = {}
    for name, qs in designs.items():
        picks, folds[name] = nested_cv(qs)
        per_q[name] = {q.qa_id: q.correct[picks[q.qa_id]] for q in qs}
    ref = designs["global"]
    videos = {q.qa_id: q.video_id for q in ref}
    per_q["relevance_rank4"] = {q.qa_id: q.correct[0] for q in ref}
    per_q["oracle"] = {q.qa_id: float(q.correct.max()) for q in ref}
    # Zero-shot local rules (exploratory): pick the max of one local feature.
    for k, f in enumerate(LOCAL_FEATURES[:6]):
        per_q[f"rule_{f}"] = {q.qa_id: q.correct[int(np.argmax(local[q.qa_id][:, k]))] for q in ref}

    summary = {
        "questions": len(ref),
        "accuracy": {k: float(np.mean(list(v.values()))) for k, v in per_q.items()},
        "paired": {f"{x}_minus_{y}": _boot(per_q, x, y, videos) for x, y in (
            ("global_local", "relevance_rank4"), ("global_local", "global"),
            ("local_only", "relevance_rank4"), ("global", "relevance_rank4"))},
        "fallback_folds": {k: sum(f["ridge"] is None for f in v) for k, v in folds.items()},
        "local_features": feats["features"],
        "local_feature_cost_s_median_per_question": {
            "tile_encoding": feats["tile_encode_seconds_median_per_question"],
            "ink_maps": feats["ink_feature_seconds_median_per_question"]},
        "teacher_calls": 0,
        "notes": ["Exact four-frame accuracy from the exhaustive completion table (no new answerer calls).",
                  "Development data: 197 synthetic MIT questions already used for several decisions."],
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("accuracy", "paired", "fallback_folds")}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("features", "analyze"))
    p.add_argument("--out", default=None)
    p.add_argument("--features", default="runs/v2_completion/local_features.json")
    p.add_argument("--rows", default="runs/v2_completion/rows_completion.jsonl")
    a = p.parse_args()
    if a.out is None:
        a.out = a.features if a.stage == "features" else "reports/v2_completion_local"
    {"features": stage_features, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
