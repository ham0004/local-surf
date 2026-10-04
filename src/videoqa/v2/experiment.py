"""Main v2 experiment: speech-vs-frame credit (H1) and cost-matched learned selection.

Two pre-declared questions, answered on lecture questions with leave-one-lecture-out:

  Q1 (H1, the candidate contribution). Head A is trained on SEPARATE interventions:
     "add this transcript line" vs "add this moment's frame". Does its frame-credit
     output pick better frames than (F) zero-shot transcript relevance or (G) its own
     text-credit output? H1 predicts E > F and E > G.

  Q2 (cost-matched selection). Does a learned frame scorer (Head B) beat MobileCLIP
     ranking when it is NOT allowed OCR (so its latency equals similarity ranking)?
     The OCR variant is reported too, with its extra cost.

Arms (identical candidate pool, retained transcript, answerer and K frames):
    transcript_only
    A_mobileclip_topk            similarity ranking
    B_mobileclip_mmr             similarity + diversity
    C_ocr_s{seed}                Head B (single-frame utility labels), with OCR features
    C_noocr_s{seed}              Head B without OCR (still pays for nearby-speech embeddings)
    C_noocr_notext_s{seed}       ...and without explicit speech features in the selector (pool still uses speech)
    E_headA_frame_credit         top-K candidates by trained Head A FRAME credit
    F_relevance                  top-K by zero-shot transcript relevance (Head A untrained)
    G_headA_text_credit          top-K by trained Head A TEXT credit (wrong target, control)

Stages (resumable, one GPU-time budget):
    python -m videoqa.v2.experiment pools  --data data/mit_lectures --qa-file qa_v2.jsonl --run runs/v2_main
    python -m videoqa.v2.experiment label  --run runs/v2_main
    python -m videoqa.v2.experiment eval   --run runs/v2_main
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from ..datasets import load_local_dataset
from .candidates import PoolConfig, _with_neighbours, build_pool, load_pool, save_pool
from .head_a import HotMomentScorer
from .head_b import HeadB, HeadBConfig, featurize
from .labeling import head_a_labels, independent_labels
from .pilot import Budget, _read_rows, _teacher, _write_rows, summarize
from .selectors import MMRSelector, ScoreTopK, SimilarityTopK, UnaryUtility, timed_select

# ---------------------------------------------------------------------------
# Declared settings (fixed before any result of this run)
# ---------------------------------------------------------------------------
K_FRAMES = 4
LABEL_BUDGET = 9                 # Head B: baseline + 8 single-frame additions per question
SEEDS = (0, 1, 2)
GPU_BUDGET_S = 6 * 3600
POOL_CFG = PoolConfig(scan_step_s=5.0, scan_cap=24)     # cheaper visual scan than the pilot (3 s / 60)
HEAD_B_CFG = dict(proj_dim=4, weight_decay=1e-2)        # smaller and more regularised than the pilot
PAIRS = [("E_headA_frame_credit", "F_relevance"), ("E_headA_frame_credit", "G_headA_text_credit"),
         ("E_headA_frame_credit", "A_mobileclip_topk"), ("C_noocr", "A_mobileclip_topk"),
         ("C_ocr", "A_mobileclip_topk"), ("C_noocr", "B_mobileclip_mmr"), ("C_ocr", "C_noocr"),
         ("C_noocr", "C_noocr_notext"), ("A_mobileclip_topk", "transcript_only")]


def _items(data: str, qa_file: str):
    items, _ = load_local_dataset(data, splits=("train", "dev", "calibration", "test"), qa_file=qa_file)
    return sorted(items, key=lambda it: it.qa.qa_id)


def _cfg(run: Path) -> dict:
    return json.loads((run / "pools.json").read_text())


def _pools(run: Path) -> dict:
    return {q: load_pool(run / "pools", q) for q in _cfg(run)["qa_ids"]}


def _nearest_segment(segs, t: float):
    """Segment whose midpoint is closest to time t (for any candidate frame)."""
    return min(segs, key=lambda s: abs(0.5 * (s.start_s + s.end_s) - t))


def _segment_ending_near(segs, t: float):
    """The hot-moment line a Path A frame came from (frames are taken 0.3 s before a line ends)."""
    return min(segs, key=lambda s: abs((s.end_s - 0.3) - t))


# ---------------------------------------------------------------------------
# Stage: pools
# ---------------------------------------------------------------------------


def stage_pools(a) -> None:
    from .encoders import FrozenEncoders  # noqa: PLC0415

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    budget = Budget(run, GPU_BUDGET_S)
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip")
    head_a = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub")       # zero-shot relevance proposals
    meta = _cfg(run) if (run / "pools.json").exists() else {"data": a.data, "qa_file": a.qa_file, "qa_ids": []}
    pool_cfg = dataclasses.replace(POOL_CFG, scan_policy=a.scan_policy)
    previous = meta.get("pool_config", dataclasses.asdict(POOL_CFG))
    if meta["qa_ids"] and previous != dataclasses.asdict(pool_cfg):
        raise ValueError("pool configuration changed; use a new --run directory for a fair candidate-pool experiment")
    meta["pool_config"] = dataclasses.asdict(pool_cfg)
    t0 = time.perf_counter()
    for n, it in enumerate(_items(a.data, a.qa_file)):
        if it.qa.qa_id in meta["qa_ids"]:
            continue
        if budget.remaining - (time.perf_counter() - t0) < 0:
            print("budget exhausted while building pools; progress saved")
            break
        pool = build_pool(it.qa, it.transcript, it.video_path, enc, head_a, pool_cfg)
        save_pool(pool, run / "pools")
        meta["qa_ids"].append(it.qa.qa_id)
        (run / "pools.json").write_text(json.dumps(meta, indent=1))
        if (n + 1) % 20 == 0:
            print(f"pools {n + 1}", flush=True)
    budget.charge("pools", time.perf_counter() - t0)
    print(json.dumps({"pools": len(meta["qa_ids"]), "encoder_seconds": enc.seconds, "budget": budget.state}, indent=1))


# ---------------------------------------------------------------------------
# Stage: label (Head B single-frame labels + Head A separate interventions)
# ---------------------------------------------------------------------------


def stage_label(a) -> None:
    run = Path(a.run)
    budget, pools = Budget(run, GPU_BUDGET_S), _pools(run)
    cfg = _cfg(run)
    items = {it.qa.qa_id: it for it in _items(cfg["data"], cfg["qa_file"])}
    teacher = _teacher(run, a.config)
    done = {r.qa_id for r in _read_rows(run / "rows_independent.jsonl")}
    t0 = time.perf_counter()
    for n, q in enumerate(sorted(pools)):
        if q in done:
            continue
        if budget.remaining - (time.perf_counter() - t0) < 0:
            print(f"budget exhausted after {n} questions; progress saved")
            break
        pool, segs = pools[q], items[q].transcript.segments
        # Head A: separate interventions for each Path A moment (line alone vs frame alone).
        pairs = [(_segment_ending_near(segs, c.time_s), c) for c in pool.candidates if "A" in c.paths]
        ha = head_a_labels(pool, teacher, pairs)
        for r, (seg, _c) in zip(ha, pairs, strict=True):
            r["context"] = _with_neighbours(segs, seg)         # Head A's input text, saved with the label
        with open(run / "head_a_labels.jsonl", "a", encoding="utf-8") as fh:
            for r in ha:
                fh.write(json.dumps(r) + "\n")
        # Head B: single-frame utility labels (written last: marks the question done).
        _write_rows(run / "rows_independent.jsonl", independent_labels(pool, teacher, LABEL_BUDGET))
        if (n + 1) % 20 == 0:
            print(f"labelled {n + 1}/{len(pools)} calls={teacher.calls} hits={teacher.hits} "
                  f"{teacher.seconds / max(teacher.calls, 1):.2f}s/call", flush=True)
    budget.charge("label", time.perf_counter() - t0)
    print(json.dumps({"teacher_calls": teacher.calls, "cache_hits": teacher.hits, "budget": budget.state}, indent=1))


# ---------------------------------------------------------------------------
# Stage: eval (leave-one-lecture-out)
# ---------------------------------------------------------------------------


def _auc(scores, positive) -> float | None:
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def stage_eval(a) -> None:
    run = Path(a.run)
    budget, pools = Budget(run, GPU_BUDGET_S), _pools(run)
    cfg = _cfg(run)
    items = {it.qa.qa_id: it for it in _items(cfg["data"], cfg["qa_file"])}
    teacher = _teacher(run, a.config)
    rows_b = _read_rows(run / "rows_independent.jsonl")
    rows_a = list({(r["qa_id"], r["candidate_id"]): r for r in (
        json.loads(x) for x in (run / "head_a_labels.jsonl").read_text(encoding="utf-8").splitlines() if x)}.values())
    labelled = sorted({r.qa_id for r in rows_b})
    scorer = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub", feature_mode="small")

    # Head A features for every labelled moment, computed once.
    feat_a = np.stack([scorer.small_features(items[r["qa_id"]].qa.question, items[r["qa_id"]].qa.options,
                                             [r["context"]])[0] for r in rows_a])
    y_a = np.array([[r["text_gain"], r["visual_gain"]] for r in rows_a], dtype=np.float32)
    vid_a = np.array([r["video_id"] for r in rows_a])
    grp_a = np.array([r["qa_id"] for r in rows_a])
    pred_a = np.zeros_like(y_a)                                   # out-of-fold predictions (for AUC)

    results, sel_seconds, answer_seconds = [], collections.defaultdict(list), []
    t0 = time.perf_counter()
    for held in sorted({pools[q].video_id for q in labelled}):
        # -- train Head A (small) and Head B (with / without OCR) on the other lectures --
        tr = vid_a != held
        head_a = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub", feature_mode="small")
        head_a._tok, head_a._model, head_a._torch = scorer._tok, scorer._model, scorer._torch   # share backbone
        head_a.fit(feat_a[tr], y_a[tr], grp_a[tr])
        pred_a[~tr] = ((feat_a[~tr] - head_a.feat_mean) / head_a.feat_std) @ head_a.head_w + head_a.head_b
        train_b = [r for r in rows_b if r.video_id != held]
        variants = [(True, True), (False, True), (False, False)]          # (use_ocr, use_transcript)
        heads = {(o, t, s): _train_head_b(train_b, pools, s, o, t) for o, t in variants for s in SEEDS}

        def credit(pool, k, model=head_a):
            """Match Path A's label-generation segment convention at inference."""
            segs = items[pool.qa_id].transcript.segments
            ctx = [_with_neighbours(segs, (_segment_ending_near if "A" in c.paths else _nearest_segment)(segs, c.time_s))
                   for c in pool.candidates]
            sc = model.score(pool.question, ctx, pool.options)
            return {c.id: float(sc[i, k]) for i, c in enumerate(pool.candidates)}

        arms = [SimilarityTopK(), MMRSelector(),
                ScoreTopK(lambda p: credit(p, 1), "E_headA_frame_credit"),
                ScoreTopK(lambda p: {c.id: c.head_a_visual for c in p.candidates}, "F_relevance"),
                ScoreTopK(lambda p: credit(p, 0), "G_headA_text_credit")]
        arms += [UnaryUtility(heads[(True, True, s)], f"C_ocr_s{s}", use_ocr=True) for s in SEEDS]
        arms += [UnaryUtility(heads[(False, True, s)], f"C_noocr_s{s}", use_ocr=False) for s in SEEDS]
        arms += [UnaryUtility(heads[(False, False, s)], f"C_noocr_notext_s{s}", use_ocr=False, use_transcript=False)
                 for s in SEEDS]

        for q in (q for q in labelled if pools[q].video_id == held):
            if budget.remaining - (time.perf_counter() - t0) < 0:
                print("budget exhausted during eval; partial results saved")
                break
            pool = pools[q]
            row = {"qa_id": q, "video_id": held, "transcript_only": teacher.quality(pool, [])}
            for arm in arms:
                chosen, secs = timed_select(arm, pool, K_FRAMES)
                c0, s0 = teacher.calls, teacher.seconds
                row[arm.name] = teacher.quality(pool, chosen)
                if teacher.calls > c0:
                    answer_seconds.append(teacher.seconds - s0)
                sel_seconds[arm.name].append(secs)
            results.append(row)
        print(f"fold {held}: {sum(r['video_id'] == held for r in results)} questions", flush=True)
    budget.charge("eval", time.perf_counter() - t0)

    with open(run / "eval_rows.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    summary = summarize(results, pools, sel_seconds, answer_seconds, pairs=PAIRS, ocr_prefixes=("C_ocr",))
    summary["head_a_out_of_fold_auc"] = {
        name: {"positives": int((y_a[:, k] > 0).sum()),
               "trained_small_head": _auc(pred_a[:, k], y_a[:, k] > 0),
               "zero_shot_relevance": _auc(feat_a[:, 0], y_a[:, k] > 0)}
        for k, name in ((0, "text_gain"), (1, "frame_gain"))}
    tg, vg = y_a[:, 0], y_a[:, 1]
    summary["head_a_label_counts"] = {"moments": len(rows_a), "frame_only": int(((vg > 0) & (tg <= 0)).sum()),
                                      "text_only": int(((tg > 0) & (vg <= 0)).sum()),
                                      "both": int(((tg > 0) & (vg > 0)).sum())}
    summary["budget"] = budget.state
    (run / "eval_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def _train_head_b(rows, pools, seed: int, use_ocr: bool, use_transcript: bool = True) -> HeadB:
    by = {q: {c.id: c for c in p.candidates} for q, p in pools.items()}
    X = np.stack([featurize(pools[r.qa_id], by[r.qa_id][r.candidate_id], [], use_ocr, use_transcript)
                  for r in rows])
    y = np.array([r.gain for r in rows], dtype=np.float32)
    ctx = np.array([r.qa_id for r in rows])
    head = HeadB(HeadBConfig(seed=seed, **HEAD_B_CFG))
    head.fit(X, y, ctx)
    return head


def main() -> None:
    p = argparse.ArgumentParser(prog="videoqa.v2.experiment")
    p.add_argument("stage", choices=["pools", "label", "eval"])
    p.add_argument("--data", default="data/mit_lectures")
    p.add_argument("--qa-file", default="qa_v2.jsonl")
    p.add_argument("--run", default="runs/v2_main")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    p.add_argument("--scan-policy", choices=("legacy", "balanced"), default="legacy",
                   help="candidate-pool experiment only; balanced needs a new --run directory")
    a = p.parse_args()
    {"pools": stage_pools, "label": stage_label, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
