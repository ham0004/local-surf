"""The controlled pilot for one hypothesis:

    A small selector trained on context-dependent evidence utility chooses better
    frames than similarity ranking or independently scored frames, under the same
    teacher-labelling budget and measured inference cost.

Stages (each resumable; all GPU time is charged to one budget file):

    pools    build every question's fixed pool (both paths, frozen features)
    audit    ~50 questions: single-frame coverage, non-zero/harmful utility, determinism
    label    independent vs prefix labels for every question, at the SAME call budget
    eval     leave-one-lecture-out: train C and D on 4 lectures, compare arms on the 5th
    head-a   separate text vs frame interventions for Path A hot moments (if budget remains)

    python -m videoqa.v2.pilot pools  --data data/mit_lectures --run runs/v2_pilot
    python -m videoqa.v2.pilot audit  --run runs/v2_pilot
    python -m videoqa.v2.pilot label  --run runs/v2_pilot
    python -m videoqa.v2.pilot eval   --run runs/v2_pilot
    python -m videoqa.v2.pilot head-a --run runs/v2_pilot
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from ..config import load_config
from ..datasets import load_local_dataset
from .candidates import PoolConfig, build_pool, load_pool, save_pool
from .head_b import HeadB, HeadBConfig, featurize
from .labeling import head_a_labels, independent_labels, prefix_labels
from .records import UtilityRow
from .selectors import GreedyUtility, MMRSelector, SimilarityTopK, UnaryUtility, timed_select

# ---------------------------------------------------------------------------
# Fixed experiment settings (declared before any result)
# ---------------------------------------------------------------------------
K_FRAMES = 4            # final frames given to the answerer, identical for every arm
CHAINS, CHAIN_LEN = 2, 4
LABEL_BUDGET = 1 + CHAINS * CHAIN_LEN      # per-question teacher calls for EACH scheme (= 9)
SEEDS = (0, 1, 2)
GPU_BUDGET_S = 2 * 3600
AUDIT_N = 50


# ---------------------------------------------------------------------------
# GPU-time budget (persisted, so it survives restarts)
# ---------------------------------------------------------------------------


class Budget:
    def __init__(self, run: Path) -> None:
        self.path = run / "budget.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"spent_s": 0.0, "stages": {}}

    @property
    def remaining(self) -> float:
        return GPU_BUDGET_S - self.state["spent_s"]

    def charge(self, stage: str, seconds: float) -> None:
        self.state["spent_s"] += seconds
        self.state["stages"][stage] = self.state["stages"].get(stage, 0.0) + seconds
        self.path.write_text(json.dumps(self.state, indent=1))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _audit_order(qids) -> list[str]:
    """Fixed pseudo-random order (sha256), identical across processes and machines."""
    return sorted(qids, key=lambda q: hashlib.sha256(q.encode()).hexdigest())


def _items(data: str):
    items, _ = load_local_dataset(data, splits=("train", "dev", "calibration", "test"))
    return sorted(items, key=lambda it: it.qa.qa_id)


def _teacher(run: Path, cfg_path: str):
    from ..answerer import make_answerer  # noqa: PLC0415
    from .teacher import CachedTeacher  # noqa: PLC0415

    cfg = load_config(cfg_path)
    ans = make_answerer(cfg)
    a = cfg["answerer"]
    return CachedTeacher(ans, f"{a['model_id']}@{a.get('revision')}", run / "teacher_cache.jsonl")


def _pools(run: Path) -> dict:
    meta = json.loads((run / "pools.json").read_text())
    return {q: load_pool(run / "pools", q) for q in meta["qa_ids"]}


def _write_rows(path: Path, rows) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dataclasses.asdict(r)) + "\n")


def _read_rows(path: Path) -> list[UtilityRow]:
    if not path.exists():
        return []
    return [UtilityRow(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Stage: pools
# ---------------------------------------------------------------------------


def stage_pools(a) -> None:
    from .encoders import FrozenEncoders  # noqa: PLC0415
    from .head_a import HotMomentScorer  # noqa: PLC0415

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    budget = Budget(run)
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip")
    head_a = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub")     # zero-shot relevance (untrained)
    done = json.loads((run / "pools.json").read_text())["qa_ids"] if (run / "pools.json").exists() else []
    t0 = time.perf_counter()
    for n, it in enumerate(_items(a.data)):
        if it.qa.qa_id in done:
            continue
        pool = build_pool(it.qa, it.transcript, it.video_path, enc, head_a, PoolConfig())
        save_pool(pool, run / "pools")
        done.append(it.qa.qa_id)
        (run / "pools.json").write_text(json.dumps({"data": a.data, "qa_ids": done}, indent=1))
        print(f"[{n + 1}] {it.qa.qa_id}: {len(pool.candidates)} candidates "
              f"(A {sum('A' in c.paths for c in pool.candidates)}, B {sum('B' in c.paths for c in pool.candidates)}, "
              f"both {sum(len(c.paths) == 2 for c in pool.candidates)})  "
              + " ".join(f"{k}={v:.2f}s" for k, v in pool.timings.items()), flush=True)
    budget.charge("pools", time.perf_counter() - t0)
    print(json.dumps({"encoder_seconds": enc.seconds, "encoder_counts": enc.counts,
                      "budget_spent_s": budget.state["spent_s"]}, indent=1))


# ---------------------------------------------------------------------------
# Stage: audit
# ---------------------------------------------------------------------------


def stage_audit(a) -> None:
    run = Path(a.run)
    budget, pools = Budget(run), _pools(run)
    teacher = _teacher(run, a.config)
    qids = _audit_order(pools)[:AUDIT_N]
    stats = collections.Counter()
    t0 = time.perf_counter()
    for q in qids:
        if budget.remaining - (time.perf_counter() - t0) < 0:
            print("budget exhausted during audit")
            break
        pool = pools[q]
        rows = independent_labels(pool, teacher, budget=1 + len(pool.candidates))   # every single frame
        base = rows[0].before if rows else teacher.quality(pool, [])
        gains = [r.gain for r in rows]
        stats["questions"] += 1
        stats["candidates"] += len(pool.candidates)
        stats["baseline_correct"] += int(base == 1.0)
        stats["any_positive"] += int(any(g > 0 for g in gains))
        stats["any_negative"] += int(any(g < 0 for g in gains))
        stats["positive_rows"] += sum(g > 0 for g in gains)
        stats["negative_rows"] += sum(g < 0 for g in gains)
        stats["zero_rows"] += sum(g == 0 for g in gains)
    # Determinism: re-ask 20 cached evaluations directly (bypassing the cache).
    agree = 0
    recs = random.Random(0).sample(list(teacher.cache.values()), min(20, len(teacher.cache)))
    for rec in recs:
        pool = pools[rec["qa_id"]]
        frames = [c for c in pool.candidates if c.id in rec["frames"]]
        key_before = len(teacher.cache)
        from .teacher import evidence_key  # noqa: PLC0415

        teacher.cache.pop(evidence_key(pool, frames, teacher.teacher_id))
        agree += int(teacher.quality(pool, frames) == rec["quality"])
        assert len(teacher.cache) == key_before
    spent = time.perf_counter() - t0
    budget.charge("audit", spent)
    out = {**stats, "determinism_agree": f"{agree}/{len(recs)}", "teacher_calls": teacher.calls,
           "teacher_hits": teacher.hits, "seconds_per_call": teacher.seconds / max(teacher.calls, 1),
           "budget_spent_s": budget.state["spent_s"]}
    (run / "audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# Stage: label (both schemes, same budget, every question)
# ---------------------------------------------------------------------------


def stage_label(a) -> None:
    run = Path(a.run)
    budget, pools = Budget(run), _pools(run)
    teacher = _teacher(run, a.config)
    done = {r.qa_id for r in _read_rows(run / "rows_prefix.jsonl")}
    acct = {"independent": [0, 0], "prefix": [0, 0]}      # [real calls, cache hits] attributed per scheme
    t0 = time.perf_counter()
    for n, q in enumerate(sorted(pools)):
        if q in done:
            continue
        if budget.remaining - (time.perf_counter() - t0) < 0:
            print(f"budget exhausted after {n} questions; progress saved")
            break
        for scheme, fn in (("independent", lambda p: independent_labels(p, teacher, LABEL_BUDGET)),
                           ("prefix", lambda p: prefix_labels(p, teacher, CHAINS, CHAIN_LEN))):
            c0, h0 = teacher.calls, teacher.hits
            rows = fn(pools[q])
            acct[scheme][0] += teacher.calls - c0
            acct[scheme][1] += teacher.hits - h0
            _write_rows(run / f"rows_{scheme}.jsonl", rows)
        if (n + 1) % 10 == 0:
            print(f"labelled {n + 1}/{len(pools)}  calls={teacher.calls} hits={teacher.hits} "
                  f"{teacher.seconds / max(teacher.calls, 1):.2f}s/call", flush=True)
    budget.charge("label", time.perf_counter() - t0)
    prev = json.loads((run / "label_accounting.json").read_text()) if (run / "label_accounting.json").exists() else {}
    for s in acct:
        old = prev.get(s, {"real_calls": 0, "cache_hits": 0})
        acct[s] = {"real_calls": old["real_calls"] + acct[s][0], "cache_hits": old["cache_hits"] + acct[s][1]}
    acct["budget_per_question_each_scheme"] = LABEL_BUDGET
    acct["note"] = ("calls are attributed to the scheme that issued them first; the independent scheme runs "
                    "first, so the prefix scheme gets the shared baseline as a cache hit")
    (run / "label_accounting.json").write_text(json.dumps(acct, indent=1))
    print(json.dumps({**acct, "budget_spent_s": budget.state["spent_s"]}, indent=1))


# ---------------------------------------------------------------------------
# Stage: eval (leave-one-lecture-out)
# ---------------------------------------------------------------------------


def _train_head(rows: list[UtilityRow], pools: dict, seed: int) -> HeadB:
    by = {q: {c.id: c for c in p.candidates} for q, p in pools.items()}
    X = np.stack([featurize(pools[r.qa_id], by[r.qa_id][r.candidate_id], [by[r.qa_id][h] for h in r.history])
                  for r in rows])
    y = np.array([r.gain for r in rows], dtype=np.float32)
    ctx = np.array([f"{r.qa_id}|{','.join(r.history)}" for r in rows])
    head = HeadB(HeadBConfig(seed=seed))
    head.fit(X, y, ctx)
    return head


def _ocr_transcript(pool, frames):
    """Control: retained transcript + the OCR text of the chosen frames, and NO images."""
    from ..schemas import TranscriptSegment  # noqa: PLC0415

    extra = [TranscriptSegment(f"ocr{i}", f.time_s, f.time_s, f"[text on screen at {f.time_s:.0f}s] {f.ocr_text}")
             for i, f in enumerate(frames) if f.ocr_text.strip()]
    return sorted(list(pool.transcript) + extra, key=lambda s: s.start_s)


def stage_eval(a) -> None:
    run = Path(a.run)
    budget, pools = Budget(run), _pools(run)
    teacher = _teacher(run, a.config)
    ind_rows, pre_rows = _read_rows(run / "rows_independent.jsonl"), _read_rows(run / "rows_prefix.jsonl")
    labelled = {r.qa_id for r in pre_rows}
    videos = sorted({pools[q].video_id for q in labelled})
    results, sel_seconds, answer_seconds = [], collections.defaultdict(list), []
    t0 = time.perf_counter()
    for held in videos:
        train_ind = [r for r in ind_rows if r.video_id != held]
        train_pre = [r for r in pre_rows if r.video_id != held]
        heads_c = [_train_head(train_ind, pools, s) for s in SEEDS]
        heads_d = [_train_head(train_pre, pools, s) for s in SEEDS]
        arms = [SimilarityTopK(), MMRSelector()]
        arms += [UnaryUtility(h, f"C_independent_utility_s{s}") for h, s in zip(heads_c, SEEDS, strict=True)]
        arms += [GreedyUtility(h, f"D_history_utility_s{s}") for h, s in zip(heads_d, SEEDS, strict=True)]
        arms += [UnaryUtility(h, f"D_head_unary_s{s}") for h, s in zip(heads_d, SEEDS, strict=True)]
        for q in sorted(q for q in labelled if pools[q].video_id == held):
            if budget.remaining - (time.perf_counter() - t0) < 0:
                print("budget exhausted during eval; partial results saved")
                break
            pool = pools[q]
            row = {"qa_id": q, "video_id": held, "transcript_only": teacher.quality(pool, [])}
            top = SimilarityTopK().select(pool, K_FRAMES)
            row["ocr_transcript_no_image"] = teacher.quality(pool, [], transcript=_ocr_transcript(pool, top))
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
    summary = summarize(results, pools, sel_seconds, answer_seconds)
    summary["budget"] = budget.state
    (run / "eval_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def summarize(results, pools, sel_seconds, answer_seconds) -> dict:
    """Accuracy per arm (seed-averaged for trained arms), paired bootstrap, composed latency."""
    if not results:
        return {"questions": 0}
    names = [k for k in results[0] if k not in ("qa_id", "video_id")]
    groups = collections.defaultdict(list)          # seed-averaged arms: "D_history_utility_s0" -> "D_history_utility"
    for n in names:
        groups[re.sub(r"_s\d+$", "", n)].append(n)
    per_q = {g: np.array([np.mean([r[n] for n in members]) for r in results]) for g, members in groups.items()}
    vids = np.array([r["video_id"] for r in results])
    out = {"questions": len(results), "videos": sorted(set(vids)), "k_frames": K_FRAMES,
           "accuracy": {g: float(v.mean()) for g, v in per_q.items()}, "paired": {}}
    rng = np.random.default_rng(0)
    for a_name, b_name in [("D_history_utility", "A_mobileclip_topk"), ("D_history_utility", "B_mobileclip_mmr"),
                           ("D_history_utility", "C_independent_utility"), ("D_history_utility", "D_head_unary"),
                           ("C_independent_utility", "A_mobileclip_topk"), ("A_mobileclip_topk", "transcript_only"),
                           ("A_mobileclip_topk", "ocr_transcript_no_image")]:
        if a_name not in per_q or b_name not in per_q:
            continue
        d = per_q[a_name] - per_q[b_name]
        qb = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        uv = sorted(set(vids))
        cb = []
        for _ in range(10000):
            pick = rng.choice(uv, len(uv))
            cb.append(np.concatenate([d[vids == v] for v in pick]).mean())
        out["paired"][f"{a_name} - {b_name}"] = {
            "diff": float(d.mean()), "ci95_question_bootstrap": [float(np.percentile(qb, 2.5)), float(np.percentile(qb, 97.5))],
            "ci95_lecture_bootstrap": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]}
    # Composed end-to-end latency per question (measured stage times; see docstring of stage_eval).
    t = {k: float(np.median([p.timings.get(k, 0.0) for p in pools.values()])) for k in
         ("retrieval", "head_a", "decode", "clip", "ocr", "text_emb")}
    ans = float(np.median(answer_seconds)) if answer_seconds else float("nan")
    sel = {g: float(np.median(sum((sel_seconds.get(n, []) for n in members), []))) for g, members in groups.items()
           if any(n in sel_seconds for n in members)}
    common = t["retrieval"] + t["head_a"] + t["decode"] + t["clip"]
    out["latency_s_median"] = {
        "stage_medians": t, "answer_k_frames": ans,
        **{g: common + (t["ocr"] + t["text_emb"] if g.startswith(("C_", "D_")) else 0.0) + sel.get(g, 0.0) + ans
           for g in sel}}
    return out


# ---------------------------------------------------------------------------
# Stage: head-a (separate text vs frame interventions)
# ---------------------------------------------------------------------------


def stage_head_a(a) -> None:
    run = Path(a.run)
    budget, pools = Budget(run), _pools(run)
    teacher = _teacher(run, a.config)
    items = {it.qa.qa_id: it for it in _items(json.loads((run / "pools.json").read_text())["data"])}
    out_path = run / "head_a_labels.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["qa_id"] for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    t0 = time.perf_counter()
    for q in _audit_order(pools)[:AUDIT_N]:
        if q in done:
            continue
        if budget.remaining - (time.perf_counter() - t0) < 0:
            print("budget exhausted during head-a labelling; progress saved")
            break
        pool, segs = pools[q], items[q].transcript.segments
        pairs = []
        for c in pool.candidates:
            if "A" in c.paths:
                seg = min(segs, key=lambda s: abs((s.end_s - 0.3) - c.time_s))
                pairs.append((seg, c))
        rows = head_a_labels(pool, teacher, pairs)
        with open(out_path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    budget.charge("head_a", time.perf_counter() - t0)
    rows = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    tg = np.array([r["text_gain"] for r in rows])
    vg = np.array([r["visual_gain"] for r in rows])
    summary = {"rows": len(rows), "questions": len({r["qa_id"] for r in rows}),
               "text_gain": {"pos": int((tg > 0).sum()), "zero": int((tg == 0).sum()), "neg": int((tg < 0).sum())},
               "visual_gain": {"pos": int((vg > 0).sum()), "zero": int((vg == 0).sum()), "neg": int((vg < 0).sum())},
               "text_and_visual_both_positive": int(((tg > 0) & (vg > 0)).sum()),
               "only_text_positive": int(((tg > 0) & (vg <= 0)).sum()),
               "only_visual_positive": int(((vg > 0) & (tg <= 0)).sum()),
               "teacher_calls": teacher.calls, "budget": budget.state}
    (run / "head_a_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(prog="videoqa.v2.pilot")
    p.add_argument("stage", choices=["pools", "audit", "label", "eval", "head-a"])
    p.add_argument("--data", default="data/mit_lectures")
    p.add_argument("--run", default="runs/v2_pilot")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    {"pools": stage_pools, "audit": stage_audit, "label": stage_label, "eval": stage_eval,
     "head-a": stage_head_a}[a.stage](a)


if __name__ == "__main__":
    main()
