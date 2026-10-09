"""Baseline cycle (v3 step 2): tune the rule-based two-path framework on dev, then freeze it.

The framework is the v2 pipeline (docs/v3/baseline_framework.md): BM25 windows -> Path A (zero-shot
MiniLM speech lines, frame at line end) + Path B (visual scan, MobileCLIP) -> de-duplicated pool ->
rule selector -> frozen Qwen3-VL-2B with a transcript excerpt. No trained head.

Benchmarks: CG-Bench dev (8 videos, 82 questions) and Video-MMMU Perception dev (98 questions).
One setting is varied at a time from the v2 defaults; every trial is recorded.

Pool settings (each builds its own pools):
    v2        BM25 windows, legacy scan, 5 s step, cap 24, A 6 + B 6          (v2 main experiment)
    balanced  as v2, balanced scan inside the windows
    video     as v2, but Path B scans the whole video evenly (cap 48)
    dense     as v2, but Path B ranks dense 2 s MobileCLIP features of the whole video and decodes the
              top 24 (>= 4 s apart); needs scripts/v3_features.py output
    hybrid    Path B keeps its best 3 frames from the windows scan and the best 3 of the rest (mostly dense)

Selector rules (on the same pool): clip (question similarity), clipopt (question + options), mmr,
mmropt (MMR on question + options), relevance (Path A speech relevance); path ablations: A_only, B_only.
Answer settings: K final frames; transcript budget in words (BM25 excerpt; 0 = none).

    python scripts/v3_baseline_cycle.py pools  --dataset cgbench --pool v2
    python scripts/v3_baseline_cycle.py answer --dataset cgbench --pool v2 --selectors clip,clipopt --k 4 --words 120
    python scripts/v3_baseline_cycle.py open   --dataset cgbench --pool dense --selectors clip --k 4 --words 120
    python scripts/v3_baseline_cycle.py judge      # open-ended correct/incorrect (CG-Bench short references)
    python scripts/v3_baseline_cycle.py report
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path("runs/v3_baseline")
OUT = Path("reports/v3_baseline")
DATASETS = {"cgbench": ("data/cgbench", "dev", None), "videommmu": ("data/videommmu", "dev", "vmmmu:Perception:"),
            # Head B training pools (CG-Bench train videos we hold; never used for selection decisions)
            "cgbench_train": ("data/cgbench", "train", None)}
POOLS = {
    "v2": dict(scan_step_s=5.0, scan_cap=24),
    "balanced": dict(scan_step_s=5.0, scan_cap=24, scan_policy="balanced"),
    "video": dict(scan_step_s=5.0, scan_cap=48, path_b_scope="video"),
    # dense 2 s MobileCLIP features over the whole video (the gate's strongest frame rule), top 24 decoded
    "dense": dict(scan_step_s=5.0, scan_cap=24, path_b_scope="features"),
    # half of Path B from the dense whole-video features, half from the v2 scan inside the BM25 windows
    "hybrid": dict(scan_step_s=5.0, scan_cap=24, path_b_scope="hybrid"),
    # stage 3, path budgets on the chosen pool (hybrid): Path A 3 instead of 6; Path B 12 instead of 6
    "hybrid_a3": dict(scan_step_s=5.0, scan_cap=24, path_b_scope="hybrid", n_path_a=3),
    "hybrid_b12": dict(scan_step_s=5.0, scan_cap=24, path_b_scope="hybrid", n_path_b=12),
    # Head A cycle (frozen baseline, only Path A changes): zero-shot MiniLM over the whole transcript, and the
    # learned Head A (light residual head, density moments, bge features; 3 seeds averaged; research_log step 39)
    "hybrid_azs": dict(scan_step_s=5.0, scan_cap=24, path_b_scope="hybrid", path_a="zeroshot"),
    "hybrid_ahead": dict(scan_step_s=5.0, scan_cap=24, path_b_scope="hybrid", path_a="head"),
}
HEAD_A_CKPTS = [f"runs/v3_head_a/ckpt/light_dens_ens_human_{s}.pt" for s in range(3)]
FEATURES = {"cgbench": "data/cgbench/features", "videommmu": "data/videommmu/features",
            "cgbench_train": "data/cgbench/features"}
SELECTIONS = Path("runs/v3_head_b/selections")
LIMITS = (4000, 4 * 3600.0)
# second ledger, declared in research_log step 35 after the first was used up by stages 1-3
EXTRA_LIMITS = (1500, 2 * 3600.0)
OPEN_LIMITS = (1500, 4 * 3600.0)
JUDGE = "nvidia/nemotron-3-ultra-550b-a55b"
JUDGE_LIMITS = (3000, 4 * 3600.0)


def items(dataset: str):
    from videoqa.datasets import load_local_dataset  # noqa: PLC0415

    root, split, prefix = DATASETS[dataset]
    loaded, _ = load_local_dataset(root, splits=(split,))
    rows = [x for x in loaded if prefix is None or x.qa.qa_id.startswith(prefix)]
    return sorted(rows, key=lambda x: hashlib.sha256(x.qa.qa_id.encode()).hexdigest())


def pool_dir(dataset: str, pool: str) -> Path:
    return RUN / dataset / pool / "pools"


def stage_pools(a) -> None:
    from videoqa.v2.candidates import PoolConfig, build_pool, save_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415
    from videoqa.v2.head_a import HotMomentScorer  # noqa: PLC0415

    settings = dict(POOLS[a.pool])
    path_a = settings.pop("path_a", "v2")
    cfg = PoolConfig(**settings, excerpt_words=300,
                     features_dir=FEATURES[a.dataset] if a.pool == "dense" or a.pool.startswith("hybrid") else None)
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    head_a = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub")
    if path_a == "zeroshot":
        from videoqa.v3.head_a import ZeroShotMoments  # noqa: PLC0415
        head_a = ZeroShotMoments(head_a)
    elif path_a == "head":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from v3_head_a_eval import Reranker  # noqa: PLC0415
        from v3_head_a_light import BGE  # noqa: PLC0415

        from videoqa.v3.head_a import LearnedMoments  # noqa: PLC0415
        head_a = LearnedMoments(head_a, HEAD_A_CKPTS, bge=Reranker(BGE))
    out = pool_dir(a.dataset, a.pool)
    out.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in out.iterdir()}
    meta = {"pool": a.pool, "config": dataclasses.asdict(cfg), "qa_ids": []}
    for n, it in enumerate(items(a.dataset)):
        safe = it.qa.qa_id.replace(":", "_").replace("/", "_")
        if safe not in done:
            pool = build_pool(it.qa, it.transcript, it.video_path, enc, head_a, cfg)
            save_pool(pool, out)
        meta["qa_ids"].append(it.qa.qa_id)
        if (n + 1) % 10 == 0:
            print(f"{n + 1} pools", flush=True)
    (out.parent / "pools.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def excerpt(pool, words: int):
    """The pool's retained excerpt (packed at 300 words) cut to the first ``words`` words, in time order."""
    if words <= 0:
        return []
    out, n = [], 0
    for s in sorted(pool.transcript, key=lambda s: s.start_s):
        n += len(s.text.split())
        if n > words and out:
            break
        out.append(s)
    return out


_FILE_SELECTIONS: dict = {}


def select(pool, rule: str, k: int, qo_emb):
    cands = list(pool.candidates)
    if rule.startswith("file:"):                     # a Head B method's saved choice (scripts/v3_head_b.py)
        path = SELECTIONS / f"{rule[5:]}.json"
        if path not in _FILE_SELECTIONS:
            _FILE_SELECTIONS[path] = json.loads(path.read_text(encoding="utf-8"))
        ids = _FILE_SELECTIONS[path][pool.qa_id][:k]
        by_id = {c.id: c for c in cands}
        return [by_id[i] for i in ids]
    if rule == "A_only":
        cands = [c for c in cands if "A" in c.paths]
    if rule == "B_only":
        cands = [c for c in cands if "B" in c.paths]
    if rule in ("A_only", "B_only", "clipopt", "mmropt"):
        sim = {c.id: float(np.asarray(c.emb) @ qo_emb) for c in cands}
    elif rule == "relevance":
        sim = {c.id: c.head_a_visual for c in cands}
    else:
        sim = {c.id: c.clip_sim for c in cands}
    if rule in ("mmr", "mmropt"):
        chosen, rest = [], list(cands)
        while rest and len(chosen) < k:
            best = max(rest, key=lambda c: 0.7 * sim[c.id] - 0.3 * max((float(np.asarray(c.emb) @ np.asarray(h.emb))
                                                                       for h in chosen), default=0.0))
            chosen.append(best)
            rest.remove(best)
        return chosen
    return sorted(cands, key=lambda c: -sim[c.id])[:k]


def request_key(pool, frames, ex, identity) -> str:
    from videoqa.answerer import AnswerRequest, build_prompt_text  # noqa: PLC0415

    prompt = build_prompt_text(AnswerRequest(pool.question, pool.options, ex, []))
    payload = {"identity": identity, "prompt": prompt,
               "frames": [[c.digest, f"Frame at {c.time_s:.2f}s:"] for c in sorted(frames, key=lambda c: c.time_s)]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_answer(a) -> None:
    from videoqa.answerer import AnswerRequest, make_answerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415
    from videoqa.v2.teacher import _as_frame, answerer_identity  # noqa: PLC0415

    meta = json.loads((pool_dir(a.dataset, a.pool).parent / "pools.json").read_text())
    pools = [load_pool(pool_dir(a.dataset, a.pool), q) for q in meta["qa_ids"]]
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    qo = enc.embed_texts([p.question + " Options: " + ", ".join(p.options) for p in pools])
    cfg = load_config(a.config)
    identity = answerer_identity(cfg)
    RUN.mkdir(parents=True, exist_ok=True)
    cache_path, rows_path = RUN / "answers.jsonl", RUN / "rows.jsonl"
    cache = {json.loads(x)["key"]: json.loads(x) for x in cache_path.read_text().splitlines()} if cache_path.exists() else {}
    budget = CallBudget(RUN / a.ledger, *(LIMITS if a.ledger == "budget.json" else EXTRA_LIMITS))
    answerer = None
    qa = {it.qa.qa_id: it.qa for it in items(a.dataset)}
    for rule in a.selectors.split(","):
        for p, e in zip(pools, qo, strict=True):
            frames = select(p, rule, a.k, e)
            ex = excerpt(p, a.words)
            key = request_key(p, frames, ex, identity)
            if key not in cache:
                if not budget.reserve(1):
                    print("budget reached; stopping cleanly")
                    return
                answerer = answerer or make_answerer(cfg)
                t0 = time.perf_counter()
                ans, _ = answerer.answer(AnswerRequest(p.question, p.options, ex,
                                                       [_as_frame(c) for c in sorted(frames, key=lambda c: c.time_s)]))
                budget.charge(time.perf_counter() - t0)
                cache[key] = {"key": key, "option": ans.option_index, "seconds": time.perf_counter() - t0}
                with open(cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(cache[key]) + "\n")
            iv = qa[p.qa_id].evidence_intervals_s
            inside = lambda t: any(x - 1.0 <= t <= y + 1.0 for x, y in iv)  # noqa: E731
            row = {"dataset": a.dataset, "pool": a.pool, "selector": rule, "k": a.k, "words": a.words,
                   "qa_id": p.qa_id, "video_id": p.video_id, "correct": cache[key]["option"] == p.gold_option_index,
                   "pool_size": len(p.candidates), "pool_hits_evidence": any(inside(c.time_s) for c in p.candidates) if iv else None,
                   "chosen_hits_evidence": any(inside(c.time_s) for c in frames) if iv else None,
                   "pool_seconds": sum(p.timings.get(x, 0.0) for x in ("retrieval", "head_a", "decode", "clip"))}
            with open(rows_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"{a.dataset}/{a.pool}/{rule} K={a.k} words={a.words}: done; calls {budget.calls}", flush=True)


def stage_open(a) -> None:
    """Open-ended answers (no options shown) for one pool + selector; CG-Bench has short human references."""
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415
    from videoqa.v2.teacher import _as_frame  # noqa: PLC0415
    from videoqa.v3.openended import AnswerCache, answerer_identity, make_open_answerer  # noqa: PLC0415

    meta = json.loads((pool_dir(a.dataset, a.pool).parent / "pools.json").read_text())
    pools = [load_pool(pool_dir(a.dataset, a.pool), q) for q in meta["qa_ids"]]
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    qo = enc.embed_texts([p.question + " Options: " + ", ".join(p.options) for p in pools])
    cfg = load_config(a.config)
    budget = CallBudget(RUN / "open_budget.json", *OPEN_LIMITS)
    cache = AnswerCache(RUN / "open_answers.jsonl", answerer_identity(cfg), None, budget)
    qa = {it.qa.qa_id: it.qa for it in items(a.dataset)}
    for rule in a.selectors.split(","):
        for p, e in zip(pools, qo, strict=True):
            frames = [_as_frame(c) for c in select(p, rule, a.k, e)]
            ex = excerpt(p, a.words)
            if cache.get(p.question, ex, frames) is None and cache.answerer is None:
                cache.answerer = make_open_answerer(cfg)
            rec = cache.answer(p.qa_id, p.question, ex, frames)
            row = {"dataset": a.dataset, "pool": a.pool, "selector": rule, "k": a.k, "words": a.words,
                   "qa_id": p.qa_id, "question": p.question, "reference": qa[p.qa_id].gold_answer,
                   "answer": rec["text"]}
            with open(RUN / "open_rows.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"open {a.dataset}/{a.pool}/{rule} K={a.k} words={a.words}: done; calls {budget.calls}", flush=True)


def stage_judge(a) -> None:
    """Correct / incorrect for every open-ended row (NVIDIA NIM free tier, validated prompt, cached)."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v3.factqa import ShortAnswerJudge  # noqa: PLC0415

    rows = [json.loads(x) for x in (RUN / "open_rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    judge = ShortAnswerJudge(JUDGE, RUN / "judge.jsonl", budget=CallBudget(RUN / "judge_budget.json", *JUDGE_LIMITS),
                             max_tokens=4096)
    judge.time_share = 1 / 4

    def one(r):
        try:
            return judge.correct(r["qa_id"], r["question"], r["reference"], r["answer"])["correct"]
        except Exception as e:  # noqa: BLE001 - one failed verdict must not stop the batch
            print(f"judge failed {r['qa_id']}: {str(e)[:120]}", flush=True)
            return None

    with ThreadPoolExecutor(4) as ex:
        verdicts = list(ex.map(one, rows))
    for r, v in zip(rows, verdicts, strict=True):
        r["open_correct"] = v
    (RUN / "open_judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"judged {sum(v is not None for v in verdicts)}/{len(rows)}; new calls {judge.calls}")


def stage_report(a) -> None:
    rows = [json.loads(x) for x in (RUN / "rows.jsonl").read_text().splitlines() if x.strip()]
    latest = {}
    for r in rows:                                   # last row wins for identical settings
        latest[(r["dataset"], r["pool"], r["selector"], r["k"], r["words"], r["qa_id"])] = r
    groups = {}
    for (d, p, s, k, w, _), r in latest.items():
        groups.setdefault((d, p, s, k, w), []).append(r)
    table = []
    for (d, p, s, k, w), g in sorted(groups.items()):
        ev = [r["chosen_hits_evidence"] for r in g if r["chosen_hits_evidence"] is not None]
        pe = [r["pool_hits_evidence"] for r in g if r["pool_hits_evidence"] is not None]
        table.append({"dataset": d, "pool": p, "selector": s, "k": k, "words": w, "n": len(g),
                      "accuracy": float(np.mean([r["correct"] for r in g])),
                      "pool_evidence_recall": float(np.mean(pe)) if pe else None,
                      "chosen_evidence_recall": float(np.mean(ev)) if ev else None,
                      "pool_size_median": float(np.median([r["pool_size"] for r in g])),
                      "pool_seconds_median": float(np.median([r["pool_seconds"] for r in g]))})
    judged = RUN / "open_judged.jsonl"
    if judged.exists():
        last = {}
        for r in (json.loads(x) for x in judged.read_text(encoding="utf-8").splitlines() if x.strip()):
            last[(r["dataset"], r["pool"], r["selector"], r["k"], r["words"], r["qa_id"])] = r["open_correct"]
        for t in table:
            v = [c for (d, p, s, k, w, _), c in last.items()
                 if (d, p, s, k, w) == (t["dataset"], t["pool"], t["selector"], t["k"], t["words"]) and c is not None]
            t["open_accuracy"], t["open_n"] = (float(np.mean(v)), len(v)) if v else (None, 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for t in table:
        print(f"{t['dataset']:9s} {t['pool']:8s} {t['selector']:9s} K={t['k']} w={t['words']:3d} n={t['n']:3d} "
              f"acc={t['accuracy']:.3f} pool_ev={t['pool_evidence_recall']} chosen_ev={t['chosen_evidence_recall']} "
              f"pool={t['pool_size_median']:.0f} t={t['pool_seconds_median']:.1f}s"
              + (f" open={t['open_accuracy']:.3f} (n={t['open_n']})" if t.get("open_accuracy") is not None else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("pools", "answer", "open", "judge", "report"))
    p.add_argument("--dataset", choices=tuple(DATASETS), default="cgbench")
    p.add_argument("--pool", choices=tuple(POOLS), default="v2")
    p.add_argument("--selectors", default="clip,clipopt,mmr,mmropt,relevance")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--words", type=int, default=120)
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    p.add_argument("--ledger", default="budget.json", help="answer-call ledger (budget_2.json = the step-35 extension)")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"pools": stage_pools, "answer": stage_answer, "open": stage_open, "judge": stage_judge,
     "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
