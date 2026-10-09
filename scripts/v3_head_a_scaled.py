"""Scaled Head A on CG-Bench (declared 2026-10-09, before any scaled result).

Why: the pilot overfit 306 training questions (docs/v3/research_log.md, step 19). This tests whether
more supervision fixes it, with the method otherwise unchanged.

Training data: every released CG-Bench video with features EXCEPT our 14 dev/test videos
(scripts/research/cgbench_train_features.py), with all its questions and human clue intervals.
Query: question + options text (label-free; +5.4 hit@4 over question-only on the 37 pilot train videos,
CI +1.6 to +9.0), used for BOTH the MobileCLIP baseline and Head A's input.

Model selection uses training videos only: 10% of training videos (sha256 rule) are an inner
validation set. Small grid chosen there: hidden {128, 256} x dropout {0.1, 0.3}, up to 15 epochs, best
epoch by inner-val hit@4. 3 seeds of the chosen setting; dev scores = average of the 3 seeds.

Primary dev comparison (used once): Head A vs option-aware MobileCLIP, offline hit@4 (82 dev questions).
Stop rule: if the paired CI of Head A − option-aware MobileCLIP includes 0 or is negative, Head A is
stopped and reported as not beating the baseline. Otherwise: online QA on dev with the frozen VLM.

    python scripts/v3_head_a_scaled.py prepare
    python scripts/v3_head_a_scaled.py run
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np

DATA = Path("data/cgbench")
RUN = Path("runs/v3_head_a_scaled")
OUT = Path("reports/v3_head_a")
K, MIN_SEP, SEEDS = 4, 8.0, (0, 1, 2)
GRID = [{"hidden": h, "dropout": d} for h in (128, 256) for d in (0.1, 0.3)]


def query_text(question: str, options) -> str:
    return question + " Options: " + ", ".join(options)


def all_questions() -> list[dict]:
    """All CG-Bench questions on videos with features; dev/test videos keep their split, others are train."""
    split = {json.loads(x)["video_id"]: json.loads(x)["experiment_split"]
             for x in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    have = {p.stem for p in (DATA / "features").glob("*.npz")}
    out = []
    for q in json.loads(Path("data/cgbench_raw/cgbench.json").read_text(encoding="utf-8")):
        if q["video_uid"] not in have:
            continue
        opts = q["choices"] if isinstance(q["choices"], list) else ast.literal_eval(q["choices"])
        clues = q["clue_intervals"] if isinstance(q["clue_intervals"], list) else ast.literal_eval(q["clue_intervals"])
        out.append({"qa_id": f"cgbench:{q['qid']}", "video_id": q["video_uid"], "question": q["question"],
                    "options": list(opts), "intervals": [[float(a), float(b)] for a, b in clues],
                    "split": split.get(q["video_uid"], "train")})
    return out


def stage_prepare(a) -> None:
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    RUN.mkdir(parents=True, exist_ok=True)
    qs = all_questions()
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    e = enc.embed_texts([query_text(q["question"], q["options"]) for q in qs])
    np.savez(RUN / "q_emb.npz", **{q["qa_id"]: v for q, v in zip(qs, e, strict=True)})
    counts = {s: sum(q["split"] == s for q in qs) for s in ("train", "dev", "test")}
    vids = {s: len({q["video_id"] for q in qs if q["split"] == s}) for s in ("train", "dev", "test")}
    (RUN / "data_counts.json").write_text(json.dumps({"questions": counts, "videos": vids}, indent=1))
    print(json.dumps({"questions": counts, "videos": vids}))


def items(split: str) -> list[dict]:
    qe = np.load(RUN / "q_emb.npz")
    cache, out = {}, []
    for q in all_questions():
        if q["split"] != split:
            continue
        if q["video_id"] not in cache:
            f = np.load(DATA / "features" / f"{q['video_id']}.npz")
            cache[q["video_id"]] = (f["times"].astype(np.float32), f["emb"].astype(np.float32), float(f["duration"]))
        t, e, d = cache[q["video_id"]]
        out.append({**q, "times": t, "emb": e, "duration": d, "q": qe[q["qa_id"]].astype(np.float32)})
    return out


def stage_run(a) -> None:
    from dataclasses import replace  # noqa: PLC0415

    from videoqa.v3.temporal_head import HeadAConfig, hit_at_k, score_items, select_peaks, train_head  # noqa: PLC0415

    OUT.mkdir(parents=True, exist_ok=True)
    train, dev = items("train"), items("dev")
    inner = lambda v: int(hashlib.sha256(f"inner:{v}".encode()).hexdigest(), 16) % 10 == 0  # noqa: E731
    fit = [x for x in train if not inner(x["video_id"])]
    val = [x for x in train if inner(x["video_id"])]
    print(f"train {len(train)} questions ({len({x['video_id'] for x in fit})} fit videos / "
          f"{len({x['video_id'] for x in val})} val videos), dev {len(dev)}", flush=True)
    base = HeadAConfig(epochs=15)
    grid_scores = []
    for g in GRID:
        cfg = replace(base, **g)
        _, hist = train_head(fit, val, cfg, seed=0, log=lambda m: None)
        grid_scores.append({**g, "best_val_hit@4": max(h["hit@4"] for h in hist)})
        print(f"grid {g}: inner-val hit@4 {grid_scores[-1]['best_val_hit@4']:.3f}", flush=True)
    best = max(grid_scores, key=lambda x: x["best_val_hit@4"])
    cfg = replace(base, hidden=best["hidden"], dropout=best["dropout"])
    val_clip = np.mean([hit_at_k(select_peaks(x["emb"] @ x["q"], x["times"], K, MIN_SEP), x["intervals"]) for x in val])
    dev_scores, hist_all = [], {}
    for seed in SEEDS:
        model, hist = train_head(fit, val, cfg, seed=seed, log=lambda m: print(f"[seed {seed}] {m}", flush=True))
        hist_all[f"seed{seed}"] = hist
        dev_scores.append(score_items(model, dev))
    mean_scores = [np.mean([d[i] for d in dev_scores], 0) for i in range(len(dev))]
    head = np.array([hit_at_k(select_peaks(s, x["times"], K, MIN_SEP), x["intervals"]) for s, x in zip(mean_scores, dev, strict=True)])
    clip_o = np.array([hit_at_k(select_peaks(x["emb"] @ x["q"], x["times"], K, MIN_SEP), x["intervals"]) for x in dev])
    d = head - clip_o
    idx = np.random.default_rng(0).integers(0, len(d), (5000, len(d)))
    result = {"data": json.loads((RUN / "data_counts.json").read_text()), "fit_questions": len(fit), "val_questions": len(val),
              "grid": grid_scores, "chosen": best, "inner_val_hit@4_option_aware_mobileclip": float(val_clip),
              "dev_hit@4": {"option_aware_mobileclip": float(clip_o.mean()), "head_a_scaled": float(head.mean())},
              "head_a_minus_option_aware_mobileclip": {"difference": float(d.mean()),
                                                       "ci95": np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()},
              "history": hist_all}
    result["continue_to_online_qa"] = result["head_a_minus_option_aware_mobileclip"]["ci95"][0] > 0
    chosen = {x["qa_id"]: select_peaks(s, x["times"], K, MIN_SEP) for s, x in zip(mean_scores, dev, strict=True)}
    chosen_clip = {x["qa_id"]: select_peaks(x["emb"] @ x["q"], x["times"], K, MIN_SEP) for x in dev}
    (RUN / "dev_times.json").write_text(json.dumps({"head_a_scaled": chosen, "option_aware_mobileclip": chosen_clip}, indent=1))
    (OUT / "scaled_offline_dev.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "run"))
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"prepare": stage_prepare, "run": stage_run}[a.stage](a)


if __name__ == "__main__":
    main()
