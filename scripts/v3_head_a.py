"""Head A pilot on CG-Bench: does a learned "where to look" beat MobileCLIP? (declared 2026-10-09)

Data: CG-Bench, 51 English-subtitled videos we hold (split in prepare_cgbench.py): train 37 videos /
373 questions, dev 8 / 82. Features: MobileCLIP-S2 every 2 s (scripts/v3_features.py).

Protocol (fixed before any Head A result):
- Model selection uses TRAIN videos only: train videos are split by a sha256 rule into fit (80%) and
  val (20%); the epoch with the best val hit@4 is kept. 3 seeds; Head A's dev scores are the average of
  the 3 seeds' scores. Dev is used once, for the comparison below.
- Offline metric (no answerer calls): hit@4 = share of questions with at least one of the 4 chosen frames
  inside a human clue interval (padded 1 s). Same selection rule for every method: top-4 peaks >= 8 s apart.
- Baselines on the same features: uniform, MobileCLIP similarity, temporally smoothed MobileCLIP,
  AKS-style relevance + coverage (ADA split; threshold tuned on train), random (expected).
- Pilot decision rule: continue to online QA evaluation and scaling only if Head A's dev hit@4 beats
  MobileCLIP's with a paired bootstrap CI above zero.

    python scripts/v3_head_a.py prepare     # question text embeddings
    python scripts/v3_head_a.py run         # baselines + Head A training + dev comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

DATA = Path("data/cgbench")
RUN = Path("runs/v3_head_a")
OUT = Path("reports/v3_head_a")
K, MIN_SEP = 4, 8.0
SEEDS = (0, 1, 2)


def load_items(split: str) -> list[dict]:
    qe = np.load(RUN / "q_emb.npz")
    rows = [json.loads(x) for x in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    items, cache = [], {}
    for r in rows:
        if r["experiment_split"] != split or not (DATA / "features" / f"{r['video_id']}.npz").exists():
            continue
        if r["video_id"] not in cache:
            f = np.load(DATA / "features" / f"{r['video_id']}.npz")
            cache[r["video_id"]] = (f["times"].astype(np.float32), f["emb"].astype(np.float32), float(f["duration"]))
        times, emb, dur = cache[r["video_id"]]
        items.append({"qa_id": r["qa_id"], "video_id": r["video_id"], "times": times, "emb": emb, "duration": dur,
                      "q": qe[r["qa_id"]].astype(np.float32), "intervals": r["evidence_intervals_s"]})
    return items


def stage_prepare(a) -> None:
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    RUN.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    e = enc.embed_texts([r["question"] for r in rows])
    np.savez(RUN / "q_emb.npz", **{r["qa_id"]: v for r, v in zip(rows, e, strict=True)})
    print(f"{len(rows)} question embeddings")


# ---------------------------------------------------------------- baselines (label-free)


def s_clip(it):
    return it["emb"] @ it["q"]


def s_smooth(it, width: int = 2):
    s = s_clip(it)
    k = np.exp(-0.5 * (np.arange(-2 * width, 2 * width + 1) / width) ** 2)
    return np.convolve(s, k / k.sum(), mode="same")


def aks_select(it, thr: float, k: int = K, depth: int = 0, lo: int = 0, hi: int | None = None) -> list[float]:
    """AKS-style ADA split: take top-k inside a segment when relevance is concentrated, else split in two."""
    from videoqa.v3.temporal_head import select_peaks  # noqa: PLC0415

    s, t = s_clip(it), it["times"]
    hi = len(s) if hi is None else hi
    if k <= 0 or hi - lo <= 0:
        return []
    seg = s[lo:hi]
    top = np.sort(seg)[-k:].mean() if len(seg) >= k else seg.mean()
    if top - seg.mean() > thr or depth >= 4 or hi - lo < 2 * k:
        return select_peaks(seg, t[lo:hi], k, MIN_SEP)
    mid = (lo + hi) // 2
    k1 = k // 2 + (k % 2 if s[lo:mid].mean() >= s[mid:hi].mean() else 0)
    return sorted(aks_select(it, thr, k1, depth + 1, lo, mid) + aks_select(it, thr, k - k1, depth + 1, mid, hi))


def uniform_select(it, k: int = K) -> list[float]:
    return [float(x) for x in np.linspace(0, it["duration"], k + 2)[1:-1]]


def random_hit(it, k: int = K, draws: int = 200, seed: int = 0) -> float:
    from videoqa.v3.temporal_head import hit_at_k, select_peaks  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    return float(np.mean([hit_at_k(select_peaks(rng.random(len(it["times"])), it["times"], k, MIN_SEP), it["intervals"])
                          for _ in range(draws)]))


def paired_boot(a: np.ndarray, b: np.ndarray, repeats: int = 5000) -> dict:
    d = a - b
    i = np.random.default_rng(0).integers(0, len(d), (repeats, len(d)))
    return {"difference": float(d.mean()), "ci95": np.percentile(d[i].mean(1), [2.5, 97.5]).tolist()}


def stage_run(a) -> None:
    from videoqa.v3.temporal_head import HeadAConfig, hit_at_k, score_items, select_peaks, train_head  # noqa: PLC0415

    OUT.mkdir(parents=True, exist_ok=True)
    train, dev = load_items("train"), load_items("dev")
    fit = [x for x in train if int(hashlib.sha256(f"fit:{x['video_id']}".encode()).hexdigest(), 16) % 5 != 0]
    val = [x for x in train if int(hashlib.sha256(f"fit:{x['video_id']}".encode()).hexdigest(), 16) % 5 == 0]
    print(f"train {len(train)} ({len(fit)} fit / {len(val)} val), dev {len(dev)}")

    def hits(items, chooser):
        return np.array([hit_at_k(chooser(it), it["intervals"]) for it in items])

    thr_grid = (0.0, 0.005, 0.01, 0.02, 0.05)
    aks_thr = max(thr_grid, key=lambda th: hits(train, lambda it: aks_select(it, th)).mean())
    methods = {
        "uniform": lambda it: uniform_select(it),
        "mobileclip": lambda it: select_peaks(s_clip(it), it["times"], K, MIN_SEP),
        "mobileclip_smoothed": lambda it: select_peaks(s_smooth(it), it["times"], K, MIN_SEP),
        "aks": lambda it: aks_select(it, aks_thr),
    }
    result = {"questions": {"train": len(train), "fit": len(fit), "val": len(val), "dev": len(dev)},
              "aks_threshold_tuned_on_train": aks_thr, "hit@4": {}, "history": {}}
    dev_hits = {m: hits(dev, f) for m, f in methods.items()}
    dev_hits["random_expected"] = np.array([random_hit(it) for it in dev])

    cfg = HeadAConfig()
    dev_scores = []
    for seed in SEEDS:
        model, hist = train_head(fit, val, cfg, seed=seed, log=lambda m: print(f"[seed {seed}] {m}", flush=True))
        result["history"][f"seed{seed}"] = hist
        dev_scores.append(score_items(model, dev))
        dev_hits[f"head_a_seed{seed}"] = np.array([hit_at_k(select_peaks(s, it["times"], K, MIN_SEP), it["intervals"])
                                                   for s, it in zip(dev_scores[-1], dev, strict=True)])
    mean_scores = [np.mean([d[i] for d in dev_scores], 0) for i in range(len(dev))]
    chosen = {it["qa_id"]: select_peaks(s, it["times"], K, MIN_SEP) for s, it in zip(mean_scores, dev, strict=True)}
    dev_hits["head_a"] = np.array([hit_at_k(chosen[it["qa_id"]], it["intervals"]) for it in dev])
    for m, h in dev_hits.items():
        result["hit@4"][m] = float(h.mean())
    result["head_a_minus_mobileclip"] = paired_boot(dev_hits["head_a"], dev_hits["mobileclip"])
    result["head_a_minus_aks"] = paired_boot(dev_hits["head_a"], dev_hits["aks"])
    result["pilot_passes"] = result["head_a_minus_mobileclip"]["ci95"][0] > 0
    (RUN / "a4_times_dev.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    (OUT / "offline_dev.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "run"))
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"prepare": stage_prepare, "run": stage_run}[a.stage](a)


if __name__ == "__main__":
    main()
