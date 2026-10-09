"""Head A cycle: learned Path A methods, trained on data/head_a/train.jsonl, scored with v3_head_a_eval.

All methods read the question (optionally with the answer options) and each transcript line with one
neighbouring line on each side, and return ranked moments. Methods (one setting changes per method):

    ft_rel        MiniLM cross-encoder fine-tuned for line relevance (listwise loss: the lines near the
                  human evidence against hard and random negatives); moment = line end - 0.3 s
    ft_rel_off    ft_rel + an offset head: for each line, a distribution over where the evidence lies
                  relative to it (-40..+40 s, 4 s bins); moments = peaks of the evidence density
                  D(t) = sum_i softmax(rel)_i * q_i(t - t_i)   (the proposed speech-to-sight localiser)
    ft_rel_dens   ablation of ft_rel_off: the same density, offsets fixed at 0 (a Gaussian, sigma 4 s);
                  separates "several lines vote" from "learned offsets"
    zs_dens       the density built from zero-shot MiniLM relevance (training-free control)

Labels (CG-Bench human clue intervals; EduVidQA question time +/- 20 s as weak positives with lower weight):
    near line     |line time - nearest evidence| <= 10 s  (positives for relevance)
    offset target evidence centre - line time, for lines within 40 s of the evidence

Protocol: inner validation = 15% of train videos (fixed hash), used only to pick the epoch; 3 seeds; dev
(164 CG-Bench questions, 17 videos) scored once per trained model. Test-split videos are never read.

    python scripts/v3_head_a_train.py prepare                     # zero-shot relevance cache for hard negatives
    python scripts/v3_head_a_train.py train --method ft_rel_off --seed 0 [--query qo] [--no-weak]
    python scripts/v3_head_a_train.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_head_a_eval import KS, NMS_GAP_S, ZeroShot, load, nms, score_metrics, segments  # noqa: E402

RUN = Path("runs/v3_head_a")
OUT = Path("reports/v3_head_a")
NEAR_S = 10.0
OFF_RANGE_S, OFF_BIN_S = 40.0, 4.0
N_BINS = int(2 * OFF_RANGE_S / OFF_BIN_S) + 1          # 21 bins, centre bin = offset 0
GRID_S = 1.0
WEAK_WEIGHT = 0.5


def inner_val(video_id: str) -> bool:
    return int(hashlib.sha256(f"heada-inner:{video_id}".encode()).hexdigest(), 16) % 100 < 15


def contexts(rec) -> list[str]:
    lines = rec["lines"]
    return [" ".join(x[2] for x in lines[max(0, i - 1): i + 2]) for i in range(len(lines))]


def line_times(rec) -> np.ndarray:
    return np.asarray([max(0.0, min(rec["duration_s"] - 0.05, b - 0.3)) for _, b, _ in rec["lines"]])


def query_text(rec, mode: str) -> str:
    if mode == "qo" and rec["options"]:
        return rec["question"] + " Options: " + " | ".join(rec["options"])
    return rec["question"]


def labels(rec):
    """(near mask, offset bin or -1) per line."""
    t = line_times(rec)
    ev = rec["evidence"]
    dist = np.asarray([min(0.0 if a <= x <= b else min(abs(x - a), abs(x - b)) for a, b in ev) for x in t])
    centres = np.asarray([(a + b) / 2 for a, b in ev])
    off = centres[np.argmin(np.abs(centres[None, :] - t[:, None]), axis=1)] - t
    bins = np.where(np.abs(off) <= OFF_RANGE_S, np.round(off / OFF_BIN_S).astype(int) + N_BINS // 2, -1)
    return dist <= NEAR_S, bins


# -- zero-shot cache (hard negatives) -----------------------------------------------------------
def stage_prepare(a) -> None:
    zs = ZeroShot()
    RUN.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "dev"):
        for r in load(split):
            out[r["qa_id"]] = zs.minilm(r, segments(r)).astype(np.float32)
    np.savez_compressed(RUN / "zeroshot_rel.npz", **{k.replace(":", "__"): v for k, v in out.items()})
    print(f"cached zero-shot relevance for {len(out)} questions")


def zeroshot_rel() -> dict[str, np.ndarray]:
    z = np.load(RUN / "zeroshot_rel.npz")
    return {k.replace("__", ":"): z[k] for k in z.files}


# -- model ------------------------------------------------------------------------------------------
class Model:
    def __init__(self, device: str = "cuda") -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        from videoqa.v2.head_a import BACKBONE, BACKBONE_REV  # noqa: PLC0415

        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(BACKBONE, revision=BACKBONE_REV, cache_dir="cache/hf/hub")
        self.enc = AutoModelForSequenceClassification.from_pretrained(
            BACKBONE, revision=BACKBONE_REV, cache_dir="cache/hf/hub").to(device)
        hidden = self.enc.config.hidden_size
        self.off = torch.nn.Linear(hidden, N_BINS).to(device)
        torch.nn.init.zeros_(self.off.weight)
        torch.nn.init.zeros_(self.off.bias)

    def parameters(self):
        return list(self.enc.parameters()) + list(self.off.parameters())

    def forward(self, query: str, texts: list[str]):
        b = self.tok([query] * len(texts), texts, padding=True, truncation=True, max_length=256,
                     return_tensors="pt").to(self.device)
        out = self.enc(**b, output_hidden_states=True)
        return out.logits[:, 0], self.off(out.hidden_states[-1][:, 0])

    def predict(self, query: str, texts: list[str], batch: int = 256):
        torch = self.torch
        self.enc.eval()
        rel, off = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(texts), batch):
                r, o = self.forward(query, texts[i:i + batch])
                rel.append(r.float().cpu().numpy())
                off.append(torch.softmax(o.float(), -1).cpu().numpy())
        if not rel:
            return np.zeros(0), np.zeros((0, N_BINS))
        return np.concatenate(rel), np.concatenate(off)

    def state(self):
        return {"enc": self.enc.state_dict(), "off": self.off.state_dict()}

    def load_state(self, s) -> None:
        self.enc.load_state_dict(s["enc"])
        self.off.load_state_dict(s["off"])


# -- proposals --------------------------------------------------------------------------------------
def density_moments(rec, rel: np.ndarray, off: np.ndarray | None, top_lines: int = 64,
                    temperature: float = 1.0, sigma_s: float = 4.0) -> list[float]:
    """Peaks of D(t) = sum_i softmax(rel/T)_i * q_i(t - t_i); q_i = learned offset bins, or a Gaussian at 0."""
    if len(rel) == 0:
        return []
    t = line_times(rec)
    keep = np.argsort(-rel)[:top_lines]
    w = np.exp((rel[keep] - rel[keep].max()) / temperature)
    w /= w.sum()
    grid = np.arange(0.0, rec["duration_s"], GRID_S)
    dens = np.zeros_like(grid)
    centres = (np.arange(N_BINS) - N_BINS // 2) * OFF_BIN_S
    for wi, i in zip(w, keep, strict=True):
        d = grid - t[i]
        if off is None:
            dens += wi * np.exp(-0.5 * (d / sigma_s) ** 2)
        else:
            # mixture of Gaussians at the bin centres (width = half a bin) weighted by the offset distribution
            dens += wi * (off[i][None, :] * np.exp(-0.5 * ((d[:, None] - centres[None, :]) / (OFF_BIN_S / 2)) ** 2)).sum(1)
    order = np.argsort(-dens, kind="stable")
    return nms([float(grid[j]) for j in order[:2000]], NMS_GAP_S)[:16]


def propose(method: str, rec, rel: np.ndarray, off: np.ndarray | None) -> list[float]:
    t = line_times(rec)
    if method == "ft_rel":
        return [float(t[i]) for i in np.argsort(-rel, kind="stable")[:64]]
    if method in ("ft_rel_dens", "zs_dens"):
        return density_moments(rec, rel, None)
    if method == "ft_rel_off":
        return density_moments(rec, rel, off)
    raise ValueError(method)


# -- training ---------------------------------------------------------------------------------------
def build_examples(recs, zs_rel, use_weak: bool):
    ex = []
    for r in recs:
        if r["weak"] and not use_weak:
            continue
        near, bins = labels(r)
        if not near.any():
            continue
        ex.append({"rec": r, "near": near, "bins": bins, "zs": zs_rel[r["qa_id"]],
                   "weight": WEAK_WEIGHT if r["weak"] else 1.0})
    return ex


def train_one(a) -> dict:
    torch = __import__("torch")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    zs_rel = zeroshot_rel()
    train = load("train")
    tr = build_examples([r for r in train if not inner_val(r["video_id"])], zs_rel, not a.no_weak)
    va = [r for r in train if inner_val(r["video_id"]) and not r["weak"]]
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    use_off = a.method == "ft_rel_off"
    best, best_state, history = -1.0, None, []
    for epoch in range(a.epochs):
        model.enc.train()
        random.shuffle(tr)
        tic, losses = time.perf_counter(), []
        for e in tr:
            r, ctx = e["rec"], contexts(e["rec"])
            pos = np.flatnonzero(e["near"])
            neg = np.flatnonzero(~e["near"])
            if len(neg) == 0:
                continue
            hard = [i for i in np.argsort(-e["zs"]) if not e["near"][i]][: a.hard]
            rand = list(np.random.choice(neg, size=min(a.random_neg, len(neg)), replace=False))
            negs = list(dict.fromkeys(hard + rand))
            if a.loss == "mil":
                # multiple-instance: the near lines form a bag; the loss rewards the bag's best lines, so the
                # model may rank whichever near line really relates to the question (others can be chatter)
                bag = [int(i) for i in np.random.permutation(pos)[: a.bag]]
            else:
                bag = [int(np.random.choice(pos))]
            # near lines outside the bag also get offset supervision
            extra = [int(i) for i in np.flatnonzero(e["bins"] >= 0) if i not in bag and i not in negs][:4]
            idx = bag + negs + extra
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rel, off = model.forward(query_text(r, a.query), [ctx[i] for i in idx])
            rel = rel.float()
            scored = rel[: len(bag) + len(negs)]
            loss = torch.logsumexp(scored, 0) - torch.logsumexp(scored[: len(bag)], 0)
            if use_off:
                tb = torch.as_tensor(e["bins"][idx], device=rel.device)
                m = tb >= 0
                if m.any():
                    loss = loss + a.off_weight * torch.nn.functional.cross_entropy(off.float()[m], tb[m])
            loss = e["weight"] * loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        score = evaluate(model, va, a.method, a.query)["recall@6"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "inner_val_recall@6": score,
                        "seconds": time.perf_counter() - tic})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best, best_state = score, {k: {n: v.detach().cpu().clone() for n, v in s.items()}
                                       for k, s in model.state().items()}
    model.load_state(best_state)
    dev = evaluate(model, load("dev"), a.method, a.query)
    return {"method": a.method, "seed": a.seed, "query": a.query, "weak": not a.no_weak, "lr": a.lr, "loss": a.loss,
            "epochs": a.epochs, "best_inner_val_recall@6": best, "history": history,
            "train_questions": len(tr), "inner_val_questions": len(va), "dev": dev}


def evaluate(model: Model, recs, method: str, query: str) -> dict:
    props = []
    for r in recs:
        rel, off = model.predict(query_text(r, query), contexts(r))
        props.append(propose(method, r, rel, off))
    return score_metrics(recs, props)


def stage_train(a) -> None:
    res = train_one(a)
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    d = res["dev"]
    print(f"DEV {a.method} seed={a.seed} " + " ".join(f"R@{k}={d[f'recall@{k}']:.3f}" for k in KS))


def stage_zs_dens(a) -> None:
    zs_rel = zeroshot_rel()
    recs = load("dev")
    res = score_metrics(recs, [propose("zs_dens", r, zs_rel[r["qa_id"]], None) for r in recs])
    with open(RUN / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"method": "zs_dens", "seed": 0, "query": "q", "weak": None, "dev": res}) + "\n")
    print("DEV zs_dens " + " ".join(f"R@{k}={res[f'recall@{k}']:.3f}" for k in KS))


def stage_report(a) -> None:
    rows = [json.loads(x) for x in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["method"], r["query"], r["weak"]), []).append(r)
    table = []
    for (m, q, w), g in groups.items():
        row = {"method": m, "query": q, "weak": w, "seeds": len(g)}
        for k in KS:
            v = [x["dev"][f"recall@{k}"] for x in g]
            row[f"recall@{k}"], row[f"recall@{k}_sd"] = float(np.mean(v)), float(np.std(v))
        table.append(row)
        print(f"{m:12s} q={q:2s} weak={w!s:5s} seeds={len(g)} "
              + " ".join(f"R@{k}={row[f'recall@{k}']:.3f}±{row[f'recall@{k}_sd']:.3f}" for k in KS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "learned_dev.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "train", "zs_dens", "report"))
    p.add_argument("--method", default="ft_rel_off", choices=("ft_rel", "ft_rel_off", "ft_rel_dens"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--query", default="q", choices=("q", "qo"))
    p.add_argument("--no-weak", action="store_true", help="CG-Bench human intervals only")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--hard", type=int, default=8)
    p.add_argument("--random-neg", type=int, default=8)
    p.add_argument("--off-weight", type=float, default=0.5)
    p.add_argument("--loss", default="mil", choices=("mil", "single"),
                   help="mil: near lines as a bag (multiple-instance); single: one random near line is the positive")
    p.add_argument("--bag", type=int, default=8)
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"prepare": stage_prepare, "train": stage_train, "zs_dens": stage_zs_dens, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
