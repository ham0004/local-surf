"""Head B cycle: choose the final K frames as a SET from the pooled Path A + Path B candidates.

Head B replaces box [5] of the frozen baseline (MMR on MobileCLIP similarity to question + options, K = 4).
Every method writes its choice per question to runs/v3_head_b/selections/<dataset>_<pool>_<method>.json;
`v3_baseline_cycle.py answer --selectors file:<dataset>_<pool>_<method>` then answers with exactly the same
frozen answerer, prompt and cache as every other trial.

Per-candidate features (label-free; MobileCLIP image/text space, computed from the saved pools):
    sim_q, sim_qo (raw and z-scored within the pool, rank); option profile over "question + option i" texts:
    max, mean, max - mean, max - second; provenance (Path A, Path B, both); Path A speech score (z within
    pool); speech near the frame vs question (MobileCLIP text-text); relative time; redundancy (max
    similarity to another candidate, candidates within 10 s); pool size.

Methods:
    optset     training-free option-evidence set rule (the proposed ingredient): greedy marginal gain
               rel(c) + lam * [decisiveness(S + c) - decisiveness(S)] - mu * max_sim(c, S), where the set's
               option distribution is the mean of its frames' softmax option profiles and decisiveness is
               its top-1 minus top-2 probability
    ev         learned evidence scorer (MLP on the features, human clue intervals of CG-Bench train), then
               MMR (0.7 / 0.3) on the learned score
    ev_set     ev + the option-evidence set term of optset
Ablations follow from the parameters (lam = 0: no option term; mu = 0: no set context).

    python scripts/v3_head_b.py features --dataset cgbench_train --pool hybrid
    python scripts/v3_head_b.py features --dataset cgbench --pool hybrid
    python scripts/v3_head_b.py train --seed 0                   # ev scorer on cgbench_train
    python scripts/v3_head_b.py select --dataset cgbench --pool hybrid --method ev_set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN = Path("runs/v3_head_b")
FEATURE_NAMES = ["sim_q", "sim_qo", "z_q", "z_qo", "rank_qo", "opt_max", "opt_mean", "opt_gap_mean", "opt_gap_2",
                 "is_A", "is_B", "is_both", "speech_z", "near_q", "position", "redund", "close10", "pool_n"]
TAU = 50.0            # softmax temperature on MobileCLIP cosines for option profiles


def pool_dir(dataset: str, pool: str) -> Path:
    return Path("runs/v3_baseline") / dataset / pool / "pools"


def inner_val(video_id: str) -> bool:
    return int(hashlib.sha256(f"headb-inner:{video_id}".encode()).hexdigest(), 16) % 100 < 20


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def pool_features(pool, q_emb, qo_emb, opt_embs) -> tuple[np.ndarray, np.ndarray]:
    """(n, len(FEATURE_NAMES)) features and (n, options) softmax option profiles for one pool."""
    c = pool.candidates
    n = len(c)
    E = np.stack([np.asarray(x.emb, np.float32) for x in c])
    sq, sqo = E @ q_emb, E @ qo_emb
    so = E @ opt_embs.T                                           # (n, options)
    prof = np.exp(TAU * (so - so.max(1, keepdims=True)))
    prof /= prof.sum(1, keepdims=True)
    srt = np.sort(so, 1)[:, ::-1]
    second = srt[:, 1] if so.shape[1] > 1 else srt[:, 0]
    is_a = np.array([float("A" in x.paths) for x in c])
    is_b = np.array([float("B" in x.paths) for x in c])
    speech = np.array([x.head_a_visual for x in c], np.float32)
    near = np.array([float(np.asarray(x.near_emb) @ q_emb) if x.near_emb is not None and np.any(x.near_emb) else 0.0
                     for x in c])
    t = np.array([x.time_s for x in c])
    sim = E @ E.T
    np.fill_diagonal(sim, -1)
    close = np.array([np.sum(np.abs(t - x) <= 10.0) - 1 for x in t])
    f = np.column_stack([sq, sqo, _z(sq), _z(sqo), np.argsort(np.argsort(-sqo)) / max(n - 1, 1),
                         so.max(1), so.mean(1), so.max(1) - so.mean(1), so.max(1) - second,
                         is_a, is_b, is_a * is_b, _z(speech), near, t / max(pool.duration_s, 1.0),
                         sim.max(1) if n > 1 else np.zeros(n), close, np.full(n, np.log(n))])
    return f.astype(np.float32), prof.astype(np.float32)


def stage_features(a) -> None:
    from v3_baseline_cycle import items  # noqa: PLC0415

    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    meta = json.loads((pool_dir(a.dataset, a.pool).parent / "pools.json").read_text())
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    qa = {it.qa.qa_id: it.qa for it in items(a.dataset)}
    out = {}
    for qid in meta["qa_ids"]:
        p = load_pool(pool_dir(a.dataset, a.pool), qid)
        if not p.candidates:
            continue
        t = enc.embed_texts([p.question, p.question + " Options: " + ", ".join(p.options)]
                            + [p.question + " " + o for o in p.options])
        f, prof = pool_features(p, t[0], t[1], t[2:])
        iv = qa[qid].evidence_intervals_s or []
        lab = np.array([any(x - 1.0 <= c.time_s <= y + 1.0 for x, y in iv) for c in p.candidates], np.float32)
        E = np.stack([np.asarray(c.emb, np.float32) for c in p.candidates])
        out[qid] = {"ids": [c.id for c in p.candidates], "f": f, "prof": prof, "label": lab, "emb": E,
                    "video_id": p.video_id, "gold": p.gold_option_index}
    RUN.mkdir(parents=True, exist_ok=True)
    np.save(RUN / f"features_{a.dataset}_{a.pool}.npy", out, allow_pickle=True)
    print(f"{len(out)} pools; evidence candidates per pool {np.mean([v['label'].sum() for v in out.values()]):.2f}")


def load_features(dataset: str, pool: str) -> dict:
    return np.load(RUN / f"features_{dataset}_{pool}.npy", allow_pickle=True).item()


# -- set selection ------------------------------------------------------------------------------------
def decisiveness(prof_rows: np.ndarray) -> float:
    p = prof_rows.mean(0)
    s = np.sort(p)[::-1]
    return float(s[0] - (s[1] if len(s) > 1 else 0.0))


def greedy(rel: np.ndarray, emb: np.ndarray, prof: np.ndarray, k: int, lam: float, mu: float) -> list[int]:
    chosen: list[int] = []
    rest = list(range(len(rel)))
    while rest and len(chosen) < k:
        def gain(i):
            red = max((float(emb[i] @ emb[j]) for j in chosen), default=0.0)
            dec = decisiveness(prof[chosen + [i]]) - (decisiveness(prof[chosen]) if chosen else 0.0)
            return rel[i] + lam * dec - mu * red
        best = max(rest, key=gain)
        chosen.append(best)
        rest.remove(best)
    return chosen


# -- learned evidence scorer ----------------------------------------------------------------------------
def build_scorer(d: int, width: int = 32):
    import torch  # noqa: PLC0415

    return torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Linear(width, 1))


def train_scorer(data: dict, seed: int, epochs: int = 200, lr: float = 3e-3, wd: float = 1e-3):
    """Listwise (softmax over each pool) loss towards the evidence candidates; returns (net, mu, sd, history)."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    tr = {q: v for q, v in data.items() if not inner_val(v["video_id"]) and v["label"].any()}
    va = {q: v for q, v in data.items() if inner_val(v["video_id"]) and v["label"].any()}
    allf = np.concatenate([v["f"] for v in tr.values()])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    net = build_scorer(allf.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    T = [(torch.as_tensor((v["f"] - mu) / sd), torch.as_tensor(v["label"] > 0)) for v in tr.values()]

    def val_recall(net):
        hits = []
        with torch.no_grad():
            for v in va.values():
                s = net(torch.as_tensor((v["f"] - mu) / sd))[:, 0].numpy()
                hits.append(bool(v["label"][np.argsort(-s)[:4]].any()))
        return float(np.mean(hits))

    best, state, hist = -1.0, None, []
    for ep in range(epochs):
        net.train()
        for x, y in T:
            s = net(x)[:, 0]
            loss = torch.logsumexp(s, 0) - torch.logsumexp(s[y], 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 9:
            net.eval()
            r = val_recall(net)
            hist.append({"epoch": ep + 1, "inner_val_recall@4": r})
            if r > best:
                best, state = r, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    return net, mu, sd, hist, best


def stage_train(a) -> None:
    import torch  # noqa: PLC0415

    data = load_features("cgbench_train", a.pool)
    net, mu, sd, hist, best = train_scorer(data, a.seed)
    (RUN / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "hist": hist, "inner_val_recall@4": best},
               RUN / "ckpt" / f"ev_{a.pool}_{a.seed}.pt")
    print(f"seed {a.seed}: inner validation recall@4 {best:.3f} (epoch {max(hist, key=lambda h: h['inner_val_recall@4'])['epoch']})")


def learned_scores(v, pool: str, seeds=(0, 1, 2)) -> np.ndarray:
    import torch  # noqa: PLC0415

    out = []
    for s in seeds:
        ck = torch.load(RUN / "ckpt" / f"ev_{pool}_{s}.pt", map_location="cpu", weights_only=False)
        net = build_scorer(len(ck["mu"]))
        net.load_state_dict(ck["state"])
        net.eval()
        with torch.no_grad():
            out.append(net(torch.as_tensor((v["f"] - ck["mu"]) / ck["sd"]))[:, 0].numpy())
    return np.mean(out, 0)


def choose(v, method: str, k: int, lam: float, mu: float, train_pool: str) -> list[int]:
    if method == "optset":
        rel = _z(v["f"][:, FEATURE_NAMES.index("sim_qo")])
        return greedy(rel, v["emb"], v["prof"], k, lam, mu)
    rel = _z(learned_scores(v, train_pool))
    if method == "ev":
        return greedy(rel, v["emb"], v["prof"], k, 0.0, mu)
    if method == "ev_set":
        return greedy(rel, v["emb"], v["prof"], k, lam, mu)
    raise ValueError(method)


def stage_select(a) -> None:
    data = load_features(a.dataset, a.pool)
    sel, hits = {}, []
    for q, v in data.items():
        idx = choose(v, a.method, a.k, a.lam, a.mu, a.train_pool)
        sel[q] = [v["ids"][i] for i in idx]
        hits.append(bool(v["label"][idx].any()))
    name = f"{a.dataset}_{a.pool}_{a.method}" + (a.tag and f"_{a.tag}")
    (RUN / "selections").mkdir(parents=True, exist_ok=True)
    (RUN / "selections" / f"{name}.json").write_text(json.dumps(sel), encoding="utf-8")
    print(f"{name}: chosen evidence recall@{a.k} {np.mean(hits):.3f} (n={len(hits)}; only meaningful with labels)")


def stage_tune(a) -> None:
    """Choose lam / mu for optset and ev_set on the INNER VALIDATION part of cgbench_train (evidence recall@4)."""
    data = {q: v for q, v in load_features("cgbench_train", a.pool).items() if inner_val(v["video_id"])}
    grid = [(lam, mu) for lam in (0.0, 0.5, 1.0, 2.0, 4.0) for mu in (0.0, 0.3, 0.6, 1.0)]
    for method in ("optset", "ev_set"):
        res = []
        for lam, mu in grid:
            hits = [bool(v["label"][choose(v, method, 4, lam, mu, a.pool)].any()) for v in data.values()]
            res.append((float(np.mean(hits)), lam, mu))
        res.sort(reverse=True)
        print(method, "best (recall, lam, mu):", res[:4], "| lam=0, mu=0.3:",
              [r for r in res if r[1] == 0.0 and r[2] == 0.3])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("features", "train", "tune", "select"))
    p.add_argument("--dataset", default="cgbench")
    p.add_argument("--pool", default="hybrid")
    p.add_argument("--train-pool", default="hybrid")
    p.add_argument("--method", default="ev_set", choices=("optset", "ev", "ev_set"))
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"features": stage_features, "train": stage_train, "tune": stage_tune, "select": stage_select}[a.stage](a)


if __name__ == "__main__":
    main()
