"""Head A cycle, method family 2: a light learned head over FROZEN line features (the encoder is not tuned).

Step 37 showed that fine-tuning the MiniLM cross-encoder on near-evidence lines destroys its relevance
ranking (inner validation recall@6 0.45 -> 0.19). This family keeps the frozen zero-shot score z_i and learns
only a residual on top of it, initialised at zero, so the untrained head IS the zero-shot rule:

    score_i = z_i + f(features_i)            f: MLP 32 -> 32 -> 1, last layer zero-initialised

Per-line features (all label-free at inference):
    zero-shot logit, its gap to the question's best line, rank percentile;
    local relevance context: max / mean of z over +/-1, +/-3, +/-7 lines (several lines talking about it);
    BM25 of the line against the question and against question + options (each / question max);
    best and spread of option-word overlap; words in the line; has a digit; relative position in the video;
    speech density (lines within +/-15 s);
    question cues (the same for all lines of a question): asks about speech ("say", "mention", "subtitle",
    "explain", "why"), about appearance ("color", "wear", "how many", "shape", "letters", "sign", "screen"),
    about order/time ("after", "before", "when", "first", "last"); has options.

Loss: multiple-instance listwise over ALL lines of the question: -log sum_{near} softmax(score) (near = within
10 s of the evidence). Optional offset head (method "light_off"): a distribution over where the evidence is
relative to the line (-40..+40 s, 4 s bins) from the same features; moments = peaks of the evidence density
(as v3_head_a_train.density_moments). Inner validation (15% of train videos) picks the epoch; 3 seeds.

    python scripts/v3_head_a_light.py features                  # caches per-line features (CPU)
    python scripts/v3_head_a_light.py train --method light --seed 0 [--no-weak]
    python scripts/v3_head_a_light.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_head_a_eval import KS, load, score_metrics  # noqa: E402
from v3_head_a_train import inner_val, labels, zeroshot_rel  # noqa: E402

from videoqa.v3.head_a import (  # noqa: E402
    bge_features,
    build_net,
    density_moments,
    line_features,
    line_times,
)

RUN = Path("runs/v3_head_a")
BGE = "BAAI/bge-reranker-base"           # revision 2cfc18c9415c912f9d8155881c133215df768a70 (step 38)
OUT = Path("reports/v3_head_a")
def stage_bge(a) -> None:
    """Second frozen scorer for the 'ens' feature set: bge-reranker-base on question + options (step 38's best)."""
    from v3_head_a_eval import Reranker, segments  # noqa: PLC0415

    from videoqa.v2.candidates import _with_neighbours  # noqa: PLC0415

    rr = Reranker(BGE)
    out = {}
    for split in ("train", "dev"):
        for r in load(split):
            segs = segments(r)
            q = r["question"] + (" Options: " + " | ".join(r["options"]) if r["options"] else "")
            out[r["qa_id"].replace(":", "__")] = rr.score(q, [_with_neighbours(segs, x) for x in segs]).astype(np.float32)
    np.savez_compressed(RUN / "bge_rel.npz", **out)
    print(f"bge scores for {len(out)} questions")


def bge_features(z2: np.ndarray) -> np.ndarray:
    n = len(z2)
    ctx3 = np.array([z2[max(0, i - 3): i + 4].max() for i in range(n)])
    order = np.argsort(np.argsort(-z2))
    return np.column_stack([z2 - z2.max(), order / max(n - 1, 1), ctx3 - z2.max()]).astype(np.float32)


def stage_features(a) -> None:
    zs = zeroshot_rel()
    bge = np.load(RUN / "bge_rel.npz") if a.feature_set == "ens" else None
    out = {}
    for split in ("train", "dev"):
        for r in load(split):
            f = line_features(r, zs[r["qa_id"]])
            if bge is not None and len(f):
                f = np.concatenate([f, bge_features(bge[r["qa_id"].replace(":", "__")])], axis=1)
            out[r["qa_id"].replace(":", "__")] = f
    np.savez_compressed(RUN / f"light_features{'' if a.feature_set == 'base' else '_' + a.feature_set}.npz", **out)
    print(f"features for {len(out)} questions ({a.feature_set})")


def stage_train(a) -> None:
    import torch  # noqa: PLC0415

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    z = np.load(RUN / f"light_features{'' if a.feature_set == 'base' else '_' + a.feature_set}.npz")
    feats = {k.replace("__", ":"): z[k] for k in z.files}
    train = [r for r in load("train") if len(r["lines"]) and (not r["weak"] or not a.no_weak)]
    tr = [r for r in train if not inner_val(r["video_id"])]
    va = [r for r in train if inner_val(r["video_id"]) and not r["weak"]]
    dev = load("dev")
    allf = np.concatenate([feats[r["qa_id"]] for r in tr])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    use_off = a.method == "light_off"

    def tens(r):
        return torch.as_tensor((feats[r["qa_id"]] - mu) / sd), torch.as_tensor(feats[r["qa_id"]][:, 0])

    data = []
    for r in tr:
        near, bins = labels(r)
        if near.any():
            x, z0 = tens(r)
            data.append((x, z0, torch.as_tensor(near), torch.as_tensor(bins), 0.5 if r["weak"] else 1.0))
    net = build_net(next(iter(feats.values())).shape[1], a.width, use_off)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    def predict(r):
        x, z0 = tens(r)
        with torch.no_grad():
            o = net(x)
        rel = (z0 + o[:, 0]).numpy()
        off = torch.softmax(o[:, 1:], -1).numpy() if use_off else None
        return rel, off

    def proposals(r):
        rel, off = predict(r)
        if a.method == "light":
            return [float(x) for x in line_times(r)[np.argsort(-rel, kind="stable")[:64]]]
        return density_moments(r, rel, off if use_off else None)

    def evaluate(recs):
        return score_metrics(recs, [proposals(r) for r in recs])

    best, best_state, hist = evaluate(va)["recall@6"], None, []
    hist.append({"epoch": 0, "inner_val_recall@6": best})
    for epoch in range(1, a.epochs + 1):
        net.train()
        np.random.shuffle(data)
        tot = 0.0
        for x, z0, near, bins, w in data:
            o = net(x)
            s = z0 + o[:, 0]
            loss = torch.logsumexp(s, 0) - torch.logsumexp(s[near], 0)
            if use_off and (bins >= 0).any():
                m = bins >= 0
                loss = loss + a.off_weight * torch.nn.functional.cross_entropy(o[m, 1:], bins[m])
            loss = w * loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        net.eval()
        score = evaluate(va)["recall@6"]
        hist.append({"epoch": epoch, "loss": tot / len(data), "inner_val_recall@6": score})
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    res = {"method": a.method, "seed": a.seed, "weak": not a.no_weak, "features": a.feature_set,
           "width": a.width, "lr": a.lr,
           "epochs": a.epochs, "best_inner_val_recall@6": best,
           "chosen_epoch": max(hist, key=lambda h: h["inner_val_recall@6"])["epoch"], "history": hist,
           "dev": evaluate(dev)}
    if a.save:
        ck = RUN / "ckpt" / f"{a.method}_{a.feature_set}_{'weak' if not a.no_weak else 'human'}_{a.seed}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": a.method, "feature_set": a.feature_set, "width": a.width, "mu": mu, "sd": sd,
                    "state": net.state_dict(), "dev": {k: v for k, v in res["dev"].items() if k != "_hits6"}}, ck)
        res["checkpoint"] = str(ck)
    with open(RUN / "light_results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    dv = res["dev"]
    print(f"DEV {a.method} {a.feature_set} seed={a.seed} weak={not a.no_weak} epoch={res['chosen_epoch']} "
          f"inner={best:.3f} " + " ".join(f"R@{k}={dv[f'recall@{k}']:.3f}" for k in KS), flush=True)


def stage_report(a) -> None:
    rows = [json.loads(x) for x in (RUN / "light_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["method"], r["weak"], r.get("features", "base")), []).append(r)
    table = []
    for (m, w, fs), g in groups.items():
        row = {"method": m, "weak": w, "features": fs, "seeds": len(g)}
        for k in KS:
            v = [x["dev"][f"recall@{k}"] for x in g]
            row[f"recall@{k}"], row[f"recall@{k}_sd"] = float(np.mean(v)), float(np.std(v))
        table.append(row)
        print(f"{m:10s} {fs:4s} weak={w!s:5s} seeds={len(g)} "
              + " ".join(f"R@{k}={row[f'recall@{k}']:.3f}±{row[f'recall@{k}_sd']:.3f}" for k in KS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "light_dev.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("bge", "features", "train", "report"))
    p.add_argument("--feature-set", default="base", choices=("base", "ens"),
                   help="ens = base + bge-reranker-base relevance features")
    p.add_argument("--method", default="light", choices=("light", "light_dens", "light_off"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-weak", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--off-weight", type=float, default=0.5)
    p.add_argument("--save", action="store_true", help="write the trained head to runs/v3_head_a/ckpt/")
    a = p.parse_args()
    {"bge": stage_bge, "features": stage_features, "train": stage_train, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
