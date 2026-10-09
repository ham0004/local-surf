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
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_head_a_eval import KS, load, score_metrics  # noqa: E402
from v3_head_a_train import N_BINS, density_moments, inner_val, labels, line_times, zeroshot_rel  # noqa: E402

RUN = Path("runs/v3_head_a")
OUT = Path("reports/v3_head_a")
CUES = {
    "speech": ("say", "said", "says", "mention", "subtitle", "explain", "why", "talk", "according", "narrator"),
    "visual": ("color", "colour", "wear", "how many", "shape", "letters", "sign", "screen", "pattern", "logo",
               "written", "appear", "shown", "picture"),
    "order": ("after", "before", "when", "first", "last", "then", "finally"),
}
FEATURE_NAMES = ["z", "z_gap", "z_rank", "ctx1_max", "ctx1_mean", "ctx3_max", "ctx3_mean", "ctx7_max", "ctx7_mean",
                 "bm25_q", "bm25_qo", "opt_best", "opt_spread", "words", "digit", "position", "density",
                 "cue_speech", "cue_visual", "cue_order", "has_options"]
_WORD = re.compile(r"[a-z0-9]+")


def _toks(t: str) -> set[str]:
    return {w for w in _WORD.findall(t.lower()) if len(w) > 2}


def line_features(rec, z: np.ndarray) -> np.ndarray:
    from videoqa.retrieval import bm25_scores, tokenize  # noqa: PLC0415

    lines, n = rec["lines"], len(rec["lines"])
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    texts = [x[2] for x in lines]
    order = np.argsort(np.argsort(-z))
    ctx = []
    for w in (1, 3, 7):
        mx = np.array([z[max(0, i - w): i + w + 1].max() for i in range(n)])
        mn = np.array([z[max(0, i - w): i + w + 1].mean() for i in range(n)])
        ctx += [mx - z.max(), mn - z.max()]
    corpus = [tokenize(t) for t in texts]
    bq = np.asarray(bm25_scores(tokenize(rec["question"]), corpus))
    bqo = np.asarray(bm25_scores(tokenize(rec["question"] + " " + " ".join(rec["options"] or [])), corpus))
    opts = [_toks(o) for o in (rec["options"] or [])]
    ov = np.array([[len(_toks(t) & o) / (len(o) + 1) for o in opts] for t in texts]) if opts else np.zeros((n, 1))
    t = line_times(rec)
    dens = np.array([np.sum(np.abs(t - x) <= 15.0) for x in t])
    q = rec["question"].lower()
    cue = [float(any(c in q for c in CUES[k])) for k in ("speech", "visual", "order")]
    rows = np.column_stack([
        z, z - z.max(), order / max(n - 1, 1), *ctx,
        bq / (bq.max() + 1e-6), bqo / (bqo.max() + 1e-6), ov.max(1), ov.max(1) - ov.min(1),
        np.log1p([len(x.split()) for x in texts]), [float(bool(re.search(r"\d", x))) for x in texts],
        t / max(rec["duration_s"], 1.0), np.log1p(dens),
        np.tile(cue, (n, 1)), np.full(n, float(bool(rec["options"])))])
    return rows.astype(np.float32)


def stage_features(a) -> None:
    zs = zeroshot_rel()
    out = {}
    for split in ("train", "dev"):
        for r in load(split):
            out[r["qa_id"].replace(":", "__")] = line_features(r, zs[r["qa_id"]])
    np.savez_compressed(RUN / "light_features.npz", **out)
    print(f"features for {len(out)} questions, {len(FEATURE_NAMES)} per line")


def stage_train(a) -> None:
    import torch  # noqa: PLC0415

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    z = np.load(RUN / "light_features.npz")
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
    d = len(FEATURE_NAMES)
    net = torch.nn.Sequential(torch.nn.Linear(d, a.width), torch.nn.GELU(), torch.nn.Linear(a.width, a.width),
                              torch.nn.GELU(), torch.nn.Linear(a.width, 1 + (N_BINS if use_off else 0)))
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
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
    res = {"method": a.method, "seed": a.seed, "weak": not a.no_weak, "width": a.width, "lr": a.lr,
           "epochs": a.epochs, "best_inner_val_recall@6": best,
           "chosen_epoch": max(hist, key=lambda h: h["inner_val_recall@6"])["epoch"], "history": hist,
           "dev": evaluate(dev)}
    with open(RUN / "light_results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    dv = res["dev"]
    print(f"DEV {a.method} seed={a.seed} weak={not a.no_weak} epoch={res['chosen_epoch']} "
          f"inner={best:.3f} " + " ".join(f"R@{k}={dv[f'recall@{k}']:.3f}" for k in KS), flush=True)


def stage_report(a) -> None:
    rows = [json.loads(x) for x in (RUN / "light_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["method"], r["weak"]), []).append(r)
    table = []
    for (m, w), g in groups.items():
        row = {"method": m, "weak": w, "seeds": len(g)}
        for k in KS:
            v = [x["dev"][f"recall@{k}"] for x in g]
            row[f"recall@{k}"], row[f"recall@{k}_sd"] = float(np.mean(v)), float(np.std(v))
        table.append(row)
        print(f"{m:10s} weak={w!s:5s} seeds={len(g)} "
              + " ".join(f"R@{k}={row[f'recall@{k}']:.3f}±{row[f'recall@{k}_sd']:.3f}" for k in KS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "light_dev.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("features", "train", "report"))
    p.add_argument("--method", default="light", choices=("light", "light_dens", "light_off"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-weak", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--off-weight", type=float, default=0.5)
    a = p.parse_args()
    {"features": stage_features, "train": stage_train, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
