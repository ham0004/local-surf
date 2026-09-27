"""Train V0 / V1 / V2 utility heads on assay-v2 labels and calibrate lambda on dev.

Variants (identical rows, optimiser, seed and features; only the loss differs):

    V0  absolute gain only                       Huber(u - g)
    V1  V0 + targeted-minus-control difference   + Huber((u_T - u_C) - (g_T - g_C))
    V2  V1 + clean->targeted and clean->control  + Huber((u_T - u_K) - (g_T - g_K))
        differences + a ranking term               + Huber((u_C - u_K) - (g_C - g_K))
                                                   + softplus ranking where g_T != g_C

Only exact-key visual triplets (assay.build_triplets) enter the difference
and ranking terms; EXPAND rows train only the absolute term.

Decision rule at inference (controller._UtilityController):
    net(a) = u(a) - lambda * cost_ms(a) / 1000,  STOP if max net <= 0.

Calibration: the declared PRICE of time (``price_per_s``, from the config's
cost model) is fixed in advance. On dev decision points (question, condition,
history) we sweep the controller's decision lambda and score each decision by
what it ACTUALLY produced in the labels: realised gain - price * cost. The
lambda with the best mean realised utility wins; the full sweep is saved as a
gain-vs-cost Pareto curve. Test labels are never read here.

Metrics reported on dev (never test): gain MAE, targeted-minus-control
difference MAE, ranking accuracy on triplets where the measured gains differ,
realised gain / cost / utility per condition at the chosen lambda.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assay import AssayRow, read_assay
from .heads import TrainConfig, UtilityHead

VARIANTS = ("V0", "V1", "V2")
LAMBDA_GRID = tuple(float(x) for x in np.concatenate([[0.0], np.geomspace(0.01, 10.0, 25)]))


def arrays(rows: list[AssayRow]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([r.features for r in rows], dtype=np.float64),
            np.asarray([r.gain for r in rows], dtype=np.float64))


def triplet_terms(rows: list[AssayRow], triplets: list[dict]):
    tc = np.asarray([(t["targeted"], t["control"]) for t in triplets], dtype=int).reshape(-1, 2)
    tk = np.asarray([(t["targeted"], t["clean"]) for t in triplets], dtype=int).reshape(-1, 2)
    ck = np.asarray([(t["control"], t["clean"]) for t in triplets], dtype=int).reshape(-1, 2)
    rank = []
    for t in triplets:
        gt, gc = rows[t["targeted"]].gain, rows[t["control"]].gain
        if gt != gc:
            rank.append((t["targeted"], t["control"]) if gt > gc else (t["control"], t["targeted"]))
    return tc, tk, ck, np.asarray(rank, dtype=int).reshape(-1, 2)


def fit_variant(variant: str, rows: list[AssayRow], triplets: list[dict], cfg: TrainConfig,
                w_pair: float = 1.0, w_clean: float = 0.5, w_rank: float = 0.2) -> tuple[UtilityHead, list[float]]:
    X, y = arrays(rows)
    tc, tk, ck, rank = triplet_terms(rows, triplets)
    head = UtilityHead(X.shape[1], cfg.hidden, cfg.seed)
    c = dataclasses.replace(cfg, lambda_pair=0.0 if variant == "V0" else w_pair)
    extra = [(tk, w_clean), (ck, w_clean)] if variant == "V2" else None
    curve = head.fit(X, y, tc, c, extra_pairs=extra, rank_pairs=rank if variant == "V2" else None,
                     lambda_rank=w_rank if variant == "V2" else 0.0)
    return head, curve


def decision_groups(rows: list[AssayRow]) -> dict[tuple, list[int]]:
    g: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        g[(r.qa_id, r.condition, r.history_id)].append(i)
    return g


def simulate(pred: np.ndarray, rows: list[AssayRow], lam: float, price: float) -> dict:
    """Controller decisions at decision-lambda ``lam``, scored at ``price``."""
    per_cond: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (_, cond, _), idx in decision_groups(rows).items():
        cost_s = np.array([rows[i].cost_ms / 1000.0 for i in idx])
        net = pred[idx] - lam * cost_s
        b = int(np.argmax(net))
        if net[b] <= 0.0:
            per_cond[cond].append((0.0, 0.0))                   # STOP
        else:
            per_cond[cond].append((rows[idx[b]].gain, cost_s[b]))
    out = {}
    for cond, v in per_cond.items():
        g = np.array([a for a, _ in v])
        c = np.array([b for _, b in v])
        out[cond] = {"n": len(v), "gain": float(g.mean()), "cost_s": float(c.mean()),
                     "act_rate": float((c > 0).mean()), "utility": float((g - price * c).mean())}
    allv = [x for v in per_cond.values() for x in v]
    g = np.array([a for a, _ in allv])
    c = np.array([b for _, b in allv])
    out["all"] = {"n": len(allv), "gain": float(g.mean()), "cost_s": float(c.mean()),
                  "act_rate": float((c > 0).mean()), "utility": float((g - price * c).mean())}
    return out


def evaluate_head(head: UtilityHead, rows: list[AssayRow], triplets: list[dict]) -> dict:
    X, y = arrays(rows)
    pred = head.predict(X)
    out = {"gain_mae": float(np.abs(pred - y).mean())}
    if triplets:
        t = np.array([x["targeted"] for x in triplets])
        c = np.array([x["control"] for x in triplets])
        out["tc_diff_mae"] = float(np.abs((pred[t] - pred[c]) - (y[t] - y[c])).mean())
        differ = y[t] != y[c]
        out["tc_rank_n"] = int(differ.sum())
        out["tc_rank_acc"] = (float((np.sign(pred[t] - pred[c])[differ] == np.sign(y[t] - y[c])[differ]).mean())
                              if differ.any() else None)
    return out


def calibrate(head: UtilityHead, rows: list[AssayRow], price: float, grid=LAMBDA_GRID) -> tuple[float, list[dict]]:
    X, _ = arrays(rows)
    pred = head.predict(X)
    curve = [{"lambda": lam, **simulate(pred, rows, lam, price)["all"]} for lam in grid]
    # ties -> larger lambda (cheaper policy)
    best = max(range(len(curve)), key=lambda i: (round(curve[i]["utility"], 12), curve[i]["lambda"]))
    return curve[best]["lambda"], curve


def train_assay(train_dir: str | Path, dev_dir: str | Path, out_dir: str | Path, cfg: TrainConfig,
                price_per_s: float, seeds: tuple[int, ...] = (0, 1, 2), variants=VARIANTS) -> dict:
    tr_rows, tr_trip = read_assay(train_dir)
    dv_rows, dv_trip = read_assay(dev_dir)
    for r in tr_rows + dv_rows:
        if r.experiment_split == "test":
            raise ValueError("test rows must never be used for training or calibration")
    overlap = {r.video_id for r in tr_rows} & {r.video_id for r in dv_rows}
    if overlap:
        raise ValueError(f"train/dev share videos: {sorted(overlap)[:5]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"train": {"rows": len(tr_rows), "triplets": len(tr_trip),
                               "questions": len({r.qa_id for r in tr_rows})},
                     "dev": {"rows": len(dv_rows), "triplets": len(dv_trip),
                             "questions": len({r.qa_id for r in dv_rows})},
                     "price_per_s": price_per_s, "variants": {}}
    # Reference points that need no training.
    Xd, yd = arrays(dv_rows)
    results["reference"] = {
        "always_stop": simulate(np.full(len(dv_rows), -1.0), dv_rows, 0.0, price_per_s),
        "oracle_gain": simulate(yd, dv_rows, price_per_s, price_per_s),
    }
    for v in variants:
        per_seed = []
        for s in seeds:
            head, curve = fit_variant(v, tr_rows, tr_trip, dataclasses.replace(cfg, seed=s))
            lam, pareto = calibrate(head, dv_rows, price_per_s)
            pred = head.predict(Xd)
            path = out / f"head_{v}_seed{s}.npz"
            head.save(path, lambda_per_s=lam, stop_threshold=0.0, seed=s)
            per_seed.append({"seed": s, "lambda": lam, "final_train_loss": curve[-1],
                             "dev": evaluate_head(head, dv_rows, dv_trip),
                             "dev_at_lambda": simulate(pred, dv_rows, lam, price_per_s),
                             "pareto": pareto, "checkpoint": str(path)})
        results["variants"][v] = per_seed
    (out / "train_assay.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
