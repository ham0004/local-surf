"""Train the controller's utility head and tune its STOP threshold.

Two variants are trained on IDENTICAL rows with the same optimiser and seed:

    paired     lambda_pair > 0   (the proposed transcript-intervention loss)
    unpaired   lambda_pair = 0   (ablation: same labels, no pairing)

so any difference between them is attributable to the paired term.

STOP-threshold tuning happens on a separate DEV label set (different videos),
never on test.  For every labelled decision point (question, condition, step)
we simulate the controller's choice at a candidate threshold and score it by
the gain that choice ACTUALLY produced in the labels, minus the per-frame cost
penalty.  The threshold with the best mean realised net gain wins.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .heads import TrainConfig, UtilityHead
from .labels import LabelRow, read_rows


def rows_to_arrays(rows: list[LabelRow]) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([r.features for r in rows], dtype=np.float64)
    y = np.asarray([r.gain for r in rows], dtype=np.float64)
    return X, y


def decision_groups(rows: list[LabelRow]) -> list[list[int]]:
    """Row indices grouped by decision point (same question, condition, step)."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.qa_id, r.condition, r.step)].append(i)
    return list(groups.values())


def realised_net_gain(head: UtilityHead, rows: list[LabelRow], threshold: float, look_cost: float) -> float:
    """Mean realised (gain - cost penalty) of the head's decisions at ``threshold``.
    STOP realises 0."""
    if not rows:
        return 0.0
    X, _ = rows_to_arrays(rows)
    pred = head.predict(X)
    totals = []
    for idx in decision_groups(rows):
        penalties = np.array([look_cost if rows[i].candidate_id else 0.0 for i in idx])
        net_pred = pred[idx] - penalties
        best = int(np.argmax(net_pred))
        if net_pred[best] < threshold:
            totals.append(0.0)                          # STOP
        else:
            i = idx[best]
            totals.append(rows[i].gain - penalties[best])
    return float(np.mean(totals))


def tune_threshold(head: UtilityHead, dev_rows: list[LabelRow], look_cost: float,
                   grid=np.linspace(-0.2, 1.0, 61)) -> tuple[float, float]:
    """Grid search the STOP threshold on dev rows; ties go to the HIGHER
    threshold (cheaper policy)."""
    best_t, best_v = float(grid[0]), -np.inf
    for t in grid:
        v = realised_net_gain(head, dev_rows, float(t), look_cost)
        if v >= best_v - 1e-12:
            best_t, best_v = float(t), v
    return best_t, float(best_v)


def pair_difference_mae(head: UtilityHead, rows: list[LabelRow], pairs: list[tuple[int, int]]) -> float | None:
    """How well the head predicts (gain_targeted - gain_control) on paired rows."""
    if not pairs:
        return None
    X, y = rows_to_arrays(rows)
    pred = head.predict(X)
    t, c = np.array(pairs).T
    return float(np.mean(np.abs((pred[t] - pred[c]) - (y[t] - y[c]))))


def train_variant(train_dir: str | Path, dev_dir: str | Path | None, out_path: str | Path,
                  cfg: TrainConfig, look_cost: float = 0.02) -> dict:
    """Train one head, tune its threshold on dev, save, and return metrics."""
    rows, pairs = read_rows(train_dir)
    X, y = rows_to_arrays(rows)
    head = UtilityHead(X.shape[1], cfg.hidden, cfg.seed)
    curve = head.fit(X, y, np.asarray(pairs, dtype=int).reshape(-1, 2), cfg)

    dev_rows, dev_pairs = read_rows(dev_dir) if dev_dir else ([], [])
    threshold, dev_value = tune_threshold(head, dev_rows, look_cost) if dev_rows else (0.05, None)
    head.save(out_path, stop_threshold=threshold, look_cost=look_cost, lambda_pair=cfg.lambda_pair)

    metrics = {
        "checkpoint": str(out_path),
        "train_config": dataclasses.asdict(cfg),
        "n_train_rows": len(rows), "n_train_pairs": len(pairs),
        "n_dev_rows": len(dev_rows), "n_dev_pairs": len(dev_pairs),
        "final_train_loss": curve[-1],
        "train_pair_diff_mae": pair_difference_mae(head, rows, pairs),
        "dev_pair_diff_mae": pair_difference_mae(head, dev_rows, dev_pairs) if dev_rows else None,
        "stop_threshold": threshold,
        "dev_realised_net_gain": dev_value,
    }
    Path(out_path).with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_paired_and_unpaired(train_dir, dev_dir, out_dir, base: TrainConfig | None = None,
                              lambda_pair: float = 1.0, look_cost: float = 0.02) -> dict[str, dict]:
    """Train the paired model and its no-pairing ablation with identical settings."""
    base = base or TrainConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "paired": train_variant(train_dir, dev_dir, out / "head_paired.npz",
                                dataclasses.replace(base, lambda_pair=lambda_pair), look_cost),
        "unpaired": train_variant(train_dir, dev_dir, out / "head_unpaired.npz",
                                  dataclasses.replace(base, lambda_pair=0.0), look_cost),
    }
