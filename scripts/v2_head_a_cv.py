"""Leave-one-lecture-out check of Head A's two outputs (no teacher calls).

Uses the separately intervened labels in runs/v2_pilot/head_a_labels.jsonl:
does the trained two-output head rank segments by text gain / frame gain better
than the zero-shot relevance score it starts from? Metric: ROC-AUC of
"gain > 0" within held-out lectures (0.5 = chance).

    python scripts/v2_head_a_cv.py --run runs/v2_pilot --data data/mit_lectures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.datasets import load_local_dataset  # noqa: E402
from videoqa.v2.candidates import _with_neighbours  # noqa: E402
from videoqa.v2.head_a import HotMomentScorer  # noqa: E402


def auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2_pilot")
    ap.add_argument("--data", default="data/mit_lectures")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (Path(a.run) / "head_a_labels.jsonl").read_text(encoding="utf-8").splitlines() if x]
    items, _ = load_local_dataset(a.data, splits=("train", "dev", "calibration", "test"))
    by_q = {it.qa.qa_id: it for it in items}
    scorer = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub")
    feats = []
    for r in rows:
        it = by_q[r["qa_id"]]
        seg = next(s for s in it.transcript.segments if s.id == r["segment_id"])
        feats.append(scorer.features(it.qa.question, [_with_neighbours(it.transcript.segments, seg)])[0])
    X = np.stack(feats)
    Y = np.array([[r["text_gain"], r["visual_gain"]] for r in rows], dtype=np.float32)
    vids = np.array([r["video_id"] for r in rows])
    groups = np.array([r["qa_id"] for r in rows])
    pred_trained = np.zeros_like(Y)
    for v in np.unique(vids):
        tr, te = vids != v, vids == v
        head = HotMomentScorer()
        head.fit(X[tr], Y[tr], groups[tr])
        pred_trained[te] = ((X[te] - head.feat_mean) / head.feat_std) @ head.head_w + head.head_b
    zero_shot = X[:, 0]
    out = {"rows": len(rows), "questions": len(set(groups)), "lectures": int(len(np.unique(vids)))}
    for k, name in ((0, "text"), (1, "visual")):
        positive = Y[:, k] > 0
        out[name] = {"positives": int(positive.sum()),
                     "auc_zero_shot_relevance": auc(zero_shot, positive),
                     "auc_trained_head_a": auc(pred_trained[:, k], positive)}
    out["note"] = "pooled over held-out lectures; small positive counts, so AUCs are noisy"
    (Path(a.run) / "head_a_cv.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
