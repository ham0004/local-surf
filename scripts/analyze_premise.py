"""Premise check: does losing answer-relevant speech make LOOKING more valuable?

The paired-training hypothesis only has something to learn if, for the same
question and the same candidate frame, the measured gain from looking is
larger under TARGETED damage than under the matched CONTROL damage. This
reads the label rows written by `videoqa build-labels` (real frozen-answerer
before/after measurements) and reports:

  * per condition: base quality before any action, mean action gain,
    P(gain > 0), P(gain < 0), mean best-action gain (step 0)
  * over matched pairs: gain_targeted - gain_control, with a 95% interval
    from resampling whole questions
  * optionally, the same paired difference per question category

Usage:
  uv run python scripts/analyze_premise.py --labels runs/lvb_full/labels_train \
      --categories cache/lvb_meta/lvb_val.json --out reports/lvb_full/premise.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.labels import read_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--categories", help="LongVideoBench lvb_val.json, for a per-category breakdown")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    rows, pairs = read_rows(a.labels)
    report: dict = {"labels": a.labels, "n_rows": len(rows), "n_pairs": len(pairs), "conditions": {}}

    step0 = defaultdict(list)
    for r in rows:
        if r.step == 0:
            step0[r.condition].append(r)
    for cond, rs in step0.items():
        base = {r.qa_id: r.quality_before for r in rs}
        best: dict[str, float] = defaultdict(float)
        for r in rs:
            best[r.qa_id] = max(best[r.qa_id], r.gain)
        g = np.array([r.gain for r in rs])
        report["conditions"][cond] = {
            "n_questions": len(base), "base_quality": float(np.mean(list(base.values()))),
            "mean_action_gain": float(g.mean()), "p_gain_pos": float(np.mean(g > 0)),
            "p_gain_neg": float(np.mean(g < 0)), "mean_best_action_gain": float(np.mean(list(best.values()))),
        }

    t, c = np.array(pairs, dtype=int).T
    diff = np.array([rows[i].gain for i in t]) - np.array([rows[j].gain for j in c])
    qids = np.array([rows[i].qa_id for i in t])
    by_q = {q: diff[qids == q] for q in np.unique(qids)}
    rng = np.random.default_rng(0)
    keys = list(by_q)
    boots = [np.mean(np.concatenate([by_q[keys[k]] for k in rng.choice(len(keys), len(keys))]))
             for _ in range(a.n_boot)]
    report["paired_gain_targeted_minus_control"] = {
        "mean": float(diff.mean()), "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)), "p_pos": float(np.mean(diff > 0)),
        "p_neg": float(np.mean(diff < 0)), "p_zero": float(np.mean(diff == 0)), "n_questions": len(keys),
    }

    if a.categories:
        cat = {d["id"]: d["question_category"] for d in json.loads(Path(a.categories).read_text(encoding="utf-8"))}
        per = defaultdict(list)
        for q, dq in by_q.items():
            per[cat.get(q, "?")].extend(dq.tolist())
        report["per_category"] = {k: {"n_pairs": len(v), "mean": float(np.mean(v))} for k, v in sorted(per.items())}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
