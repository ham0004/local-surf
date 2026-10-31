"""Confirmation run on fresh LongVideoBench questions: H1-H4 with Holm correction, and the secondary comparisons
(docs/v3/confirmation_protocol.md). Reads runs/v3_baseline/rows.jsonl; no model calls.

Primary analysis: the fixed set of 894 eligible questions; an answer that failed (GPU out of memory) counts as
wrong. A comparison is final only when both systems answered every eligible question, otherwise it is PENDING.
Sensitivity analysis: the questions both systems answered without a failure.

    python scripts/v3_confirmation_report.py      # reports/v3_confirmation/summary.{json,md}
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

RUN, OUT = Path("runs/v3_baseline"), Path("reports/v3_confirmation")
DS, N_ELIGIBLE, B, SEED = "lvb_fresh", 894, 10_000, 20261024
SYSTEMS = {  # id: (pool, selector, words)
    "S0": ("v2", "clip", 120),
    "S1": ("hybrid", "mmropt", 0),
    "S1t": ("hybrid", "mmropt|tp=frames+hybrid", 600),
    "S2": ("hybrid_ahead", "mmropt", 0),
    "S3": ("hybrid", f"file:{DS}_hybrid_ev_set", 0),
    "S4": ("hybrid_ahead", f"file:{DS}_hybrid_ahead_ev_set", 0),
    "S4t": ("hybrid_ahead", f"file:{DS}_hybrid_ahead_ev_set|tp=frames+hybrid", 600),
    "R2": ("uniform4", "clip", 0),
    "R3": ("dense4", "clip", 0),
}
MODELS = {"2B": "", "4B": "|ans=4b", "Nemotron-120B": "|ans=nemo"}
PRIMARY = [("H1", "S4", "S1", "4B"), ("H2", "S4", "S1", "Nemotron-120B"),
           ("H3", "S1t", "S0", "4B"), ("H4", "S1t", "S0", "Nemotron-120B")]
SECONDARY = [("S2", "S1"), ("S3", "S1"), ("S4t", "S1t"), ("S1", "R3"), ("S1", "R2")]


def load() -> dict:
    last = {}
    for line in (RUN / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["dataset"] == DS and r["k"] == 4:
            last[(r["pool"], r["selector"], r["words"], r["qa_id"])] = r
    out = collections.defaultdict(dict)
    for (p, s, w, q), r in last.items():
        out[(p, s, w)][q] = r
    return out


def system(rows: dict, sid: str, model: str) -> dict:
    p, s, w = SYSTEMS[sid]
    return rows.get((p, s + MODELS[model], w), {})


def paired(a: dict, b: dict, qs: list) -> dict:
    """Difference in points, 95% interval and two-sided p from a paired bootstrap over videos."""
    byv = collections.defaultdict(list)
    for q in qs:
        byv[a[q]["video_id"]].append(float(a[q]["correct"]) - float(b[q]["correct"]))
    vids = sorted(byv)
    rng = np.random.default_rng(SEED)
    boots = np.array([np.mean(np.concatenate([byv[vids[i]] for i in rng.choice(len(vids), len(vids))]))
                      for _ in range(B)])
    d = float(np.mean([float(a[q]["correct"]) - float(b[q]["correct"]) for q in qs]))
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {"diff": round(100 * d, 1), "lo": round(100 * np.percentile(boots, 2.5), 1),
            "hi": round(100 * np.percentile(boots, 97.5), 1), "p": round(float(p), 4), "n": len(qs), "videos": len(vids)}


def compare(a: dict, b: dict) -> dict:
    complete = len(a) == N_ELIGIBLE and len(b) == N_ELIGIBLE
    both = sorted(set(a) & set(b))
    res = {"status": "final" if complete else "PENDING", "answered": [len(a), len(b)],
           "failed": [sum(bool(a[q].get("failed")) for q in a), sum(bool(b[q].get("failed")) for q in b)]}
    if len(both) >= 10:
        res["primary"] = paired(a, b, both)
        ok = [q for q in both if not a[q].get("failed") and not b[q].get("failed")]
        if len(ok) < len(both):
            res["common_success"] = paired(a, b, ok)
    return res


def holm(ps: dict) -> dict:
    """Holm-adjusted p-values over the declared family (only meaningful when every member is final)."""
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[h]))
        adj[h] = round(run, 4)
    return adj


def main() -> None:
    rows = load()
    summary = {"eligible": N_ELIGIBLE, "accuracy": {}, "primary": {}, "secondary": {}}
    for m in MODELS:
        for sid in SYSTEMS:
            x = system(rows, sid, m)
            if x:
                summary["accuracy"][f"{sid} {m}"] = {"answered": len(x), "correct": sum(r["correct"] for r in x.values()),
                                                     "failed": sum(bool(r.get("failed")) for r in x.values())}
    for h, a, b, m in PRIMARY:
        summary["primary"][h] = {"contrast": f"{a} - {b}", "model": m, **compare(system(rows, a, m), system(rows, b, m))}
    final = {h: v["primary"]["p"] for h, v in summary["primary"].items() if v["status"] == "final"}
    adj = holm({h: v["primary"]["p"] for h, v in summary["primary"].items() if "primary" in v})
    for h, v in summary["primary"].items():
        v["p_holm"] = adj.get(h)
        v["holm_final"] = len(final) == len(PRIMARY)
    for m in MODELS:
        for a, b in SECONDARY:
            x, y = system(rows, a, m), system(rows, b, m)
            if x and y:
                summary["secondary"][f"{a} - {b} {m}"] = compare(x, y)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def line(name, v):
        if "primary" not in v:
            return f"| {name} | {v['status']} | answered {v['answered']} | | | |"
        q = v["primary"]
        cs = v.get("common_success")
        extra = f"; without failures {cs['diff']:+.1f} ({cs['lo']:+.1f}..{cs['hi']:+.1f}), n {cs['n']}" if cs else ""
        return (f"| {name} | {v['status']} | {q['diff']:+.1f} ({q['lo']:+.1f}..{q['hi']:+.1f}) | {q['p'] or '<0.0001'} | "
                f"{'<0.0001' if v.get('p_holm') == 0 else v.get('p_holm', '')} | n {q['n']}, videos {q['videos']}, failed {v['failed']}{extra} |")

    md = ["# Confirmation on fresh LongVideoBench questions", "",
          f"Primary analysis on the {N_ELIGIBLE} eligible questions; failed answers count as wrong. Holm-adjusted "
          "p-values are final only when all four hypotheses are final.", "",
          "| Hypothesis | Status | Difference (95% interval) | p | p (Holm) | Questions |", "|---|---|---|---|---|---|"]
    md += [line(f"{h}: {v['contrast']}, {v['model']}", v) for h, v in summary["primary"].items()]
    md += ["", "## Secondary (no correction)", "", "| Comparison | Status | Difference (95% interval) | p | | Questions |",
           "|---|---|---|---|---|---|"]
    md += [line(k, {**v, "p_holm": ""}) for k, v in summary["secondary"].items()]
    md += ["", "## Accuracy", "", "| System | Answered | Correct | Accuracy | Failed |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['answered']} | {v['correct']} | {100 * v['correct'] / v['answered']:.1f} | {v['failed']} |"
           for k, v in summary["accuracy"].items()]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
