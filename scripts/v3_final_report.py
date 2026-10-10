"""Final frozen comparison: tables and paired tests exactly as declared in docs/v3/final_test_protocol.md.

    python scripts/v3_final_report.py        # writes reports/v3_final/test_results.json and .md
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

ROWS = Path("runs/v3_baseline/rows.jsonl")
OUT = Path("reports/v3_final")
SETS = ("cgbench_test", "videommmu_test", "videommmu_comp_test")
NAMES = {"cgbench_test": "CG-Bench", "videommmu_test": "VMMMU Perception", "videommmu_comp_test": "VMMMU Comprehension"}
SYSTEMS = {  # id: (description, pool, selector template, words)
    "R1": ("question only", "hybrid", "none", 0),
    "R2": ("4 evenly spaced frames", "uniform4", "clip", 0),
    "R3": ("dense MobileCLIP top-4", "dense4", "clip", 0),
    "S0": ("v2 default (starting framework)", "v2", "clip", 120),
    "S1": ("frozen baseline (tuned rules)", "hybrid", "mmropt", 0),
    "S2": ("+ Head A", "hybrid_ahead", "mmropt", 0),
    "S3": ("+ Head B (ev_set)", "hybrid", "file:{ds}_hybrid_ev_set", 0),
    "S4": ("Head A + Head B", "hybrid_ahead", "file:{ds}_hybrid_ahead_ev_set", 0),
    "S5": ("answer-confidence selection (4 sets)", "verify", "max", 0),
}
COMPARISONS = [("S1", "S0", "primary"), ("S2", "S1", "secondary"), ("S3", "S1", "secondary"),
               ("S4", "S1", "secondary"), ("S5", "S1", "secondary"), ("S1", "R3", "secondary"),
               ("S1", "R2", "secondary")]
B = 10_000


def load() -> dict:
    last = {}
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[(r["dataset"], r["pool"], r["selector"], r["k"], r["words"], r["qa_id"])] = r
    return last


def system_rows(last: dict, sid: str, ds: str) -> dict:
    _, pool, sel, words = SYSTEMS[sid]
    sel = sel.format(ds=ds)
    return {q: r for (d, p, s, k, w, q), r in last.items() if (d, p, s, k, w) == (ds, pool, sel, 4, words)}


def paired(a: dict, b: dict, rng) -> dict:
    qs = sorted(set(a) & set(b))
    byv = collections.defaultdict(list)
    for q in qs:
        byv[a[q]["_v"]].append(float(a[q]["correct"]) - float(b[q]["correct"]))
    vids = sorted(byv)
    boots = [np.mean(np.concatenate([byv[vids[i]] for i in rng.choice(len(vids), len(vids))])) for _ in range(B)]
    d = np.mean([float(a[q]["correct"]) - float(b[q]["correct"]) for q in qs])
    return {"diff": float(d), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": len(qs)}


def main() -> None:
    last = load()
    rng = np.random.default_rng(20261010)
    res = {"systems": {}, "comparisons": []}
    pooled = {}
    for sid, (desc, *_r) in SYSTEMS.items():
        res["systems"][sid] = {"description": desc}
        allrows = {}
        for ds in SETS:
            rows = system_rows(last, sid, ds)
            for q, r in rows.items():
                allrows[(ds, q)] = {**r, "_v": ds + ":" + r["video_id"]}
            acc = float(np.mean([r["correct"] for r in rows.values()])) if rows else None
            ev = [r["chosen_hits_evidence"] for r in rows.values() if r.get("chosen_hits_evidence") is not None]
            res["systems"][sid][ds] = {"n": len(rows), "accuracy": acc,
                                       "evidence_recall": float(np.mean(ev)) if ev else None}
        pooled[sid] = allrows
        res["systems"][sid]["pooled"] = {"n": len(allrows),
                                         "accuracy": float(np.mean([r["correct"] for r in allrows.values()]))
                                         if allrows else None}
    for a, b, kind in COMPARISONS:
        entry = {"a": a, "b": b, "kind": kind, "pooled": paired(pooled[a], pooled[b], rng)}
        for ds in SETS:
            pa = {q: r for (d, q), r in pooled[a].items() if d == ds}
            pb = {q: r for (d, q), r in pooled[b].items() if d == ds}
            entry[ds] = paired(pa, pb, rng)
        res["comparisons"].append(entry)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
    lines = ["| System | " + " | ".join(NAMES[d] for d in SETS) + " | Pooled | CG-Bench evidence |",
             "|---|" + "---|" * (len(SETS) + 2)]
    for sid, s in res["systems"].items():
        lines.append(f"| {sid} {s['description']} | " + " | ".join(f"{pct(s[d]['accuracy'])} (n={s[d]['n']})" for d in SETS)
                     + f" | {pct(s['pooled']['accuracy'])} | {pct(s['cgbench_test']['evidence_recall'])} |")
    lines += ["", "| Comparison | kind | " + " | ".join(NAMES[d] for d in SETS) + " | Pooled |",
              "|---|---|" + "---|" * (len(SETS) + 1)]
    for c in res["comparisons"]:
        cell = lambda e: f"{100 * e['diff']:+.1f} ({100 * e['ci'][0]:+.1f}..{100 * e['ci'][1]:+.1f})"  # noqa: E731
        lines.append(f"| {c['a']} − {c['b']} | {c['kind']} | " + " | ".join(cell(c[d]) for d in SETS)
                     + f" | {cell(c['pooled'])} |")
    (OUT / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
