"""Analyse gates G2 (judge vs human labels) and G1 (frames vs transcript only).

Inputs: runs/v3_gate/verdicts_<judge>.jsonl, runs/v3_gate/audit_key.json and the human
labels exported from the audit page to runs/v3_gate/human_labels.json
({"i07": {"consistency": 2, "coverage": 50, ...}, ...}). Output: reports/v3_gate/summary.json.

G2 reports, per judge on the audit items: Spearman correlation of quality with the human
quality, exact agreement on consistency, and the false-accept rate (judge says consistent
where the human says wrong). G1 reports the paired TF - T difference in judge quality with a
video-clustered bootstrap, using the judge chosen by G2.

    python scripts/v3_gate_analyze.py --judge phi4mini
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

RUN = Path("runs/v3_gate")


def _gate():
    spec = importlib.util.spec_from_file_location("gate", Path(__file__).with_name("v3_gate_frames.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):                       # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b) -> float:
    a, b = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def video_bootstrap(diffs: dict, videos: dict, repeats: int = 5000) -> dict:
    by: dict[str, list[float]] = {}
    for q, d in diffs.items():
        by.setdefault(videos[q], []).append(d)
    s = np.array([sum(v) for v in by.values()])
    c = np.array([len(v) for v in by.values()])
    ids = np.random.default_rng(0).integers(0, len(s), (repeats, len(s)))
    b = s[ids].sum(1) / c[ids].sum(1)
    return {"difference": float(s.sum() / c.sum()), "ci95_video_bootstrap": np.percentile(b, [2.5, 97.5]).tolist(),
            "questions": int(c.sum()), "videos": len(s)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge", default=None, help="judge for G1 (default: best G2 agreement)")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    from videoqa.v3.judge import quality, verdict_key  # noqa: PLC0415

    g = _gate()
    rows = g._answer_rows(a)
    verdicts = {}
    for name in ("phi4mini", "qwen3vl4b"):
        path = RUN / f"verdicts_{name}.jsonl"
        if path.exists():
            verdicts[name] = {json.loads(s)["key"]: json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()
                              if s.strip()}
    out = {"answers": len(rows), "questions": len({r["qa_id"] for r in rows})}

    # --- G2: agreement with human labels on the audit items
    key = json.loads((RUN / "audit_key.json").read_text(encoding="utf-8")) if (RUN / "audit_key.json").exists() else []
    human_path = RUN / "human_labels.json"
    if key and human_path.exists():
        human = json.loads(human_path.read_text(encoding="utf-8"))
        by_qc = {(r["qa_id"], r["condition"]): r for r in rows}
        g2 = {"labelled": 0}
        for name, vs in verdicts.items():
            hq, jq, hc, jc = [], [], [], []
            for k in key:
                h = human.get(f"i{k['item']:02d}")
                r = by_qc.get((k["qa_id"], k["condition"]))
                v = vs.get(verdict_key(name, r["question"], r["reference"], r["answer"])) if r else None
                if not h or v is None or v.get("quality") is None or h.get("coverage") is None:
                    continue
                hq.append(quality(h["consistency"], h["coverage"]))
                jq.append(v["quality"])
                hc.append(h["consistency"])
                jc.append(v["consistency"])
            hc, jc = np.array(hc), np.array(jc)
            g2["labelled"] = max(g2["labelled"], len(hq))
            g2[name] = {"n": len(hq), "spearman_quality": spearman(hq, jq) if len(hq) > 2 else None,
                        "consistency_exact_agreement": float(np.mean(hc == jc)) if len(hc) else None,
                        "false_accept_rate": float(np.mean(jc[hc == 0] == 2)) if (hc == 0).any() else None,
                        "human_wrong_items": int((hc == 0).sum()),
                        "malformed_verdicts": sum(1 for v in vs.values() if v.get("malformed"))}
        out["G2"] = g2
        if a.judge is None:
            scored = [(n, d["spearman_quality"]) for n, d in g2.items() if isinstance(d, dict) and d.get("spearman_quality") is not None]
            a.judge = max(scored, key=lambda x: x[1])[0] if scored else None

    # --- G1: paired TF - T with the chosen judge
    if a.judge and a.judge in verdicts:
        vs = verdicts[a.judge]
        q = {}
        for r in rows:
            v = vs.get(verdict_key(a.judge, r["question"], r["reference"], r["answer"]))
            if v and v.get("quality") is not None:
                q.setdefault(r["qa_id"], {})[r["condition"]] = v
        both = {k: d for k, d in q.items() if "T" in d and "TF" in d}
        videos = {r["qa_id"]: r["video_id"] for r in rows}
        out["G1"] = {"judge": a.judge, "paired_questions": len(both),
                     "mean_quality": {c: float(np.mean([d[c]["quality"] for d in both.values()])) for c in ("T", "TF")},
                     "mean_coverage": {c: float(np.mean([d[c]["coverage"] for d in both.values()])) for c in ("T", "TF")},
                     "share_consistent": {c: float(np.mean([d[c]["consistency"] == 2 for d in both.values()])) for c in ("T", "TF")},
                     "TF_minus_T_quality": video_bootstrap({k: d["TF"]["quality"] - d["T"]["quality"] for k, d in both.items()}, videos),
                     "TF_better_same_worse": [sum(d["TF"]["quality"] > d["T"]["quality"] for d in both.values()),
                                              sum(d["TF"]["quality"] == d["T"]["quality"] for d in both.values()),
                                              sum(d["TF"]["quality"] < d["T"]["quality"] for d in both.values())]}
    Path("reports/v3_gate").mkdir(parents=True, exist_ok=True)
    Path("reports/v3_gate/summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
