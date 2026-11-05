"""Blind human review of the 100 lecture test packets: the analyses declared in docs/v3/m3av_protocol.md
("Question review", amendment 3) and research log steps 95 and 98. Written and checked before any human verdict
was seen; reads saved files only, runs no model.

Inputs: the human reviewers' answers, one or more files, each either
  - the review page's shared store as saved by the owner (JSON; every document holding an `answers` map), or
  - the page's "Copy my answers" text (CSV: a `reviewer,<name>` line, then item, qa_id, verdict, key, one, slide,
    legible, shortcut, reason, note), or a filled form.csv (same fields, the packet form's column names).
  An answer counts only if it is complete as the page defines it: the four checks, the shortcut, a verdict, a reason
  code, and a note for reject or uncertain. Reviewers are reported as H1, H2, ... (names stay out of git).

Combination rule (fixed before any human verdict was seen): one reviewer on an item, that verdict; several, the
majority verdict; a tie, uncertain. Items no human completed are reported as unreviewed and left out of every rate.

Analyses:
  1. Rejection rate of the reviewed sample: lecture-balanced (mean over sampled lectures; one question each) and the
     cohort's question-weighted rate sum(n_l r_l) / sum(n_l), n_l = the lecture's cohort questions; 95% lecture-
     bootstrap intervals (10,000 draws, seed 20261101). Also the reject-or-uncertain rate.
  2. The four checks, the shortcut and the reason codes; agreement of the combined human verdict with each model
     review (builder, auditor): percent agreement, Cohen's kappa, the 3 x 3 table; and between humans where they
     overlap.
  3. Secondary 1 (sensitivity, never promoted over the primary): every system's hit@4 and S-src hit@4 on the cohort
     without the human-rejected sampled questions (one mask for all systems), L1 and L2 on it with paired lecture-
     bootstrap intervals, no p-values or Holm; the same with rejected-or-uncertain removed; hit@4 on the accepted
     sample questions alone; the answer model's accuracy (descriptive) under the same masks.

    python scripts/m3av_human_review.py --human runs/m3av/review_human/people.json [more files]
        # reports/m3av/human_review_test.{json,md}
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_hits import PRIMARY, SYSTEMS, paired  # noqa: E402

B, SEED = 10_000, 20261101
VERDICTS = ("accept", "reject", "uncertain")
CHECKS = ("key", "one", "slide", "legible")
FORM_COLS = {"verdict": "verdict", "key correct": "key", "one best answer": "one", "supported by the slide": "slide",
             "legible at 360p": "legible", "shortcut": "shortcut", "reason code": "reason", "note": "note",
             "qa_id": "qa_id", "n": "item", "item": "item"}
MODEL_REVIEWS = {"builder": Path("collab/review_builder_test.csv"), "auditor": Path("collab/review_auditor_test.csv")}
CODES = {name: name.split()[0] for name in SYSTEMS}


def complete(a: dict) -> bool:
    return (all(a.get(c) in ("y", "n") for c in CHECKS) and a.get("shortcut") in ("none", "knowledge", "transcript")
            and a.get("verdict") in VERDICTS and bool(a.get("reason"))
            and (a["verdict"] == "accept" or bool((a.get("note") or "").strip())))


def _norm(row: dict) -> dict:
    out = {}
    page = set(FORM_COLS.values())
    for k, v in row.items():
        k = (k or "").strip().lower()
        key = k if k in page else next((FORM_COLS[c] for c in FORM_COLS if k.startswith(c)), None)
        if key:
            out[key] = str("" if v is None else v).strip()
    for c in (*CHECKS, "verdict", "shortcut"):
        out[c] = out.get(c, "").lower()
    return out


def read_csv(path: Path) -> list[tuple[str, dict]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    name = path.stem
    if lines and lines[0].lower().startswith("reviewer,"):
        name = next(csv.reader([lines[0]]))[1] or name
        text = "\n".join(lines[1:])
    answers = {}
    for r in csv.DictReader(io.StringIO(text)):
        a = _norm(r)
        if a.get("qa_id"):
            answers[a["qa_id"]] = a
    return [(name, answers)]


def read_store(path: Path) -> list[tuple[str, dict]]:
    """Every object in the saved JSON that holds an `answers` map is one reviewer."""
    found = []

    def walk(x, uid=None):
        if isinstance(x, dict):
            if isinstance(x.get("answers"), dict):
                ans = {v.get("qa_id"): _norm(v) | {"qa_id": v.get("qa_id")} for v in x["answers"].values()
                       if isinstance(v, dict) and v.get("qa_id")}
                found.append((x.get("name") or uid or f"{path.stem}#{len(found) + 1}", ans))
                return
            for k, v in x.items():
                walk(v, x.get("id") or x.get("doc_id") or k)
        elif isinstance(x, list):
            for v in x:
                walk(v, uid)

    walk(json.loads(path.read_text(encoding="utf-8")))
    return found


def kappa(a: list, b: list) -> float | None:
    if not a:
        return None
    po = np.mean([x == y for x, y in zip(a, b, strict=True)])
    pe = sum((a.count(v) / len(a)) * (b.count(v) / len(b)) for v in VERDICTS)
    return None if pe == 1 else round(float((po - pe) / (1 - pe)), 3)


def agreement(a: dict, b: dict) -> dict:
    qs = sorted(set(a) & set(b))
    x, y = [a[q] for q in qs], [b[q] for q in qs]
    return {"n": len(qs), "agree": int(sum(u == v for u, v in zip(x, y, strict=True))),
            "percent": round(100 * float(np.mean([u == v for u, v in zip(x, y, strict=True)])), 1) if qs else None,
            "kappa": kappa(x, y),
            "table": {u: {v: int(sum(1 for p, s in zip(x, y, strict=True) if p == u and s == v)) for v in VERDICTS}
                      for u in VERDICTS}}


def boot_rate(r: dict, w: dict) -> tuple:
    """Weighted mean of r_l with weights w_l and a 95% bootstrap interval over the sampled lectures."""
    ls = sorted(r)
    rv, wv = np.array([r[x] for x in ls], float), np.array([w[x] for x in ls], float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ls), (B, len(ls)))
    boots = (rv[idx] * wv[idx]).sum(1) / wv[idx].sum(1)
    return (round(100 * float((rv * wv).sum() / wv.sum()), 1), round(100 * float(np.percentile(boots, 2.5)), 1),
            round(100 * float(np.percentile(boots, 97.5)), 1))


def rates(flag: dict, lecture: dict, n_l: dict) -> dict:
    r = {lecture[q]: float(v) for q, v in flag.items()}
    return {"lecture_balanced": boot_rate(r, {x: 1.0 for x in r}), "question_weighted": boot_rate(r, {x: n_l[x] for x in r}),
            "count": int(sum(flag.values())), "of": len(flag)}


def main() -> None:  # noqa: C901, PLR0915
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", nargs="+", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/m3av"), help="folder for human_review_test.{json,md}")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    sample =json.loads(Path("reports/m3av/review_sample_test.json").read_text(encoding="utf-8"))
    sample_ids = [x if isinstance(x, str) else x["qa_id"] for x in (sample.get("qa_ids") or sample.get("sample"))]
    hits = json.loads(Path("reports/m3av/hits_test.json").read_text(encoding="utf-8"))
    prim, ssrc = hits["per_question"], hits["s_src"]["per_question"]
    cohort = sorted(next(iter(prim.values())))
    lecture = {q: q.split(":")[1] for q in cohort}
    n_l = collections.Counter(lecture.values())
    if set(sample_ids) - set(cohort) or len(sample_ids) != 100:
        raise SystemExit("the review sample does not match the cohort")

    reviewers = []
    for f in a.human:
        reviewers += read_store(f) if f.suffix.lower() == ".json" else read_csv(f)
    humans = {}
    for i, (name, ans) in enumerate(reviewers, 1):
        ok = {q: v for q, v in ans.items() if q in sample_ids and complete(v)}
        humans[f"H{i}"] = {"answers": ok, "incomplete": sum(1 for q, v in ans.items() if q in sample_ids and not complete(v)),
                           "outside_sample": sum(1 for q in ans if q not in sample_ids), "_name": name}
    combined, votes = {}, {}
    for q in sample_ids:
        vs = [h["answers"][q]["verdict"] for h in humans.values() if q in h["answers"]]
        if not vs:
            continue
        c = collections.Counter(vs).most_common()
        combined[q] = c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "uncertain"
        votes[q] = vs
    reviewed = [q for q in sample_ids if q in combined]
    rep = {"what": "blind human review of the 100 test packets (protocol 'Question review'; research log step 98)",
           "reviewers": {k: {"complete": len(h["answers"]), "incomplete": h["incomplete"],
                             "outside_sample": h["outside_sample"]} for k, h in humans.items()},
           "items_reviewed": len(reviewed), "items_unreviewed": 100 - len(reviewed),
           "items_with_several_reviewers": sum(len(v) > 1 for v in votes.values()),
           "combined_counts": {v: sum(combined[q] == v for q in reviewed) for v in VERDICTS}}
    if not reviewed:
        raise SystemExit("no complete human answer for a sampled question")
    rep["rejection_rate"] = rates({q: combined[q] == "reject" for q in reviewed}, lecture, n_l)
    rep["reject_or_uncertain_rate"] = rates({q: combined[q] != "accept" for q in reviewed}, lecture, n_l)

    def per_item(field):
        out = collections.Counter()
        for h in humans.values():
            for v in h["answers"].values():
                out[v[field]] += 1
        return dict(out)

    rep["checks"] = {c: per_item(c) for c in (*CHECKS, "shortcut", "reason")}
    model = {}
    for k, f in MODEL_REVIEWS.items():
        model[k] = {r["qa_id"]: r["verdict"] for r in read_csv(f)[0][1].values() if r.get("verdict") in VERDICTS}
    rep["agreement"] = {f"human vs {k}": agreement(combined, m) for k, m in model.items()}
    rep["agreement"]["builder vs auditor"] = agreement(model["builder"], model["auditor"])
    hk = list(humans)
    for i, x in enumerate(hk):
        for y in hk[i + 1:]:
            g = agreement({q: v["verdict"] for q, v in humans[x]["answers"].items()},
                          {q: v["verdict"] for q, v in humans[y]["answers"].items()})
            if g["n"]:
                rep["agreement"][f"{x} vs {y}"] = g
    rep["model_reviews_on_reviewed_items"] = {k: {v: sum(m.get(q) == v for q in reviewed) for v in VERDICTS}
                                             for k, m in model.items()}

    masks = {"full cohort": set(),
             "without human-rejected": {q for q in reviewed if combined[q] == "reject"},
             "without human-rejected or uncertain": {q for q in reviewed if combined[q] != "accept"}}
    rep["masks"] = {k: len(v) for k, v in masks.items()}
    rec = [json.loads(x) for x in Path("runs/m3av/qa/test/records.jsonl").read_text(encoding="utf-8").splitlines()
           if x.strip()]
    qa = collections.defaultdict(dict)
    for r in rec:
        qa[r["condition"]][r["qa_id"]] = int(r["correct"])
    rep["secondary_1"] = {}
    for mk, drop in masks.items():
        keep = [q for q in cohort if q not in drop]
        sub = {"questions": len(keep), "lectures": len({lecture[q] for q in keep}),
               "hit@4": {CODES[s]: round(100 * float(np.mean([prim[s][q] for q in keep])), 1) for s in SYSTEMS},
               "s_src_hit@4": {CODES[s]: round(100 * float(np.mean([ssrc[s][q] for q in keep])), 1) for s in SYSTEMS},
               "qa_accuracy": {c: round(100 * float(np.mean([v[q] for q in keep])), 1) for c, v in qa.items()},
               "contrasts": {}}
        for name, (x, y) in PRIMARY.items():
            d = paired({q: prim[x][q] for q in keep}, {q: prim[y][q] for q in keep}, lecture)
            s = paired({q: ssrc[x][q] for q in keep}, {q: ssrc[y][q] for q in keep}, lecture)
            sub["contrasts"][name] = {"hit@4": {k: d[k] for k in ("diff", "lo", "hi")},
                                      "s_src": {k: s[k] for k in ("diff", "lo", "hi")}}
        rep["secondary_1"][mk] = sub
    acc = [q for q in reviewed if combined[q] == "accept"]
    rep["accepted_sample_only"] = {"questions": len(acc),
                                   "hit@4": {CODES[s]: round(100 * float(np.mean([prim[s][q] for q in acc])), 1)
                                             if acc else None for s in SYSTEMS}}
    rep["per_item"] = {q: {"combined": combined.get(q), "votes": votes.get(q, []),
                           "builder": model["builder"].get(q), "auditor": model["auditor"].get(q)} for q in sample_ids}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "human_review_test.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")

    def ci(t):
        return f"{t[0]} ({t[1]}..{t[2]})"

    md = ["# Lecture study: blind human review of the 100 test packets", "",
          "Analyses declared in the protocol (Question review, amendment 3) and fixed in research log step 98 before any "
          "human verdict was seen. Labels of the cohort are otherwise automatic. Sensitivity results are descriptive and "
          "never replace the primary result.", "",
          f"Reviewers: {len(humans)} ({', '.join(f'{k} {v['complete']} items' for k, v in rep['reviewers'].items())}). "
          f"Items reviewed: {len(reviewed)} of 100; with several reviewers: {rep['items_with_several_reviewers']}.", "",
          "| Combined human verdict | accept | reject | uncertain |", "|---|---|---|---|",
          "| items | " + " | ".join(str(rep["combined_counts"][v]) for v in VERDICTS) + " |", "",
          "| Rate, percent (95% lecture-bootstrap interval) | lecture-balanced | cohort question-weighted |",
          "|---|---|---|",
          f"| rejected | {ci(rep['rejection_rate']['lecture_balanced'])} | {ci(rep['rejection_rate']['question_weighted'])} |",
          f"| rejected or uncertain | {ci(rep['reject_or_uncertain_rate']['lecture_balanced'])} | "
          f"{ci(rep['reject_or_uncertain_rate']['question_weighted'])} |", "",
          "## Agreement", "", "| Pair | items | agree | percent | Cohen's kappa |", "|---|---|---|---|---|"]
    for k, g in rep["agreement"].items():
        md.append(f"| {k} | {g['n']} | {g['agree']} | {g['percent']} | {g['kappa']} |")
    md += ["", "## Checks (all complete human answers)", ""]
    for k, v in rep["checks"].items():
        md.append(f"- {k}: " + ", ".join(f"{x} {n}" for x, n in sorted(v.items())))
    md += ["", "## Secondary 1: the cohort with the human-rejected sample questions removed (descriptive)", "",
           "| Mask | questions | S1 | S2m | R5 | L1 S2m − R5 (95% interval) | L2 S2m − S1 (95% interval) |",
           "|---|---|---|---|---|---|---|"]
    for mk, sub in rep["secondary_1"].items():
        h, c = sub["hit@4"], sub["contrasts"]
        md.append(f"| {mk} | {sub['questions']} | {h['S1']} | {h['S2m']} | {h['R5']} | "
                  f"{c['L1']['hit@4']['diff']:+.1f} ({c['L1']['hit@4']['lo']:+.1f}..{c['L1']['hit@4']['hi']:+.1f}) | "
                  f"{c['L2']['hit@4']['diff']:+.1f} ({c['L2']['hit@4']['lo']:+.1f}..{c['L2']['hit@4']['hi']:+.1f}) |")
    md += ["", "S-src hit@4 under the same masks:", "",
           "| Mask | S1 | S2m | R5 | L1 (95% interval) | L2 (95% interval) |", "|---|---|---|---|---|---|"]
    for mk, sub in rep["secondary_1"].items():
        h, c = sub["s_src_hit@4"], sub["contrasts"]
        md.append(f"| {mk} | {h['S1']} | {h['S2m']} | {h['R5']} | "
                  f"{c['L1']['s_src']['diff']:+.1f} ({c['L1']['s_src']['lo']:+.1f}..{c['L1']['s_src']['hi']:+.1f}) | "
                  f"{c['L2']['s_src']['diff']:+.1f} ({c['L2']['s_src']['lo']:+.1f}..{c['L2']['s_src']['hi']:+.1f}) |")
    md += ["", "Answer-model accuracy under the same masks (descriptive):", "",
           "| Mask | S1 | S2m | R5 | q_only | t_only |", "|---|---|---|---|---|---|"]
    for mk, sub in rep["secondary_1"].items():
        q = sub["qa_accuracy"]
        md.append(f"| {mk} | {q['S1']} | {q['S2m']} | {q['R5']} | {q['q_only']} | {q['t_only']} |")
    s = rep["accepted_sample_only"]
    md += ["", f"Accepted sample questions alone ({s['questions']}): hit@4 "
           + ", ".join(f"{k} {v}" for k, v in s["hit@4"].items()) + "."]
    (a.out / "human_review_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
