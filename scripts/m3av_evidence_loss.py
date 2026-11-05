"""Lecture evidence-loss breakdown (auditor P125; research log step 96). Reads saved files only; no model runs.

For every frozen system and question, where is the source slide lost? The test is the primary measure's
(a frame within 1 s of a visible span of a valid evidence segment), applied at two stages:
  - candidate stage: no candidate in the system's saved pool is inside a span ("not in pool");
  - selection stage: some candidate is, but none of the 4 chosen frames ("in pool, not picked");
  - picked: one of the 4 chosen frames is inside a span (= source-slide interval hit@4).
The chosen-frame hits are taken from reports/m3av/hits_{split}.json and the pool check must agree with them
(a picked hit is always in the pool). Labels are automatic (a visibility proxy with known false matches).

dev (514 questions): the decision basis for any later redesign.
test (853 questions): the same breakdown plus the link to the answer model's records. Post hoc and descriptive
only: the test is revealed, nothing here chooses a method, and no redesign is evaluated on it.

    python scripts/m3av_evidence_loss.py --split dev     # reports/m3av/evidence_loss_dev.{json,md}
    python scripts/m3av_evidence_loss.py --split test    # reports/m3av/evidence_loss_test.{json,md}
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_hits import SYSTEMS  # noqa: E402

B, SEED = 10_000, 20261101
RUN = Path("runs/v3_baseline")
CODES = {"S0 starting design": "S0", "S1 best framework (also S1t's frames)": "S1", "S2c Head A (CG-Bench)": "S2c",
         "S2m Head A (M3AV train)": "S2m", "S3 Head B": "S3", "R2 4 evenly spaced": "R2",
         "R3 dense MobileCLIP top-4 (96 ranked)": "R3", "R4 dense option-aware top-4 (96 ranked)": "R4",
         "R5 VSI-inspired (96 ranked)": "R5"}


def _safe(s: str) -> str:
    return s.replace(":", "_").replace("/", "_")


def inside(t: float, spans: list) -> bool:
    return any(a - 1.0 <= t <= b + 1.0 for a, b in spans)


def boot_ci(vals: dict, lecture: dict) -> tuple:
    """Mean over questions with a 95% lecture-bootstrap interval (same resampling as m3av_hits.paired)."""
    byl = collections.defaultdict(list)
    for q, v in vals.items():
        byl[lecture[q]].append(float(v))
    ls = sorted(byl)
    rng = np.random.default_rng(SEED)
    boots = np.array([np.mean(np.concatenate([byl[ls[i]] for i in rng.choice(len(ls), len(ls))])) for _ in range(B)])
    return (round(100 * float(np.mean(list(vals.values()))), 1), round(100 * float(np.percentile(boots, 2.5)), 1),
            round(100 * float(np.percentile(boots, 97.5)), 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    a = p.parse_args()
    ds = f"m3av_{a.split}"
    qs = [json.loads(x) for x in Path(f"data/m3av/questions_{a.split}_visible.jsonl").read_text(encoding="utf-8").splitlines()]
    spans = {q["qa_id"]: q["visible_spans"] for q in qs}
    lecture = {q["qa_id"]: q["qa_id"].split(":")[1] for q in qs}
    hits = json.loads(Path(f"reports/m3av/hits_{a.split}.json").read_text(encoding="utf-8"))
    if set(hits["per_question"][next(iter(SYSTEMS))]) != set(spans):
        raise SystemExit("cohort file and hits report hold different questions")

    qa = None
    if a.split == "test":
        recs = [json.loads(x) for x in Path("runs/m3av/qa/test/records.jsonl").read_text(encoding="utf-8").splitlines()]
        gold = {}
        for line in Path("data/m3av_test/qa.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            gold[r["qa_id"]] = r["gold_option_index"]
        qa = collections.defaultdict(dict)
        for r in recs:
            qa[r["condition"]][r["qa_id"]] = (int(r["option"] == gold[r["qa_id"]]), r["primary_hit"])
        if any(len(v) != len(spans) for v in qa.values()):
            raise SystemExit("answer records do not cover the cohort for every condition")

    out = {"split": a.split, "questions": len(spans), "lectures": len(set(lecture.values())),
           "labels": "automatic source-slide intervals (visibility proxy, known false matches); not human-verified",
           "systems": {}}
    for name, (pool, _rule) in SYSTEMS.items():
        code = CODES[name]
        picked = {q: int(v) for q, v in hits["per_question"][name].items()}
        in_pool, sizes = {}, []
        for q in spans:
            meta = json.loads((RUN / ds / pool / "pools" / _safe(q) / "pool.json").read_text(encoding="utf-8"))
            times = [c["time_s"] for c in meta["candidates"]]
            sizes.append(len(times))
            in_pool[q] = int(any(inside(t, spans[q]) for t in times))
            if picked[q] and not in_pool[q]:
                raise SystemExit(f"picked hit outside the pool: {name}, {q}")
        not_pool = {q: 1 - v for q, v in in_pool.items()}
        not_picked = {q: int(in_pool[q] and not picked[q]) for q in spans}
        npool = sum(in_pool.values())
        chosen_only = max(sizes) <= 4       # R2-R5 save only the frames they show: the two stages cannot be split
        s = {"pool": pool, "pool_is_the_chosen_frames": chosen_only, "candidates_mean": round(float(np.mean(sizes)), 1), "candidates_median": float(np.median(sizes)),
             "in_pool": boot_ci(in_pool, lecture), "picked_hit@4": boot_ci(picked, lecture),
             "lost_candidate_stage": boot_ci(not_pool, lecture), "lost_selection_stage": boot_ci(not_picked, lecture),
             "picked_given_in_pool": round(100 * sum(picked.values()) / npool, 1) if npool else None}
        if round(100 * float(np.mean(list(picked.values()))), 1) != hits["systems"][name]["hit@4"]:
            raise SystemExit(f"hit@4 differs from the report: {name}")
        if qa is not None:
            rec = qa[code]
            if any(rec[q][1] != picked[q] for q in spans):
                raise SystemExit(f"answer records' primary_hit differs from the hits report: {code}")
            qonly = {q: qa["q_only"][q][0] for q in spans}
            cells = {}
            for label, f in (("picked", lambda q: picked[q]), ("in_pool_not_picked", lambda q: not_picked[q]),
                             ("not_in_pool", lambda q: not_pool[q])):
                ids = [q for q in spans if f(q)]
                hard = [q for q in ids if not qonly[q]]
                cells[label] = {"n": len(ids), "correct": sum(rec[q][0] for q in ids),
                                "accuracy": round(100 * sum(rec[q][0] for q in ids) / len(ids), 1) if ids else None,
                                "n_without_q_only_correct": len(hard),
                                "accuracy_without_q_only_correct":
                                    round(100 * sum(rec[q][0] for q in hard) / len(hard), 1) if hard else None}
            s["answers"] = cells
        out["systems"][code] = s

    if qa is not None:  # the four-way outcome for S2m, the main system
        rec, pk = qa["S2m"], {q: int(v) for q, v in hits["per_question"]["S2m Head A (M3AV train)"].items()}
        out["s2m_outcomes"] = {"picked_and_correct": sum(1 for q in spans if pk[q] and rec[q][0]),
                               "picked_and_wrong": sum(1 for q in spans if pk[q] and not rec[q][0]),
                               "not_picked_and_correct": sum(1 for q in spans if not pk[q] and rec[q][0]),
                               "not_picked_and_wrong": sum(1 for q in spans if not pk[q] and not rec[q][0])}

    Path(f"reports/m3av/evidence_loss_{a.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda t: f"{t[0]} ({t[1]}..{t[2]})"  # noqa: E731
    lines = [f"# Where the source slide is lost, M3AV {a.split} ({out['questions']} questions, {out['lectures']} lectures)", "",
             "Automatic source-slide intervals (a visibility proxy with known false matches), not human-verified. "
             "Percent of questions with 95% lecture-bootstrap intervals (10,000 resamples, seed 20261101). "
             "Descriptive; no p-values." + (" Test is revealed: post hoc, never used to choose a method."
                                            if a.split == "test" else " Dev: the basis for any later redesign."), "",
             "| System | Candidates (mean) | Slide in pool | Lost at candidates | Lost at selection | Picked (hit@4) | Picked if in pool |",
             "|---|---|---|---|---|---|---|"]
    for code, s in out["systems"].items():
        if s["pool_is_the_chosen_frames"]:
            lines.append(f"| {code} | {s['candidates_mean']} (saved: the chosen frames only) | - | - | - | "
                         f"{f(s['picked_hit@4'])} | - |")
            continue
        lines.append(f"| {code} | {s['candidates_mean']} | {f(s['in_pool'])} | {f(s['lost_candidate_stage'])} | "
                     f"{f(s['lost_selection_stage'])} | {f(s['picked_hit@4'])} | {s['picked_given_in_pool']} |")
    lines += ["", "R2-R5 save only the frames they show (R3-R5 rank 96 dense frames first), so their loss is not split by "
                  "stage; their misses are all 'not picked'."]
    if qa is not None:
        lines += ["", "## Answer accuracy by stage (Qwen3-VL-4B, secondary)", "",
                  "Accuracy among the questions in each cell; in brackets, the same without questions the "
                  "question-only control answers correctly.", "",
                  "| System | Picked: n, acc. | In pool, not picked: n, acc. | Not in pool: n, acc. |", "|---|---|---|---|"]
        cell = lambda c: (f"{c['n']}, {c['accuracy']}% [{c['n_without_q_only_correct']}, "  # noqa: E731
                          f"{c['accuracy_without_q_only_correct']}%]" if c["n"] else "-")
        for code, s in out["systems"].items():
            c = s["answers"]
            if s["pool_is_the_chosen_frames"]:
                lines.append(f"| {code} | {cell(c['picked'])} | not picked (all stages): {cell(c['not_in_pool'])} | - |")
                continue
            lines.append("| " + code + " | " + " | ".join(cell(c[k]) for k in ("picked", "in_pool_not_picked", "not_in_pool")) + " |")
        o = out["s2m_outcomes"]
        lines += ["", f"S2m, all {out['questions']} questions: picked and correct {o['picked_and_correct']}, picked and wrong "
                      f"{o['picked_and_wrong']}, not picked and correct {o['not_picked_and_correct']}, not picked and wrong "
                      f"{o['not_picked_and_wrong']}."]
    Path(f"reports/m3av/evidence_loss_{a.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
