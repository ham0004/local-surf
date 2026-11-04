"""Deviation and sensitivity appendix for the sealed test answer-model run (auditor P109-P110, P113).

Reads only: the sealed records, sessions and ledger (runs/m3av/qa/test/), the outputs seal, the test report and
both run logs. Changes none of them. Writes docs/v3/m3av_qa_test_deviation.{json,md}.

The deviation: in session 1 the `t_only` request for m3av:NIH-D369A:1661.45 ran out of GPU memory; the handler's
own `torch.cuda.empty_cache()` failed too, so no record and no ledger charge were written and the run stopped.
The restart attempted the same request again, against the sealed rule "a failed record, wrong, never retried".
Sensitivity: the analyses with that item scored as the sealed rule would have scored it (failed: wrong, and an
unknown control left out of the t_only mask).

    python scripts/m3av_qa_deviation.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3av_hits  # noqa: E402
from m3av_freeze_supplement import sha  # noqa: E402
from m3av_qa import CONDITIONS, CONTRASTS, CONTROLS, load_records, root  # noqa: E402

CRASH = ("t_only", "m3av:NIH-D369A:1661.45")
LOGS = [Path("runs/m3av/qa_test_run.log"), Path("runs/m3av/qa_test_run2.log")]
SEAL = Path("docs/v3/m3av_qa_test_outputs.json")
REPORT = Path("reports/m3av/qa_test.json")
OUT = Path("docs/v3/m3av_qa_test_deviation")
OOM = "torch.AcceleratorError: CUDA error: out of memory"


def main() -> None:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    rec_file = root("test") / "records.jsonl"
    if sha(rec_file) != seal["records"]["sha256_lf"] or sha(root("test") / "sessions.jsonl") != seal["sessions"]["sha256_lf"]:
        raise SystemExit("records or sessions differ from the seal")
    rows = load_records("test")
    sessions = [json.loads(x) for x in (root("test") / "sessions.jsonl").read_text().splitlines() if x.strip()]
    ledger = json.loads((root("test") / "ledger.json").read_text())
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rule = json.loads(Path("docs/v3/m3av_freeze_supplement3.json").read_text(encoding="utf-8"))["rules"]["failure"]

    # where session 1 stopped: session 2 wrote the last calls + reused rows, starting with the crashed request
    s2 = sessions[-1]
    n1 = len(rows) - s2["calls"] - s2["reused"]
    if len(sessions) != 1 or (rows[n1]["condition"], rows[n1]["qa_id"]) != CRASH or rows[n1]["reused_from"] is not None:
        raise SystemExit("records do not match the recorded crash")
    first = rows[:n1]
    per_q = {}
    for r in first:
        per_q[r["qa_id"]] = per_q.get(r["qa_id"], 0) + 1
    logs = [p.read_text(encoding="utf-8", errors="replace") for p in LOGS]
    log_facts = [{"file": p.as_posix(), "sha256_lf": sha(p), "oom_errors": t.count(OOM),
                  "oom_inside_handler": t.count("During handling of the above exception"),
                  "ends": [x for x in t.strip().splitlines() if x.strip()][-1]} for p, t in zip(LOGS, logs)]
    if log_facts[0]["oom_errors"] != 2 or log_facts[0]["oom_inside_handler"] != 1 or log_facts[1]["oom_errors"]:
        raise SystemExit("logs do not match one crash in session 1 and none in session 2")

    r = {(x["condition"], x["qa_id"]): x for x in rows}
    qids = sorted({x["qa_id"] for x in rows})
    lecture = {q: r[("S1", q)]["lecture"] for q in qids}
    acc = {c: {q: float(r[(c, q)]["correct"]) for q in qids} for c in CONDITIONS}

    def masks(failed: set) -> dict:
        m = {"all": qids}
        for c in CONTROLS:
            m[f"without {c}-correct"] = [q for q in qids if (c, q) not in failed and not r[(c, q)]["failed"]
                                         and not r[(c, q)]["correct"]]
        return m

    orig, sens = masks(set()), masks({CRASH})
    if r[CRASH]["correct"]:
        raise SystemExit("the re-answered item is correct: the all-question accuracies would change")
    tables, worst = {}, 0.0
    for m in orig:
        rowsm = []
        for x, y in CONTRASTS:
            a, b = (m3av_hits.paired({q: acc[x][q] for q in v}, {q: acc[y][q] for q in v}, lecture) for v in (orig[m], sens[m]))
            da, db = (100 * statistics.fmean(acc[x][q] - acc[y][q] for q in v) for v in (orig[m], sens[m]))
            worst = max(worst, abs(db - da))
            rowsm.append({"contrast": f"{x} - {y}", "as_run": a, "crash_as_failed": b, "point_change": round(db - da, 4),
                          "same_zero_crossing": (a["lo"] > 0, a["hi"] < 0) == (b["lo"] > 0, b["hi"] < 0)})
        tables[m] = {"as_run": {"questions": len(orig[m]), "lectures": len({lecture[q] for q in orig[m]})},
                     "crash_as_failed": {"questions": len(sens[m]), "lectures": len({lecture[q] for q in sens[m]})},
                     "contrasts": rowsm}
    for m, t in tables.items():   # the run's report must agree with the as-run column
        for c in t["contrasts"]:
            rep = report["contrasts"][m][c["contrast"]]
            if any(rep[k] != c["as_run"][k] for k in ("diff", "lo", "hi", "p", "n", "lectures")):
                raise SystemExit(f"report differs: {m} {c['contrast']}")

    def med(c, new_only):
        v = [x["answer_seconds"] for x in rows if x["condition"] == c and (x["reused_from"] is None or not new_only)]
        return round(statistics.median(v), 3) if v else None

    out = {
        "item": {"condition": CRASH[0], "qa_id": CRASH[1], "lecture": r[CRASH]["lecture"]},
        "sealed_rule": rule,
        "what_happened": "session 1: a driver-level CUDA out-of-memory error during generation; the handler caught it, "
                         "then its own torch.cuda.empty_cache() raised the same error, so no record and no ledger charge "
                         "were written and the run stopped (exit 1). One restart (session 2) through the unchanged sealed "
                         "driver attempted the same request again and answered it (wrong). The retry is a protocol "
                         "deviation; the original records, seal and report are kept unchanged.",
        "re_answer": {k: r[CRASH][k] for k in ("option", "correct", "failed", "answer_seconds", "text_words",
                                               "prompt_text_tokens", "generated_tokens")},
        "session_1": {"rows_written": n1, "complete_questions": sum(v == len(CONDITIONS) for v in per_q.values()),
                      "questions_touched": len(per_q), "summary": "unavailable (not written before the crash; not "
                      "substituted by session 2)"},
        "session_2": s2,
        "logs": log_facts,
        "accounting": {
            "saved_rows": len(rows), "saved_failed_rows": sum(bool(x["failed"]) for x in rows),
            "physical_failed_attempts_known": 1, "ledger_charged_new_calls": ledger["calls"],
            "reused_rows": sum(x["reused_from"] is not None for x in rows),
            "attempted_answer_calls_at_least": ledger["calls"] + 1,
            "ledger_answer_seconds": round(ledger["seconds"], 1),
            "note": "ledger seconds cover completed new calls only: the failed attempt, model loading, pool loading, "
                    "selection and restart time are not included, so this is not end-to-end runtime"},
        "memory_gib": {"note": "session 2 only; torch.cuda.max_memory_allocated / max_memory_reserved divided by 2^30 "
                               "(GiB; the report's 'gb' keys are GiB). Allocated = tensor storage; reserved = memory "
                               "held by the caching allocator. Neither measures dedicated versus Windows shared "
                               "residency; shared-memory backing is a plausible explanation for reserved above the card, "
                               "not a measurement. Session 1 peak: unavailable",
                       "allocated": s2["torch_peak_allocated_gb"], "reserved": s2["torch_peak_reserved_gb"],
                       "device_total": s2["gpu_total_gb"]},
        "answer_seconds_median": {c: {"all_rows": med(c, False), "new_calls_only": med(c, True)} for c in CONDITIONS},
        "answer_seconds_note": "time of the answer-model call (generation, incl. input processing), not end-to-end "
                               "latency; reused rows copy the originating call's time",
        "sensitivity": tables,
        "largest_point_change": round(worst, 4),
        "inputs": {"records_sha256_lf": seal["records"]["sha256_lf"], "outputs_seal_sha256_lf": sha(SEAL),
                   "report_json_sha256_lf": sha(REPORT), "report_md_sha256_lf": sha(Path("reports/m3av/qa_test.md"))}}
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    t = tables
    lines = ["# Test answer-model run: deviation, sensitivity and resource accounting", "",
             "Additive appendix to `reports/m3av/qa_test.{json,md}` (auditor P109-P110, P113). The original records, "
             "outputs seal (`docs/v3/m3av_qa_test_outputs.json`) and report are unchanged. Generated by "
             "`scripts/m3av_qa_deviation.py` from the sealed records, sessions, ledger and both run logs; data in "
             "`docs/v3/m3av_qa_test_deviation.json`. Secondary and descriptive throughout.", "",
             "## The deviation", "",
             f"- **Item:** `{CRASH[0]}` on `{CRASH[1]}` (lecture {r[CRASH]['lecture']}).",
             f"- **Sealed rule** (third freeze supplement): \"{rule}\".",
             "- **What happened:** " + out["what_happened"],
             f"- **Session 1** wrote {n1:,} rows: {out['session_1']['complete_questions']} complete questions plus "
             f"{n1 - len(CONDITIONS) * out['session_1']['complete_questions']} conditions of question "
             f"{out['session_1']['questions_touched']} ({out['session_1']['questions_touched']} touched). Its session "
             "summary (model load, peak memory) is unavailable.",
             f"- **Logs:** `{LOGS[0].as_posix()}` (one out-of-memory error, a second one inside the handler, ends "
             f"`{log_facts[0]['ends']}`); `{LOGS[1].as_posix()}` (no error, ends `{log_facts[1]['ends']}`). One crash and one "
             "restart; no other restart happened.", "",
             "## Failures and calls", "",
             "| | Count |", "|---|---:|",
             f"| saved rows | {len(rows):,} |",
             f"| saved rows marked failed | {out['accounting']['saved_failed_rows']} |",
             "| known failed physical attempts | 1 |",
             f"| ledger-charged new calls (completed) | {ledger['calls']:,} |",
             f"| rows reusing an identical earlier request | {out['accounting']['reused_rows']} |",
             f"| answer-model calls attempted (at least) | {ledger['calls'] + 1:,} |",
             f"| ledger answer seconds | {ledger['seconds']:,.0f} |", "",
             "\"0 failed\" in the report describes the saved rows only. The ledger seconds cover completed new calls; "
             "the failed attempt, model and pool loading, selection and the restart are not included, so they are "
             "not end-to-end runtime.", "",
             "## Sensitivity: the item scored as the sealed rule would have scored it", "",
             "A failed answer is wrong, so every accuracy and contrast on all 853 questions and on the mask without "
             "q_only-correct questions is unchanged. A failed control is unknown and is left out of that control's "
             f"mask: the mask without t_only-correct questions has {t['without t_only-correct']['crash_as_failed']['questions']} "
             f"questions on {t['without t_only-correct']['crash_as_failed']['lectures']} lectures instead of "
             f"{t['without t_only-correct']['as_run']['questions']} on {t['without t_only-correct']['as_run']['lectures']}.", "",
             "| Contrast | As run (323 questions) | Crash scored failed (322) | Point change | Same zero-crossing |",
             "|---|---|---|---:|---|"]
    fmt = lambda s: f"{s['diff']:+.1f} ({s['lo']:+.1f} to {s['hi']:+.1f}), p {s['p']:.3g}"  # noqa: E731
    for c in t["without t_only-correct"]["contrasts"]:
        lines.append(f"| {c['contrast']} | {fmt(c['as_run'])} | {fmt(c['crash_as_failed'])} | {c['point_change']:+.4f} | "
                     f"{'yes' if c['same_zero_crossing'] else 'no'} |")
    lines += ["", f"Largest change in a point estimate: {worst:.4f} points (0.3 at displayed precision). Which "
              "intervals cross zero is unchanged. Paired bootstrap over lectures, 10,000 resamples, seed 20261101; raw "
              "p uncorrected. This addresses the recorded retry only, not other operational or label bias.", "",
              "## Resources (measured, session 2 only)", "",
              f"- Peak torch allocated {s2['torch_peak_allocated_gb']} GiB and reserved {s2['torch_peak_reserved_gb']} GiB "
              f"on an {s2['gpu']} with {s2['gpu_total_gb']} GiB. " + out["memory_gib"]["note"].split("; ", 1)[1] + ".",
              "- Defensible claim: the run executed locally on this single card, with one out-of-memory crash and one "
              "restart and incomplete peak accounting. Strict residence within 16 GiB and failure-free completion "
              "are not demonstrated.",
              "- Median answer-call seconds (the model call, not end-to-end latency); \"all rows\" is what the report "
              "shows, where reused rows copy the originating call's time:", "",
              "| Condition | all rows | new calls only |", "|---|---:|---:|"]
    lines += [f"| {c} | {v['all_rows']} | {v['new_calls_only']} |" for c, v in out["answer_seconds_median"].items()]
    lines += ["", "Selection and pool-build timings are partial: `select_seconds` excludes some query preparation, and "
              "preprocessing (decoding, embedding) is measured separately; the primary hit report has no cost columns.", ""]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
