"""Gates G2 (scorer audit) and G1 (do frames help?) on EduVidQA training videos.

G1 asks whether frames add anything to open-ended answers on this benchmark before any
frame-selection method is built on it. Protocol (timestamp-given, as in the EduVidQA
paper): the question's timestamp t is known; the transcript is every caption line in
[t - 120 s, t + 120 s]. Two conditions, identical text:

    T    transcript only
    TF   transcript + 4 frames evenly spaced in [t - 30 s, t + 10 s]
         (the adapter's evidence interval; the paper measured ~35 s timestamp error)

Only OFFICIAL TRAINING videos are used (never the synthetic or real test sets).
Questions: fixed sha256 order, at most 5 per video, up to --n.

Stages (each resumable, each under a CallBudget with call AND time limits):

    answer       generate T and TF answers with the frozen Qwen3-VL-2B
    judge-audit  both local judges score a fixed 50-answer audit subset
    audit-sheet  write the audit items (no judge scores) for human labelling
    judge-rest   the chosen judge scores the remaining answers
    analyze      G2 agreement (if human labels exist) and the G1 paired difference

    python scripts/v3_gate_frames.py answer --n 100
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

RUN = Path("runs/v3_gate")
DATA = Path("data/eduvidqa")
LIMITS = {"answer": (200, 2160.0), "judge": (300, 1440.0)}     # (fresh calls, seconds): the authorised gates budget
AUDIT_N = 25                                                     # questions in the audit; x2 conditions = 50 answers


def _t_from_question(q: str) -> float | None:
    m = re.match(r"\s*At (\d+):(\d{2})(?::(\d{2}))?", q)
    if not m:
        return None
    a, b, c = (int(x) if x else None for x in m.groups())
    return float(a * 3600 + b * 60 + c) if c is not None else float(a * 60 + b)


def select_questions(n: int) -> list[dict]:
    from videoqa.transcript import load_transcript  # noqa: PLC0415

    access = json.loads((DATA / "video_access.json").read_text(encoding="utf-8"))
    ok = {v for v, r in access.items() if r.get("ok")}
    rows = [json.loads(s) for s in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    per_video, out = collections.Counter(), []
    for r in sorted(rows, key=lambda r: hashlib.sha256(r["qa_id"].encode()).hexdigest()):
        vid = r["video_id"]
        if r["official_split"] != "train" or vid not in ok or per_video[vid] >= 5:
            continue
        tpath = next((DATA / "transcripts" / f"{vid}{e}" for e in (".json", ".vtt")
                      if (DATA / "transcripts" / f"{vid}{e}").exists()), None)
        t = _t_from_question(r["question"])
        if tpath is None or t is None:
            continue
        segs = [s for s in load_transcript(tpath, vid).segments if s.end_s > t - 120 and s.start_s < t + 120]
        if not segs:
            continue
        out.append({**r, "t": t, "transcript_path": str(tpath)})
        per_video[vid] += 1
        if len(out) >= n:
            break
    return out


def _excerpt(r):
    from videoqa.transcript import load_transcript  # noqa: PLC0415

    t = r["t"]
    return [s for s in load_transcript(Path(r["transcript_path"]), r["video_id"]).segments
            if s.end_s > t - 120 and s.start_s < t + 120]


def _frames(r):
    from videoqa.frames import decode_at, probe  # noqa: PLC0415

    path = DATA / "videos" / f"{r['video_id']}.mp4"
    dur = probe(str(path)).duration_s
    t = r["t"]
    times = [min(max(0.0, x), dur - 0.05) for x in np.linspace(t - 30, t + 10, 4)]
    return decode_at(str(path), times, max_side=640, video_id=r["video_id"]).frames


def stage_answer(a) -> None:
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v3.openended import AnswerCache, answerer_identity, make_open_answerer  # noqa: PLC0415

    RUN.mkdir(parents=True, exist_ok=True)
    qs = select_questions(a.n)
    (RUN / "questions.json").write_text(json.dumps([{k: q[k] for k in ("qa_id", "video_id", "t")} for q in qs],
                                                   indent=1), encoding="utf-8")
    cfg = load_config(a.config)
    budget = CallBudget(RUN / "budget_answer.json", *LIMITS["answer"])
    cache = AnswerCache(RUN / "answers.jsonl", answerer_identity(cfg), make_open_answerer(cfg), budget)
    for n, r in enumerate(qs):
        excerpt, frames = _excerpt(r), _frames(r)
        missing = sum(cache.get(r["question"], excerpt, f) is None for f in ([], frames))
        if not budget.reserve(missing):
            print(f"budget reached before question {n}; stopping cleanly")
            break
        for f in ([], frames):
            cache.answer(r["qa_id"], r["question"], excerpt, f)
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(qs)} fresh={cache.calls} {cache.seconds / max(cache.calls, 1):.1f}s/answer", flush=True)
    print(json.dumps({"questions": len(qs), "videos": len({q['video_id'] for q in qs}), "fresh_answers": cache.calls,
                      "budget": {"calls": budget.calls, "seconds": round(budget.seconds)}}, indent=1))


def _answer_rows(a) -> list[dict]:
    """(question, condition, answer) rows for every selected question with both answers cached."""
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v3.openended import AnswerCache, answerer_identity  # noqa: PLC0415

    cache = AnswerCache(RUN / "answers.jsonl", answerer_identity(load_config(a.config)))
    meta = {q["qa_id"]: q for q in json.loads((RUN / "questions.json").read_text(encoding="utf-8"))}
    rows_by_id = {json.loads(s)["qa_id"]: json.loads(s) for s in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines()
                  if s.strip()}
    out = []
    for qid, m in meta.items():
        r = {**rows_by_id[qid], **m, "transcript_path": next(
            str(DATA / "transcripts" / f"{m['video_id']}{e}") for e in (".json", ".vtt")
            if (DATA / "transcripts" / f"{m['video_id']}{e}").exists())}
        excerpt, frames = _excerpt(r), _frames(r)
        ans = {c: cache.get(r["question"], excerpt, f) for c, f in (("T", []), ("TF", frames))}
        if all(ans.values()):
            for c, rec in ans.items():
                out.append({"qa_id": qid, "video_id": r["video_id"], "condition": c, "question": r["question"],
                            "reference": r["gold_answer"], "answer": rec["text"]})
    return out


def _audit_ids(rows) -> set[str]:
    qids = sorted({r["qa_id"] for r in rows}, key=lambda q: hashlib.sha256(("audit:" + q).encode()).hexdigest())
    return set(qids[:AUDIT_N])


def stage_judge(a, which: str) -> None:
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v3.judge import LocalJudge, verdict_key  # noqa: PLC0415

    rows = _answer_rows(a)
    audit = _audit_ids(rows)
    todo = [r for r in rows if (r["qa_id"] in audit) == (which == "audit")]
    judges = ("phi4mini", "qwen3vl4b") if which == "audit" else (a.judge,)
    budget = CallBudget(RUN / "budget_judge.json", *LIMITS["judge"])
    for name in judges:
        j = LocalJudge(name, RUN / f"verdicts_{name}.jsonl", budget=budget)
        missing = sum(verdict_key(name, r["question"], r["reference"], r["answer"]) not in j.records for r in todo)
        if not budget.reserve(missing):
            print(f"judge budget cannot cover {missing} verdicts for {name}; stopping")
            break
        for r in todo:
            j.judge(r["qa_id"], r["question"], r["reference"], r["answer"])
        j.unload()
        print(f"{name}: {j.calls} fresh verdicts, {j.hits} cached; budget {budget.calls}/{budget.max_calls}")


def stage_audit_sheet(a) -> None:
    every = _answer_rows(a)
    audit = _audit_ids(every)
    rows = [r for r in every if r["qa_id"] in audit]
    # Shuffle conditions so the labeller cannot tell which answer had frames.
    rows.sort(key=lambda r: hashlib.sha256(("sheet:" + r["qa_id"] + r["condition"]).encode()).hexdigest())
    items = [{"item": i, "question": r["question"].strip(), "reference": r["reference"], "answer": r["answer"]}
             for i, r in enumerate(rows)]
    key = [{"item": i, "qa_id": r["qa_id"], "condition": r["condition"]} for i, r in enumerate(rows)]
    Path("reports/v3_gate").mkdir(parents=True, exist_ok=True)
    Path("reports/v3_gate/audit_items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    (RUN / "audit_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} audit items written (conditions hidden)")


GEMINI_MODEL = "gemini-3.8-flash"            # strongest model the free tier serves (Pro returns quota 0)
GEMINI_LIMITS = (400, 3600.0)                 # approved by the user for the gate: calls, API seconds


def _video_bootstrap(diffs: dict, videos: dict, repeats: int = 5000) -> dict:
    by: dict[str, list[float]] = {}
    for q, d in diffs.items():
        by.setdefault(videos[q], []).append(d)
    s = np.array([sum(v) for v in by.values()])
    c = np.array([len(v) for v in by.values()])
    ids = np.random.default_rng(0).integers(0, len(s), (repeats, len(s)))
    b = s[ids].sum(1) / c[ids].sum(1)
    return {"difference": float(s.sum() / c.sum()), "ci95_video_bootstrap": np.percentile(b, [2.5, 97.5]).tolist()}


def stage_score(a) -> None:
    """FactQA (Gemini) + BLEU-1/ROUGE-L/METEOR/entailment for every answer; G1 paired differences."""
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v3.factqa import GeminiFactQA, verdict_key  # noqa: PLC0415
    from videoqa.v3.metrics import NLI_MODEL, Entailment, bleu1, meteor, rouge_l  # noqa: PLC0415

    rows = _answer_rows(a)
    budget = CallBudget(RUN / "budget_gemini.json", *GEMINI_LIMITS)
    fq = GeminiFactQA(GEMINI_MODEL, RUN / "factqa_gemini.jsonl", budget)
    ent = Entailment()(([(r["reference"], r["answer"]) for r in rows]))
    scored = []
    for n, (r, e) in enumerate(zip(rows, ent, strict=True)):
        need = sum(verdict_key(GEMINI_MODEL, d, r["question"], a1, a2) not in fq.records
                   for d, a1, a2 in (("precision", r["answer"], r["reference"]), ("recall", r["reference"], r["answer"])))
        if not budget.reserve(need):                    # both calls of an answer, or none
            print("Gemini budget reached; stopping cleanly")
            break
        f = fq.score(r["qa_id"], r["question"], r["reference"], r["answer"])
        scored.append({"qa_id": r["qa_id"], "video_id": r["video_id"], "condition": r["condition"],
                       "factqa_precision": f["precision"], "factqa_recall": f["recall"],
                       "bleu1": bleu1(r["reference"], r["answer"]), "rouge_l": rouge_l(r["reference"], r["answer"]),
                       "meteor": meteor(r["reference"], r["answer"]), "entailment": e,
                       "answer_words": len(r["answer"].split())})
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(rows)} scored; Gemini calls {budget.calls}/{budget.max_calls}", flush=True)
    (RUN / "scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in scored), encoding="utf-8")

    metrics = ("factqa_precision", "factqa_recall", "bleu1", "rouge_l", "meteor", "entailment", "answer_words")
    by = {}
    for x in scored:
        by.setdefault(x["qa_id"], {})[x["condition"]] = x
    pairs = {q: d for q, d in by.items() if "T" in d and "TF" in d}
    videos = {q: d["T"]["video_id"] for q, d in pairs.items()}
    out = {"gate": "G1: do frames help open-ended EduVidQA answers? (official training videos only)",
           "questions": len(pairs), "videos": len(set(videos.values())), "judge": GEMINI_MODEL,
           "judge_versions": sorted({v.get("model_version") for v in fq.records.values()}),
           "nli_model": list(NLI_MODEL), "gemini_calls": budget.calls,
           "unparsed_factqa": sum(v.get("score") is None for v in fq.records.values()), "metrics": {}}
    for m in metrics:
        ok = {q: d for q, d in pairs.items() if d["T"][m] is not None and d["TF"][m] is not None}
        out["metrics"][m] = {"transcript_only": float(np.mean([d["T"][m] for d in ok.values()])),
                             "transcript_plus_frames": float(np.mean([d["TF"][m] for d in ok.values()])),
                             "frames_minus_transcript": _video_bootstrap({q: d["TF"][m] - d["T"][m] for q, d in ok.items()},
                                                                         videos),
                             "paired_questions": len(ok)}
    Path("reports/v3_gate").mkdir(parents=True, exist_ok=True)
    Path("reports/v3_gate/g1_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("answer", "judge-audit", "audit-sheet", "judge-rest", "score"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--judge", choices=("phi4mini", "qwen3vl4b"), default="phi4mini")
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"answer": stage_answer, "judge-audit": lambda x: stage_judge(x, "audit"),
     "audit-sheet": stage_audit_sheet, "judge-rest": lambda x: stage_judge(x, "rest"),
     "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
