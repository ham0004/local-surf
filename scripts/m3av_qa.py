"""Secondary answer-model evaluation of the lecture study (docs/v3/m3av_protocol.md, "Secondary"; third freeze
supplement, auditor P103). Not part of the primary measure or its Holm family.

Conditions: the 10 frozen QA systems (m3av_freeze.SYSTEMS; S1t shows S1's frames with 600 transcript words) and
two controls on the same questions: q_only (question and options, no frames, no transcript) and t_only (no frames,
600 words chosen by BM25 over the whole transcript, `bm25full`). Every condition uses the same frozen answerer
(configs/gpu_16gb_4b.yaml: Qwen3-VL-4B at a pinned revision, bfloat16, greedy, 64 new tokens), prompt and K = 4.

One record per condition and question in runs/m3av/qa/<split>/records.jsonl, with what was shown (frame ids,
times, pixel digests, transcript line ids and words), the request key, the raw reply, the parsed option and its
flags, measured tokens and times. Before a question is answered, the recomputed frames must give the stored
primary and S-src hits (reports/m3av/hits_<split>.json); otherwise the run stops.

Rules:
- An identical request (same key: answerer identity, prompt text, frame digests and labels) is answered once per
  run; later conditions reuse that reply and record `reused_from`. Nothing is read from earlier caches.
- Explicit abstention: option None, scored wrong. No letter and no abstention: the model's most likely letter,
  flagged `forced` (the existing parser).
- GPU out of memory: a `failed` record, scored wrong, never retried. Any other error stops the run with nothing
  written for that question; a restart resumes from the records (unattempted rows are missing, not failed).
- A repeated (condition, question) row, or two replies to one key that disagree, stop the run.
- Ledger runs/m3av/qa/<split>/ledger.json (LIMITS); reaching it stops cleanly and the run is INCOMPLETE.
- `seal` hashes the records once all conditions cover all questions; `report` needs the seal.

    python scripts/m3av_qa.py answer --split dev --limit 24      # dev check (a prefix of the hash order)
    python scripts/m3av_qa.py report --split dev --limit 24
    python scripts/m3av_qa.py answer --split test                # resumable
    python scripts/m3av_qa.py seal --split test                  # docs/v3/m3av_qa_test_outputs.json
    python scripts/m3av_qa.py report --split test                # reports/m3av/qa_test.{json,md}
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m3av_hits  # noqa: E402
from m3av_freeze import SYSTEMS as FROZEN  # noqa: E402
from v3_baseline_cycle import items, pool_dir, request_key, select, transcript_for  # noqa: E402

CONFIG, K = "configs/gpu_16gb_4b.yaml", 4
CONDITIONS = {k: (p, "file:{ds}_hybrid_ev_set" if k == "S3" else s, tp, w) for k, (p, s, w, tp) in FROZEN.items()}
CONDITIONS.update({"q_only": ("v2", "none", "none", 0), "t_only": ("v2", "none", "bm25full", 600)})
CONTROLS = ("q_only", "t_only")
HITS_NAME = {k: next(n for n, (p, s) in m3av_hits.SYSTEMS.items() if (p, s) == (CONDITIONS[k][0], CONDITIONS[k][1]))
             for k in CONDITIONS if k not in CONTROLS}
CONTRASTS = [("S2m", "R5"), ("S2m", "S1"), ("S2m", "S2c"), ("S2m", "R3"), ("S2m", "R4"), ("S3", "S1"), ("S1t", "S1")]
LIMITS = {"dev": (12 * 514, 30 * 3600.0), "test": (12 * 853, 96 * 3600.0)}   # (new answer calls, answer seconds)


def root(split: str) -> Path:
    return Path(f"runs/m3av/qa/{split}")


def load_records(split: str) -> list[dict]:
    f = root(split) / "records.jsonl"
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []
    seen, keys = set(), {}
    for r in rows:
        if (r["condition"], r["qa_id"]) in seen:
            raise SystemExit(f"repeated record: {r['condition']} {r['qa_id']}")
        seen.add((r["condition"], r["qa_id"]))
        a = (r["option"], r["raw"], r["failed"])
        if keys.setdefault(r["request_key"], a) != a:
            raise SystemExit(f"two different replies to one request: {r['request_key']}")
    return rows


class Ledger:
    def __init__(self, path: Path, max_calls: int, max_seconds: float) -> None:
        self.path, self.max_calls, self.max_seconds = path, max_calls, max_seconds
        s = json.loads(path.read_text()) if path.exists() else {}
        self.calls, self.seconds = s.get("calls", 0), s.get("seconds", 0.0)

    def room(self) -> bool:
        return self.calls < self.max_calls and self.seconds < self.max_seconds

    def charge(self, seconds: float) -> None:
        self.calls, self.seconds = self.calls + 1, self.seconds + seconds
        self.path.write_text(json.dumps({"max_calls": self.max_calls, "max_seconds": self.max_seconds,
                                         "calls": self.calls, "seconds": self.seconds}, indent=1))


def stage_answer(a) -> None:
    import torch  # noqa: PLC0415

    from videoqa.answerer import AnswerRequest, build_prompt_text, make_answerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415
    from videoqa.v2.teacher import _as_frame, answerer_identity  # noqa: PLC0415

    if a.split == "test" and a.limit:
        raise SystemExit("test runs the whole cohort")
    ds = f"m3av_{a.split}"
    its = {it.qa.qa_id: it for it in items(ds)}
    qids = list(its)[: a.limit] if a.limit else list(its)
    qa = {q: it.qa for q, it in its.items()}
    hits = json.loads(Path(f"reports/m3av/hits_{a.split}.json").read_text(encoding="utf-8"))
    src = m3av_hits.source_spans(a.split, qa)
    cfg = load_config(CONFIG)
    identity = answerer_identity(cfg)
    out = root(a.split)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_records(a.split)
    done = {(r["condition"], r["qa_id"]) for r in rows}
    by_key = {}
    for r in rows:
        by_key.setdefault(r["request_key"], r)
    ledger = Ledger(out / "ledger.json", *LIMITS[a.split])
    todo = [q for q in qids if any((c, q) not in done for c in CONDITIONS)]
    session = {"start": datetime.datetime.now().isoformat(timespec="seconds"), "split": a.split, "questions": len(qids),
               "to_answer": len(todo), "calls": 0, "reused": 0, "failed": 0}
    if todo:
        enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
        qo = dict(zip(todo, enc.embed_texts([qa[q].question + " Options: " + ", ".join(qa[q].options) for q in todo]),
                      strict=True))
        del enc
        torch.cuda.empty_cache()
    model, stop = None, False
    for n, q in enumerate(todo, 1):
        pools = {}
        for cond, (pool, rule, tp, words) in CONDITIONS.items():
            if (cond, q) in done:
                continue
            p = pools.get(pool) or pools.setdefault(pool, load_pool(pool_dir(ds, pool), q))
            t0 = time.perf_counter()
            frames = sorted(select(p, rule.format(ds=ds), K, qo[q]), key=lambda c: c.time_s)
            t_sel = time.perf_counter() - t0
            if cond not in CONTROLS:                 # the frames must be the ones the primary measure scored
                h = int(any(x - 1.0 <= c.time_s <= y + 1.0 for c in frames for x, y in qa[q].evidence_intervals_s))
                hs = int(any(x - 1.0 <= c.time_s <= y + 1.0 for c in frames for x, y in src[q]))
                name = HITS_NAME[cond]
                if (h, hs) != (hits["per_question"][name][q], hits["s_src"]["per_question"][name][q]):
                    raise SystemExit(f"recomputed frames do not reproduce the stored hits: {cond} {q}")
            t0 = time.perf_counter()
            ex = transcript_for(tp, words, p, its[q].transcript.segments, frames) if words else []
            t_text = time.perf_counter() - t0
            key = request_key(p, frames, ex, identity)
            rec = {"condition": cond, "qa_id": q, "lecture": p.video_id, "pool": pool, "selector": rule, "k": K,
                   "frame_ids": [c.id for c in frames], "frame_times": [round(c.time_s, 3) for c in frames],
                   "frame_digests": [c.digest for c in frames], "n_frames": len(frames),
                   "tpolicy": tp if words else "none", "words_allowance": words, "text_line_ids": [s.id for s in ex],
                   "text_words": sum(len(s.text.split()) for s in ex),
                   "prompt_sha256": hashlib.sha256(build_prompt_text(AnswerRequest(p.question, p.options, ex, []))
                                                   .encode()).hexdigest(),
                   "request_key": key, "select_seconds": round(t_sel, 4), "text_seconds": round(t_text, 4),
                   "pool_seconds": round(sum(p.timings.get(x, 0.0) for x in ("retrieval", "head_a", "decode", "clip")), 3),
                   "primary_hit": None if cond in CONTROLS else h, "s_src_hit": None if cond in CONTROLS else hs}
            if key in by_key:                        # identical request already answered in this run
                old = by_key[key]
                rec.update({f: old[f] for f in ("raw", "option", "abstained", "forced", "failed", "prompt_text_tokens",
                                                "visual_tokens", "generated_tokens", "answer_seconds")})
                rec["reused_from"] = old.get("reused_from") or old["condition"]
                session["reused"] += 1
            else:
                if not ledger.room():
                    stop = True
                    break
                if model is None:
                    t0 = time.perf_counter()
                    model = make_answerer(cfg)
                    model.ensure_loaded()
                    session["model_load_seconds"] = round(time.perf_counter() - t0, 1)
                t0 = time.perf_counter()
                try:
                    ans, use = model.answer(AnswerRequest(p.question, p.options, ex, [_as_frame(c) for c in frames]))
                    failed = None
                except RuntimeError as e:
                    if "out of memory" not in str(e):
                        raise
                    torch.cuda.empty_cache()
                    ans, use, failed = None, None, "gpu out of memory"
                spent = time.perf_counter() - t0
                ledger.charge(spent)
                session["calls"] += 1
                session["failed"] += failed is not None
                rec.update({"raw": ans.text if ans else None, "option": ans.option_index if ans else -1,
                            "abstained": bool(ans and ans.abstained), "forced": bool(ans and ans.forced_choice),
                            "failed": failed,
                            "prompt_text_tokens": use.text_tokens - use.generated_tokens if use else None,
                            "visual_tokens": use.visual_tokens if use else None,
                            "generated_tokens": use.generated_tokens if use else None,
                            "answer_seconds": round(spent, 3), "reused_from": None})
                by_key[key] = rec
            rec["correct"] = rec["option"] == p.gold_option_index
            with open(out / "records.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            done.add((cond, q))
        if stop:
            break
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} questions; calls {session['calls']}, reused {session['reused']}", flush=True)
    session.update({"end": datetime.datetime.now().isoformat(timespec="seconds"), "status": "budget stop" if stop else
                    "done", "ledger_calls": ledger.calls, "ledger_seconds": round(ledger.seconds, 1)})
    if torch.cuda.is_available():
        session.update({"torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)})
    with open(out / "sessions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(session) + "\n")
    if stop:
        raise SystemExit("ledger limit reached; the run is INCOMPLETE")
    print(f"answer {a.split}: {len(qids)} questions x {len(CONDITIONS)} conditions complete; {session}")


def complete(split: str, rows: list[dict], limit: int = 0) -> list[str]:
    qids = [it.qa.qa_id for it in items(f"m3av_{split}")]
    qids = qids[:limit] if limit else qids
    have = {(r["condition"], r["qa_id"]) for r in rows}
    missing = [f"{c} {q}" for c in CONDITIONS for q in qids if (c, q) not in have]
    if missing:
        raise SystemExit(f"incomplete: {len(missing)} of {len(CONDITIONS) * len(qids)} records missing")
    if len(rows) != len(CONDITIONS) * len(qids):
        raise SystemExit("records outside the cohort or the roster")
    return qids


def stage_seal(a) -> None:
    from m3av_freeze_supplement import sha  # noqa: PLC0415

    dest = Path(f"docs/v3/m3av_qa_{a.split}_outputs.json")
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    if a.limit:
        raise SystemExit("only a whole cohort is sealed")
    rows = load_records(a.split)
    qids = complete(a.split, rows)
    sessions = root(a.split) / "sessions.jsonl"
    rec = {"records": {"file": (root(a.split) / "records.jsonl").as_posix(), "rows": len(rows), "questions": len(qids),
                       "conditions": list(CONDITIONS), "sha256_lf": sha(root(a.split) / "records.jsonl")},
           "sessions": {"file": sessions.as_posix(), "sha256_lf": sha(sessions)},
           "ledger": json.loads((root(a.split) / "ledger.json").read_text())}
    dest.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{dest.as_posix()}: sha256 {sha(dest)} (LF)")


def stage_report(a) -> None:
    from m3av_freeze_supplement import sha  # noqa: PLC0415

    if a.split == "test" and a.limit:
        raise SystemExit("test reports the whole cohort")
    rows = load_records(a.split)
    qids = complete(a.split, rows, a.limit)
    seal = Path(f"docs/v3/m3av_qa_{a.split}_outputs.json")
    if a.split == "test" and (not seal.exists() or json.loads(seal.read_text(encoding="utf-8"))["records"]["sha256_lf"]
                              != sha(root(a.split) / "records.jsonl")):
        raise SystemExit("records are not sealed, or changed after the seal")
    r = {(x["condition"], x["qa_id"]): x for x in rows}
    lecture = {q: r[("S1", q)]["lecture"] for q in qids}
    acc = {c: {q: float(r[(c, q)]["correct"]) for q in qids} for c in CONDITIONS}
    masks = {"all": qids}
    control_counts = {}
    for c in CONTROLS:                               # a failed control is unknown: kept out of that mask
        failed = [q for q in qids if r[(c, q)]["failed"]]
        right = [q for q in qids if r[(c, q)]["correct"]]
        masks[f"without {c}-correct"] = [q for q in qids if q not in failed and q not in right]
        control_counts[c] = {"correct": len(right), "failed (unknown)": len(failed), "wrong": len(qids) - len(right) - len(failed)}

    def stats(c: str, qs: list) -> dict:
        g = [r[(c, q)] for q in qs]
        med = lambda f: float(np.median([x[f] for x in g if x[f] is not None])) if any(x[f] is not None for x in g) else None  # noqa: E731
        return {"accuracy": round(100 * float(np.mean([x["correct"] for x in g])), 1), "n": len(g),
                "abstained": sum(x["abstained"] for x in g), "forced": sum(x["forced"] for x in g),
                "failed": sum(bool(x["failed"]) for x in g), "frames_mean": round(float(np.mean([x["n_frames"] for x in g])), 2),
                "text_words_median": med("text_words"), "text_words_max": max(x["text_words"] for x in g),
                "prompt_text_tokens_median": med("prompt_text_tokens"), "visual_tokens_median": med("visual_tokens"),
                "generated_tokens_median": med("generated_tokens"), "answer_seconds_median": med("answer_seconds"),
                "select_seconds_median": med("select_seconds"), "pool_seconds_median": med("pool_seconds"),
                "reused": sum(x["reused_from"] is not None for x in g)}

    out = {"split": a.split, "questions": len(qids), "lectures": len(set(lecture.values())),
           "note": "secondary and descriptive: paired bootstrap over lectures (10,000, seed 20261101), raw p "
                   "uncorrected; not part of the primary Holm family. The masks depend on Qwen3-VL-4B; they are "
                   "not verified visually necessary subsets",
           "controls": control_counts, "mask_sizes": {m: len(v) for m, v in masks.items()},
           "conditions": {c: stats(c, qids) for c in CONDITIONS},
           "by_mask": {m: {c: round(100 * float(np.mean([acc[c][q] for q in v])), 1) if v else None
                           for c in CONDITIONS} for m, v in masks.items()},
           "contrasts": {m: {f"{x} - {y}": m3av_hits.paired({q: acc[x][q] for q in v}, {q: acc[y][q] for q in v},
                                                            lecture) for x, y in CONTRASTS}
                         for m, v in masks.items() if v},
           "sessions": [json.loads(x) for x in (root(a.split) / "sessions.jsonl").read_text().splitlines() if x.strip()]}
    Path("reports/m3av").mkdir(parents=True, exist_ok=True)
    Path(f"reports/m3av/qa_{a.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [f"# Answer accuracy (secondary), M3AV {a.split}: {len(qids)} questions, {out['lectures']} lectures", "",
             "Qwen3-VL-4B, K = 4, same prompt for every condition. " + out["note"] + ".", "",
             "| Condition | accuracy, all (%) | " + " | ".join(f"{m} (%)" for m in masks if m != "all")
             + " | abstained | forced | failed | frames | words (median) | prompt text tokens | visual tokens | answer s |",
             "|---|---|" + "---|" * (len(masks) - 1) + "---|---|---|---|---|---|---|---|"]
    for c, s in out["conditions"].items():
        lines.append(f"| {c} | {s['accuracy']} | " + " | ".join(str(out["by_mask"][m][c]) for m in masks if m != "all")
                     + f" | {s['abstained']} | {s['forced']} | {s['failed']} | {s['frames_mean']} | {s['text_words_median']}"
                     f" | {s['prompt_text_tokens_median']} | {s['visual_tokens_median']} | {s['answer_seconds_median']} |")
    lines += ["", "Masks: " + "; ".join(f"{m} {n}" for m, n in out["mask_sizes"].items()) + ". Controls: "
              + "; ".join(f"{c} {v}" for c, v in control_counts.items()), ""]
    for m, cs in out["contrasts"].items():
        lines += [f"## Contrasts, {m} ({len(masks[m])} questions)", "", "| Contrast | points (95% interval) | raw p |",
                  "|---|---|---|"]
        lines += [f"| {k} | {v['diff']:+.1f} ({v['lo']:+.1f}..{v['hi']:+.1f}) | {v['p']:.4g} |" for k, v in cs.items()]
        lines.append("")
    Path(f"reports/m3av/qa_{a.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def stage_check(a) -> None:
    """Dev only: the execution and accounting checks of the supplement, from the saved dev records (no model)."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    if a.split != "dev":
        raise SystemExit("the check runs on dev")
    rows = load_records(a.split)
    qids = complete(a.split, rows, a.limit)
    sessions = [json.loads(x) for x in (root(a.split) / "sessions.jsonl").read_text().splitlines() if x.strip()]
    fresh = [r for r in rows if r["reused_from"] is None and not r["failed"]]
    checks = {"complete": f"{len(CONDITIONS)} conditions x {len(qids)} questions = {len(rows)} records",
              "hits_reproduced": "every frame condition's recomputed primary and S-src hits equal hits_dev.json "
                                 "(checked before each answer)",
              "s1t_frames_equal_s1": all(r[("S1t", q)]["frame_digests"] == r[("S1", q)]["frame_digests"]
                                         for r in [{(x["condition"], x["qa_id"]): x for x in rows}] for q in qids),
              "controls_without_frames": all(x["n_frames"] == 0 for x in rows if x["condition"] in CONTROLS),
              "no_text_systems_without_text": all(x["text_words"] == 0 for x in rows if x["words_allowance"] == 0),
              "max_words": {c: max(x["text_words"] for x in rows if x["condition"] == c) for c in ("S0", "S1t", "t_only")},
              "tokens_logged": all(x["prompt_text_tokens"] is not None for x in fresh),
              "resume_no_new_calls": bool(sessions) and sessions[-1]["calls"] == 0 and sessions[-1]["status"] == "done"}
    tmp = Path(tempfile.mkdtemp())                   # integrity fixtures on a copy of the records
    try:
        for name, extra in (("repeated row", rows[0]), ("conflicting reply", {**rows[1], "condition": "X",
                                                                               "option": -2})):
            d = tmp / name.replace(" ", "_") / "runs" / "m3av" / "qa" / a.split
            d.mkdir(parents=True)
            shutil.copy(root(a.split) / "records.jsonl", d / "records.jsonl")
            with open(d / "records.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(extra) + "\n")
            here = Path.cwd()
            try:
                os.chdir(d.parents[3])
                load_records(a.split)
                checks[f"rejects {name}"] = False
            except SystemExit:
                checks[f"rejects {name}"] = True
            finally:
                os.chdir(here)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    per = {}
    for c in CONDITIONS:
        g = [x for x in rows if x["condition"] == c]
        new = [x["answer_seconds"] for x in g if x["reused_from"] is None]
        per[c] = {"new_calls": len(new), "reused": len(g) - len(new),
                  "answer_seconds_mean": round(float(np.mean(new)), 2) if new else None}
    est = sum((v["answer_seconds_mean"] or 0) * v["new_calls"] / len(qids) for v in per.values()) * 853 / 3600
    summary = {"questions": len(qids), "records": len(rows), "checks": checks, "per_condition": per,
               "test_answer_hours_estimate": round(est, 1),
               "estimate_note": "mean answer seconds and the reuse rate of these dev questions scaled to 853; "
                                "excludes model loading and pool loading"}
    if not all(v for k, v in checks.items() if k != "max_words"):
        raise SystemExit(f"dev check failed: {checks}")
    Path("reports/m3av").mkdir(parents=True, exist_ok=True)
    Path("reports/m3av/qa_dev_check.json").write_text(json.dumps({"summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("answer", "seal", "report", "check"))
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--limit", type=int, default=0, help="dev only: the first N questions in hash order (answer, report)")
    a = p.parse_args()
    {"answer": stage_answer, "seal": stage_seal, "report": stage_report, "check": stage_check}[a.action](a)


if __name__ == "__main__":
    main()
