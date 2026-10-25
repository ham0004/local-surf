"""Ceiling reference: the frozen answer model given 4 pictures from the HUMAN-marked answer seconds (CG-Bench).

Not a method (it uses the gold evidence location); it shows how far a perfect picture choice could take each
answer model. Same answer prompt as the frameworks. Conditions:
    H0     4 pictures spread over the human clue intervals, no transcript words
    H600   the same pictures + the framework's speech text (speech around the pictures + word/meaning search, 600 words)
Answer models: Qwen3-VL-2B, Qwen3-VL-4B, Nemotron-120B (API).
--oracle adds 4 shifted placements; the ORACLE ceiling counts a question when any of the 5 placements is right. Question sets: CG-Bench dev (82) and test (55).
Ledger runs/v3_ceiling/budget.json (1,000 calls / 4 GPU-hours), declared in research_log step 56.

    python scripts/v3_ceiling.py answer --config configs/gpu_12gb.yaml
    python scripts/v3_ceiling.py answer --config configs/gpu_16gb_4b.yaml
    python scripts/v3_ceiling.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN = Path("runs/v3_ceiling")
SETS = {"cgbench": "dev", "cgbench_test": "test"}


SHIFTS = (0.1, 0.3, 0.5, 0.7, 0.9)   # oracle ceiling: 5 placements of the 4 pictures inside the marked seconds


def shifted_times(r, shift: float) -> list[float]:
    """4 pictures over the union of the human clue intervals at positions (j + shift) / 4 of their total length.
    shift = 0.5 is the plain ceiling (evenly spread); the oracle tries all SHIFTS."""
    iv = [(x, y) for x, y in r["evidence_intervals_s"] if y >= x]
    total = sum(y - x for x, y in iv) or 1.0
    pts = []
    for j in range(4):
        pos = (j + shift) / 4 * total
        for x, y in iv:
            if pos <= (y - x):
                pts.append(x + pos)
                break
            pos -= (y - x)
    return pts


def stage_answer(a) -> None:
    import v3_baseline_cycle as C  # noqa: PLC0415
    import v3_gate_cgbench as G  # noqa: PLC0415

    from videoqa.answerer import AnswerRequest, make_answerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v2.teacher import answerer_identity  # noqa: PLC0415

    cfg = load_config(a.config)
    identity = answerer_identity(cfg)
    model = cfg["answerer"]["model_id"].split("/")[-1]
    RUN.mkdir(parents=True, exist_ok=True)
    out_path = RUN / "answers.jsonl"
    done = {(json.loads(x)["model"], json.loads(x)["qa_id"], json.loads(x)["condition"])
            for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()} if out_path.exists() else set()
    budget = CallBudget(RUN / a.ledger, 2000 if a.oracle else 1000, 8 * 3600.0)
    answerer = None
    shifts = SHIFTS if a.oracle else (0.5,)
    for ds, split in SETS.items():
        its = {it.qa.qa_id: it for it in C.items(ds)}
        for r in G.questions(split):
            for shift in shifts:
                frames = G.decode(r["video_id"], shifted_times(r, shift))
                cands = [types.SimpleNamespace(time_s=f.decoded_pts_s) for f in frames]
                pool = types.SimpleNamespace(question=r["question"], options=r["options"], qa_id=r["qa_id"], transcript=[])
                segs = its[r["qa_id"]].transcript.segments
                tag = "" if shift == 0.5 else f"_s{shift}"
                for cond, words in ((f"H0{tag}", 0), (f"H600{tag}", 600)):
                    if (model, r["qa_id"], cond) in done:
                        continue
                    if not budget.reserve(1):
                        print("budget reached; stopping cleanly")
                        return
                    ex = C.transcript_for("frames+hybrid", words, pool, segs, cands)
                    answerer = answerer or make_answerer(cfg)
                    t0 = time.perf_counter()
                    ans, _ = answerer.answer(AnswerRequest(r["question"], r["options"], ex, frames))
                    budget.charge(time.perf_counter() - t0)
                    rec = {"model": model, "dataset": ds, "set": split, "qa_id": r["qa_id"], "video_id": r["video_id"],
                           "condition": cond, "option": ans.option_index, "correct": ans.option_index == r["gold_option_index"],
                           "frame_times": [f.decoded_pts_s for f in frames], "words": sum(len(s.text.split()) for s in ex),
                           "seconds": time.perf_counter() - t0}
                    with open(out_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
        print(f"{model} {ds}: done; calls {budget.calls}", flush=True)
    _ = identity


def stage_report(a) -> None:
    rows = [json.loads(x) for x in (RUN / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    g = {}
    for r in rows:
        g.setdefault((r["model"], r["set"], r["condition"]), []).append(r["correct"])
    out = {f"{m} {s} {c}": {"accuracy": round(100 * float(np.mean(v)), 1), "n": len(v)} for (m, s, c), v in sorted(g.items())}
    # oracle: a question counts as answerable if any of the 5 placements is answered correctly
    by = {}
    for r in rows:
        base, _, sh = r["condition"].partition("_s")
        by.setdefault((r["model"], r["set"], base, r["qa_id"]), {})[sh or "0.5"] = r["correct"]
    orc = {}
    for (m, s, c, q), d in by.items():
        if len(d) == len(SHIFTS):
            orc.setdefault((m, s, c), []).append(any(d.values()))
    for (m, s, c), v in sorted(orc.items()):
        out[f"{m} {s} {c} ORACLE(any of 5 placements)"] = {"accuracy": round(100 * float(np.mean(v)), 1), "n": len(v)}
    Path("reports/v3_ceiling").mkdir(parents=True, exist_ok=True)
    Path("reports/v3_ceiling/summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("answer", "report"))
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    p.add_argument("--ledger", default="budget.json", help="budget_api.json = the step-63 API ledger")
    p.add_argument("--oracle", action="store_true", help="also the 4 shifted placements (oracle ceiling, step 66)")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"answer": stage_answer, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
