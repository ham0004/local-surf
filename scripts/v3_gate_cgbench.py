"""Gate on CG-Bench (long videos): do frames help, and how much could a perfect selector gain?

Declared before running (2026-10-09). Data: CG-Bench, the 51 English-subtitled videos we hold, DEV
split only (8 videos, 82 questions; split in scripts/research/prepare_cgbench.py). Answerer: frozen
Qwen3-VL-2B, multiple-choice letter prompt, exact letter scoring, no judge.

Transcript (T): the v1/v2 retrieval: BM25 over 20 s transcript units with the question and options as
the query, top 4 windows, packed into an excerpt of at most 300 words.

Conditions (same question and options):
    Q    question only
    T    retrieved transcript excerpt
    U4   T + 4 frames evenly spaced over the whole video
    C4   T + MobileCLIP-S2 top-4 frames from a 4 s scan of the whole video, >= 8 s apart
    E4   T + 4 frames evenly spaced inside the HUMAN clue intervals  (analysis only: uses gold
         evidence locations, so it is an upper reference for frame selection, not a method)

Reads: frames help if U4 or C4 beats T (CI > 0); selection headroom = E4 - C4.
Budget: 420 answer calls, 1 GPU-hour.

    python scripts/v3_gate_cgbench.py answer
    python scripts/v3_gate_cgbench.py analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

DATA = Path("data/cgbench")
RUN = Path("runs/v3_gate_cgbench")
CONDITIONS = ("Q", "T", "U4", "C4", "E4")
EXCERPT_WORDS = 300
LIMITS = (420, 3600.0)


def questions(split: str = "dev") -> list[dict]:
    rows = [json.loads(x) for x in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted((r for r in rows if r["experiment_split"] == split),
                  key=lambda r: hashlib.sha256(r["qa_id"].encode()).hexdigest())


def excerpt(r):
    """v1/v2 retrieval: BM25 windows -> packed excerpt (question + options as the query)."""
    from videoqa.frames import probe  # noqa: PLC0415
    from videoqa.packing import pack_excerpt  # noqa: PLC0415
    from videoqa.retrieval import bm25_rank, build_windows  # noqa: PLC0415
    from videoqa.transcript import build_units, load_transcript  # noqa: PLC0415

    segs = load_transcript(DATA / "transcripts" / f"{r['video_id']}.srt", r["video_id"]).segments
    if not segs:
        return []
    duration = probe(str(DATA / "videos" / f"{r['video_id']}.mp4")).duration_s
    units = build_units(segs, 20.0, 5.0)
    query = r["question"] + " " + " ".join(r["options"])
    windows = build_windows(bm25_rank(query, units, 64), units, 4, 1, duration)
    return pack_excerpt(r["question"], segs, windows, {u.id: u.segment_ids for u in units}, max_words=EXCERPT_WORDS)


def decode(video_id: str, times):
    from videoqa.frames import decode_at, probe  # noqa: PLC0415

    path = str(DATA / "videos" / f"{video_id}.mp4")
    dur = probe(path).duration_s
    return decode_at(path, sorted(min(max(0.0, t), dur - 0.05) for t in times), max_side=640, video_id=video_id).frames


def choose_c4(r, enc) -> list[float]:
    from videoqa.frames import probe  # noqa: PLC0415

    dur = probe(str(DATA / "videos" / f"{r['video_id']}.mp4")).duration_s
    cands = decode(r["video_id"], list(np.arange(1.0, dur, 4.0)))
    sims = enc.embed_images([f.image for f in cands]) @ enc.embed_texts([r["question"]])[0]
    chosen = []
    for i in np.argsort(-sims):
        t = cands[i].decoded_pts_s
        if all(abs(t - c) >= 8.0 for c in chosen):
            chosen.append(round(float(t), 3))
        if len(chosen) == 4:
            break
    return sorted(chosen)


def times_for(r, cond: str, c4: dict) -> list[float]:
    from videoqa.frames import probe  # noqa: PLC0415

    if cond in ("Q", "T"):
        return []
    if cond == "U4":
        dur = probe(str(DATA / "videos" / f"{r['video_id']}.mp4")).duration_s
        return list(np.linspace(0, dur, 6)[1:-1])
    if cond == "C4":
        return c4[r["qa_id"]]
    # E4: spread 4 frames over the union of the human clue intervals, proportionally to their length.
    iv = [(a, b) for a, b in r["evidence_intervals_s"] if b >= a]
    total = sum(b - a for a, b in iv) or 1.0
    pts = []
    for k in range(4):
        x = (k + 0.5) / 4 * total
        for a, b in iv:
            if x <= (b - a):
                pts.append(a + x)
                break
            x -= (b - a)
    return pts


def request_key(r, cond, ex, frames, identity) -> str:
    from videoqa.answerer import AnswerRequest, build_prompt_text  # noqa: PLC0415

    prompt = build_prompt_text(AnswerRequest(r["question"], r["options"], ex, []))
    payload = {"identity": identity, "prompt": prompt,
               "frames": [[f.digest, f"Frame at {f.decoded_pts_s:.2f}s:"] for f in sorted(frames, key=lambda f: f.decoded_pts_s)]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_answer(a) -> None:
    from videoqa.answerer import AnswerRequest, make_answerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415
    from videoqa.v2.teacher import answerer_identity  # noqa: PLC0415

    RUN.mkdir(parents=True, exist_ok=True)
    qs = questions()
    c4_path = RUN / "c4_times.json"
    c4 = json.loads(c4_path.read_text()) if c4_path.exists() else {}
    if any(r["qa_id"] not in c4 for r in qs):
        enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
        for r in qs:
            if r["qa_id"] not in c4:
                c4[r["qa_id"]] = choose_c4(r, enc)
                c4_path.write_text(json.dumps(c4, indent=1), encoding="utf-8")
        del enc
    cfg = load_config(a.config)
    identity = answerer_identity(cfg)
    cache_path = RUN / "answers.jsonl"
    cache = {json.loads(x)["key"]: json.loads(x) for x in cache_path.read_text().splitlines()} if cache_path.exists() else {}
    budget = CallBudget(RUN / "budget_answer.json", *LIMITS)
    answerer = make_answerer(cfg)
    for n, r in enumerate(qs):
        ex = excerpt(r)
        jobs = [(c, [] if c == "Q" else ex, decode(r["video_id"], times_for(r, c, c4)) if c not in ("Q", "T") else [])
                for c in CONDITIONS]
        keys = [request_key(r, c, e, f, identity) for c, e, f in jobs]
        if not budget.reserve(sum(k not in cache for k in keys)):
            print(f"budget reached before question {n}; stopping cleanly")
            break
        for (c, e, f), k in zip(jobs, keys, strict=True):
            if k in cache:
                continue
            budget.check_one()
            t0 = time.perf_counter()
            ans, usage = answerer.answer(AnswerRequest(r["question"], r["options"], e, f))
            spent = time.perf_counter() - t0
            budget.charge(spent)
            rec = {"key": k, "qa_id": r["qa_id"], "video_id": r["video_id"], "condition": c,
                   "category": r["evidence_type"], "option": ans.option_index, "gold": r["gold_option_index"],
                   "correct": ans.option_index == r["gold_option_index"], "n_options": len(r["options"]),
                   "frame_times": [x.decoded_pts_s for x in f], "transcript_words": sum(len(s.text.split()) for s in e),
                   "seconds": spent, "visual_tokens": usage.visual_tokens, "text": ans.text[:200]}
            cache[k] = rec
            with open(cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(qs)} calls={budget.calls} seconds={budget.seconds:.0f}", flush=True)
    print(json.dumps({"questions": len(qs), "budget": {"calls": budget.calls, "seconds": round(budget.seconds)}}))


def stage_analyze(a) -> None:
    rows = [json.loads(x) for x in (RUN / "answers.jsonl").read_text().splitlines() if x.strip()]
    by = {}
    for x in rows:
        by.setdefault(x["qa_id"], {})[x["condition"]] = x
    full = {q: d for q, d in by.items() if all(c in d for c in CONDITIONS)}
    vids = {q: d["T"]["video_id"] for q, d in full.items()}

    def boot(diffs, repeats=5000):
        g = {}
        for q, v in diffs.items():
            g.setdefault(vids[q], []).append(v)
        s = np.array([sum(v) for v in g.values()])
        c = np.array([len(v) for v in g.values()])
        i = np.random.default_rng(0).integers(0, len(s), (repeats, len(s)))
        return float(s.sum() / c.sum()), np.percentile(s[i].sum(1) / c[i].sum(1), [2.5, 97.5]).tolist()

    out = {"questions": len(full), "videos": len(set(vids.values())),
           "note": "video-clustered bootstrap over only 8 dev videos: intervals are wide",
           "accuracy": {c: float(np.mean([d[c]["correct"] for d in full.values()])) for c in CONDITIONS},
           "differences": {}}
    for x, y in (("U4", "T"), ("C4", "T"), ("E4", "T"), ("E4", "C4"), ("T", "Q")):
        out["differences"][f"{x}-{y}"] = dict(zip(("difference", "ci95"), boot({q: d[x]["correct"] - d[y]["correct"]
                                                                                for q, d in full.items()}), strict=True))
    out["frames_help"] = any(out["differences"][k]["ci95"][0] > 0 for k in ("U4-T", "C4-T"))
    Path("reports/v3_gate_cgbench").mkdir(parents=True, exist_ok=True)
    Path("reports/v3_gate_cgbench/summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("answer", "analyze"))
    p.add_argument("--config", default="configs/gpu_12gb.yaml")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"answer": stage_answer, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
