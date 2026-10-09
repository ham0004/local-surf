"""Gate on Video-MMMU: do frames help the frozen answerer on lecture questions that need the screen?

Declared before running (2026-10-09). Data: Video-MMMU Perception track (277 OCR + 23 ASR
multiple-choice questions), DEV third only (video-level split in scripts/research/fetch_videommmu.py);
the TEST two thirds stay untouched. Transcripts: our Whisper large-v3-turbo ASR (declared protocol).
Answerer: frozen Qwen3-VL-2B, multiple-choice letter prompt (v1/v2), exact letter scoring, no judge.

Conditions (same question, same options):
    Q    question only                      (prior / guessing floor; chance ~10% with 10 options)
    T    full transcript (capped at 2,000 words)
    U4   T + 4 frames evenly spaced over the video
    C4   T + the 4 frames MobileCLIP-S2 matches best to the question, from a 2 s scan of the whole
         video, at least 4 s apart

Gate passes if U4 or C4 beats T with a 95% CI above zero. Budget: 400 answer calls, 1 GPU-hour.

    python scripts/v3_gate_videommmu.py answer
    python scripts/v3_gate_videommmu.py analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

DATA = Path("data/videommmu")
RUN = Path("runs/v3_gate_videommmu")
CONDITIONS = ("Q", "T", "U4", "C4")
MAX_WORDS = 2000
LIMITS = (400, 3600.0)


def questions(track: str = "Perception", split: str = "dev") -> list[dict]:
    rows = [json.loads(x) for x in (DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["qa_id"].startswith(f"vmmmu:{track}:") and r["experiment_split"] == split]
    return sorted(rows, key=lambda r: hashlib.sha256(r["qa_id"].encode()).hexdigest())


def transcript(video_id: str):
    from videoqa.transcript import load_transcript  # noqa: PLC0415

    segs, words = [], 0
    for s in load_transcript(DATA / "transcripts" / f"{video_id}.json", video_id).segments:
        words += len(s.text.split())
        if words > MAX_WORDS:
            break
        segs.append(s)
    return segs


def decode(video_id: str, times):
    from videoqa.frames import decode_at, probe  # noqa: PLC0415

    path = str(DATA / "videos" / f"{video_id}.mp4")
    dur = probe(path).duration_s
    return decode_at(path, [min(max(0.0, t), dur - 0.05) for t in times], max_side=640, video_id=video_id).frames


def duration(video_id: str) -> float:
    from videoqa.frames import probe  # noqa: PLC0415

    return probe(str(DATA / "videos" / f"{video_id}.mp4")).duration_s


def choose_c4(r, enc) -> list[float]:
    dur = duration(r["video_id"])
    cands = decode(r["video_id"], list(np.arange(0.5, dur, 2.0)))
    sims = enc.embed_images([f.image for f in cands]) @ enc.embed_texts([r["question"]])[0]
    chosen = []
    for i in np.argsort(-sims):
        t = cands[i].decoded_pts_s
        if all(abs(t - c) >= 4.0 for c in chosen):
            chosen.append(round(float(t), 3))
        if len(chosen) == 4:
            break
    return sorted(chosen)


def frames_for(r, cond: str, c4: dict):
    if cond in ("Q", "T"):
        return []
    if cond == "U4":
        dur = duration(r["video_id"])
        return decode(r["video_id"], list(np.linspace(0, dur, 6)[1:-1]))   # 4 interior, evenly spaced
    return decode(r["video_id"], c4[r["qa_id"]])


def request_key(r, cond, excerpt, frames, identity) -> str:
    from videoqa.answerer import AnswerRequest, build_prompt_text  # noqa: PLC0415

    prompt = build_prompt_text(AnswerRequest(r["question"], r["options"], excerpt, []))
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
    qs = [r for r in questions() if (DATA / "transcripts" / f"{r['video_id']}.json").exists()]
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
        excerpt = transcript(r["video_id"])
        jobs = [(c, [] if c == "Q" else excerpt, frames_for(r, c, c4)) for c in CONDITIONS]
        keys = [request_key(r, c, ex, fr, identity) for c, ex, fr in jobs]
        if not budget.reserve(sum(k not in cache for k in keys)):
            print(f"budget reached before question {n}; stopping cleanly")
            break
        for (c, ex, fr), k in zip(jobs, keys, strict=True):
            if k in cache:
                continue
            budget.check_one()
            t0 = time.perf_counter()
            ans, usage = answerer.answer(AnswerRequest(r["question"], r["options"], ex,
                                                       sorted(fr, key=lambda f: f.decoded_pts_s)))
            spent = time.perf_counter() - t0
            budget.charge(spent)
            rec = {"key": k, "qa_id": r["qa_id"], "video_id": r["video_id"], "condition": c,
                   "qa_type": r["evidence_type"].split(":")[-1], "option": ans.option_index,
                   "gold": r["gold_option_index"], "correct": ans.option_index == r["gold_option_index"],
                   "n_options": len(r["options"]), "frame_times": [f.decoded_pts_s for f in fr],
                   "transcript_words": sum(len(s.text.split()) for s in ex), "seconds": spent,
                   "visual_tokens": usage.visual_tokens, "text": ans.text[:200]}
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

    def boot(diffs, repeats=5000):
        v = np.array(list(diffs.values()), dtype=float)
        idx = np.random.default_rng(0).integers(0, len(v), (repeats, len(v)))
        return float(v.mean()), np.percentile(v[idx].mean(1), [2.5, 97.5]).tolist()

    out = {"questions": len(full), "note": "each video has one Perception question, so the bootstrap is by question = by video",
           "accuracy": {}, "minus_T": {}, "by_qa_type": {}}
    for c in CONDITIONS:
        out["accuracy"][c] = float(np.mean([d[c]["correct"] for d in full.values()]))
    for c in ("Q", "U4", "C4"):
        out["minus_T"][c] = dict(zip(("difference", "ci95"), boot({q: d[c]["correct"] - d["T"]["correct"] for q, d in full.items()}), strict=True))
    for t in sorted({d["T"]["qa_type"] for d in full.values()}):
        sub = {q: d for q, d in full.items() if d["T"]["qa_type"] == t}
        out["by_qa_type"][t] = {"n": len(sub), **{c: float(np.mean([d[c]["correct"] for d in sub.values()])) for c in CONDITIONS}}
    out["gate_passes"] = any(out["minus_T"][c]["ci95"][0] > 0 for c in ("U4", "C4"))
    Path("reports/v3_gate_videommmu").mkdir(parents=True, exist_ok=True)
    Path("reports/v3_gate_videommmu/summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
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
