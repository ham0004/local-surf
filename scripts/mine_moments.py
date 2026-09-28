"""Mine lecture moments where speech and board/slide carry the same fact.

For each lecture (scripts/fetch_lectures.py layout) we cut the talk into
fixed WINDOW_S windows. Each window gets its speech (captions) and two
frames (middle and end, because a board fills up as the lecturer writes). A
larger open VLM (the GENERATOR, deliberately not the frozen 2B answerer)
labels each window as JSON:

    relation: both | speech_only | visual_only | conflict | none
    question + 4 options + answer index, with speech and visual evidence.

Nothing here is ground truth: every item is marked synthetic, keeps its raw
generator output, and is checked by scripts/verify_moments.py and by a
small human audit (timestamps included so a person can check the moment).

    python scripts/mine_moments.py --lectures data/mit_lectures --out runs/mined/pilot --limit-windows 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.frames import decode_at  # noqa: E402
from videoqa.transcript import load_transcript  # noqa: E402

WINDOW_S = 30.0
GENERATOR = "Qwen/Qwen3-VL-4B-Instruct"
GENERATOR_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
RELATIONS = ("both", "speech_only", "visual_only", "conflict", "none")

PROMPT = """You are building a quiz from a university lecture. Below are two frames from a {w:.0f}-second
window of the lecture (at {t0:.0f}s and {t1:.0f}s) and what the lecturer SAID in that window
(automatic captions, may contain recognition errors).

SPEECH: "{speech}"

Find ONE specific, checkable fact from this window (a number, formula, matrix entry, definition,
named result, or step of a calculation). Then say where it appears:
- "both": the lecturer says it AND it is written/shown on the board or slide
- "speech_only": said, but not visible in the frames
- "visual_only": visible on the board/slide, but not said
- "conflict": the speech and the board/slide give DIFFERENT values for the same thing
- "none": no specific fact in this window

"speech_evidence" must be copied word for word from SPEECH (never from the board).
Write a multiple-choice question about that fact with 4 options (one correct, three plausible but wrong,
same type and length). The question must not contain the answer. Do not ask about the lecturer's
appearance or the camera.

Reply with JSON only:
{{"relation": "...", "fact": "...", "speech_evidence": "exact words from SPEECH or null",
"visual_evidence": "what is written on the board/slide or null", "question": "...",
"options": ["...", "...", "...", "..."], "answer_index": 0}}"""


def clean_captions(transcript):
    """YouTube ASR captions are "rolling": each cue repeats the previous line
    and adds new words, and a ~10 ms cue re-shows the settled line. Drop the
    10 ms cues, then remove the longest word overlap between the end of the
    previous kept text and the start of each cue, so speech is not doubled."""
    out, prev = [], []
    for s in transcript.segments:
        if s.end_s - s.start_s < 0.05:
            continue
        words = s.text.split()
        k = next((k for k in range(min(len(prev), len(words)), 0, -1) if prev[-k:] == words[:k]), 0)
        new = words[k:]
        if new:
            out.append((s.start_s, s.end_s, " ".join(new)))
            prev = (prev + new)[-40:]
    return out


def windows(lines, duration: float):
    t = 0.0
    while t + WINDOW_S <= duration:
        speech = " ".join(x for a, b, x in lines if a < t + WINDOW_S and b > t)
        yield t, t + WINDOW_S, speech
        t += WINDOW_S


class Generator:
    def __init__(self, model_id: str = GENERATOR, revision: str = GENERATOR_REV, cache_dir: str = "cache/hf/hub"):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, dtype=torch.bfloat16, device_map="cuda").eval()
        self.fingerprint = {"model_id": model_id, "revision": revision, "decoding": "greedy", "max_new_tokens": 400}

    def __call__(self, images, prompt: str) -> str:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                               add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


BOILERPLATE = re.compile(r"creative commons license|opencourseware continue|ocw\.mit\.edu", re.I)

CHECK = """{context}
QUESTION: {q}
A. {o0}
B. {o1}
C. {o2}
D. {o3}
Answer with the letter only. If the information above is not enough, answer X."""


def check_answer(gen, images, context: str, item: dict) -> int | None:
    """Ask the generator the question with ONE evidence source. The label
    'both' is then measured (speech-only and frame-only both correct), not
    taken from the generator's own claim."""
    o = [str(x) for x in item["options"]]
    text = gen(images, CHECK.format(context=context, q=item["question"], o0=o[0], o1=o[1], o2=o[2], o3=o[3]))
    m = re.search(r"\b([ABCDX])\b", text.strip().upper())
    return None if not m or m.group(1) == "X" else "ABCD".index(m.group(1))


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ok = (d.get("relation") in RELATIONS and isinstance(d.get("options"), list) and len(d["options"]) == 4
          and isinstance(d.get("answer_index"), int) and 0 <= d["answer_index"] < 4 and d.get("question"))
    return d if ok or d.get("relation") == "none" else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lectures", default="data/mit_lectures")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-windows", type=int, default=0, help="per lecture, evenly spaced (0 = all)")
    ap.add_argument("--max-side", type=int, default=896, help="generator frame size (board text must be legible)")
    a = ap.parse_args()
    root, out = Path(a.lectures), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "lectures.json").read_text(encoding="utf-8"))
    done = set()
    raw_p = out / "raw.jsonl"
    if raw_p.exists():                                   # resume
        done = {(r["video_id"], r["t0"]) for r in map(json.loads, raw_p.read_text(encoding="utf-8").splitlines())}
    gen, t_start, n = Generator(), time.time(), 0
    (out / "generator.json").write_text(json.dumps(gen.fingerprint, indent=1), encoding="utf-8")
    from videoqa.frames import probe  # noqa: PLC0415

    for vid in sorted(meta):
        video = root / "videos" / f"{vid}.mp4"
        lines = clean_captions(load_transcript(root / "transcripts" / f"{vid}.vtt", vid))
        wins = list(windows(lines, probe(video).duration_s))
        if a.limit_windows and len(wins) > a.limit_windows:
            step = len(wins) / a.limit_windows
            wins = [wins[int(i * step)] for i in range(a.limit_windows)]
        for t0, t1, speech in wins:
            if (vid, t0) in done or BOILERPLATE.search(speech):
                continue
            frames = decode_at(video, [t0 + WINDOW_S / 2, t1 - 0.5], max_side=a.max_side, video_id=vid).frames
            text = gen([f.image for f in frames], PROMPT.format(w=WINDOW_S, t0=frames[0].decoded_pts_s,
                                                                t1=frames[-1].decoded_pts_s,
                                                                speech=speech[:1500] or "(silence)"))
            parsed = parse_json(text)
            checks = {}
            if parsed and parsed["relation"] != "none":
                checks["speech_only"] = check_answer(gen, [], f'The lecturer said: "{speech[:1500]}"', parsed)
                checks["frame_only"] = check_answer(gen, [frames[-1].image], "Look at the board/slide in the image.",
                                                    parsed)
            rec = {"video_id": vid, "t0": t0, "t1": t1, "frame_pts": [f.decoded_pts_s for f in frames],
                   "speech": speech, "raw": text, "parsed": parsed, "generator_checks": checks}
            with open(raw_p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            rel = parsed["relation"] if parsed else "PARSE_FAIL"
            ok = {k: v == parsed["answer_index"] for k, v in checks.items()} if parsed else {}
            print(f"{vid} {t0:6.0f}s claimed={rel:12s} measured={ok} {(time.time() - t_start) / n:.1f}s/window",
                  flush=True)


if __name__ == "__main__":
    main()
