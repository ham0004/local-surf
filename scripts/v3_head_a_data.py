"""Head A training and evaluation data: question + timed speech lines + where the visual evidence is.

Head A (docs/v3/heads_design.md) works on text and times only, so its data needs subtitles and an
evidence position, not the video. Sources:

  CG-Bench   human clue intervals; English subtitles from the release's subtitles.zip (also for videos
             whose video file we do not hold). Our declared video split is kept: train-split videos ->
             train, dev-split videos -> dev; test-split videos are never read.
  EduVidQA   synthetic_train questions whose text starts with "At m:ss," (the question timestamp, a weak
             evidence position; the paper reports ~35 s average error). The prefix is removed from the
             question, otherwise the model could read the answer position from the text. Transcript: the
             saved English caption track, else our Whisper transcript of the audio. Train only; the 269
             real-world questions are not used.

Output (one JSON record per question):
    data/head_a/train.jsonl, data/head_a/dev.jsonl, data/head_a/manifest.json
    {qa_id, source, video_id, question, options, lines: [[start, end, text], ...],
     evidence: [[a, b], ...], weak: bool}

    python scripts/v3_head_a_data.py
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.transcript import load_transcript, parse_srt_or_vtt  # noqa: E402
from videoqa.v3.language import is_english_transcript  # noqa: E402

OUT = Path("data/head_a")
CG_RAW = Path("data/cgbench_raw")
EDU = Path("data/eduvidqa")
TIMESTAMP = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")
TIME_PHRASE = re.compile(r"\s*[(\[]?\b(?:at|around|near|from|by|after|before|of)?\s*(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?"
                         r"(?![\d:])(?:\s*(?:-|to|and)\s*\d{1,2}:\d{2}(?::\d{2})?)?[)\]]?", re.IGNORECASE)
EDU_WEAK_HALF_WIDTH_S = 20.0


def cg_split(video_uid: str) -> str:
    """The declared CG-Bench split (scripts/research/prepare_cgbench.py)."""
    h = int(hashlib.sha256(f"cgbench:{video_uid}".encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else ("dev" if h < 8 else "test")


def strip_timestamp(question: str) -> tuple[str, float | None]:
    """Remove the question's time reference and return it in seconds (the first time mentioned).

    'At 3:54, can you explain X?'                      -> ('Can you explain X?', 234.0)
    'Referring to the slide at 1:46, the instructor ...' -> ('The instructor ...', 106.0)
    'Why does the bound at 12:30 hold?'                -> ('Why does the bound hold?', 750.0)
    If the first time is in the opening clause (before the first comma) the whole clause is dropped,
    otherwise only the time phrase. No time -> (question, None).
    """
    m = TIMESTAMP.search(question)
    if not m:
        return question.strip(), None
    a, b, c = m.groups()
    t = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    comma = question.find(",")
    rest = question[comma + 1:] if 0 <= m.start() < comma else question
    rest = re.sub(r"\s{2,}", " ", TIME_PHRASE.sub("", rest)).strip(" ,")
    rest = re.sub(r"\s+([,.?!])", r"\1", rest)
    return (rest[:1].upper() + rest[1:]) if rest else rest, float(t)


def lines_of(segments) -> list[list]:
    return [[round(s.start_s, 2), round(s.end_s, 2), s.text] for s in segments]


def cgbench() -> dict[str, list[dict]]:
    z = zipfile.ZipFile(CG_RAW / "subtitles.zip")
    subs = {}
    for name in z.namelist():
        uid = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name.endswith("/") or cg_split(uid) == "test":
            continue
        tr = parse_srt_or_vtt(z.read(name).decode("utf-8", "ignore"), uid)
        if tr.segments and is_english_transcript(tr.segments):
            subs[uid] = lines_of(tr.segments)
    out = collections.defaultdict(list)
    for q in json.loads((CG_RAW / "cgbench.json").read_text(encoding="utf-8")):
        uid = q["video_uid"]
        if uid not in subs:
            continue
        clues = q["clue_intervals"] if isinstance(q["clue_intervals"], list) else ast.literal_eval(q["clue_intervals"])
        choices = q["choices"] if isinstance(q["choices"], list) else ast.literal_eval(q["choices"])
        if not clues:
            continue
        out[cg_split(uid)].append({"qa_id": f"cgbench:{q['qid']}", "source": "cgbench", "video_id": uid,
                                   "question": q["question"].strip(), "options": list(choices),
                                   "lines": subs[uid], "evidence": [[float(a), float(b)] for a, b in clues],
                                   "duration_s": float(q["duration"]), "weak": False})
    return out


def eduvidqa_transcript(vid: str):
    for path in (EDU / "transcripts" / f"{vid}.json", EDU / "transcripts_asr" / f"{vid}.json"):
        if path.exists():
            segs = load_transcript(path, vid).segments
            if segs and is_english_transcript(segs):
                return segs, path.parent.name
    return None, None


def eduvidqa() -> tuple[list[dict], collections.Counter]:
    rows, why = [], collections.Counter()
    cache = {}
    for line in (EDU / "qa.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["source_dataset"] != "EduVidQA/synthetic_train":
            continue
        question, t = strip_timestamp(r["question"])
        if t is None:
            why["no timestamp"] += 1
            continue
        if TIMESTAMP.search(question) or len(question.split()) < 4:
            why["time not removable"] += 1
            continue
        if r["video_id"] not in cache:
            cache[r["video_id"]] = eduvidqa_transcript(r["video_id"])
        segs, origin = cache[r["video_id"]]
        if segs is None:
            why["no English transcript"] += 1
            continue
        if t > segs[-1].end_s + 60:
            why["timestamp past transcript"] += 1
            continue
        why["kept"] += 1
        rows.append({"qa_id": r["qa_id"], "source": f"eduvidqa:{origin}", "video_id": r["video_id"],
                     "question": question, "options": None, "lines": lines_of(segs),
                     "evidence": [[max(0.0, t - EDU_WEAK_HALF_WIDTH_S), t + EDU_WEAK_HALF_WIDTH_S]],
                     "duration_s": float(segs[-1].end_s), "weak": True})
    return rows, why


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cg = cgbench()
    edu, why = eduvidqa()
    split = {"train": cg["train"] + edu, "dev": cg["dev"]}
    for name, rows in split.items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        name: {"questions": len(rows), "videos": len({r["video_id"] for r in rows}),
               "by_source": dict(collections.Counter(r["source"] for r in rows))}
        for name, rows in split.items()}
    manifest["eduvidqa_filter"] = dict(why)
    manifest["notes"] = ("CG-Bench test-split videos are excluded; EduVidQA weak evidence = question timestamp "
                         f"+/- {EDU_WEAK_HALF_WIDTH_S:.0f} s with the time reference removed from the question")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
