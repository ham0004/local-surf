"""EduVidQA adapter, step 1: questions + YouTube transcripts (no video).

Reads the three CSVs from github.com/sourjyadip/eduvidqa-emnlp25 (data/) and
writes our local layout (datasets.py):

    <out>/qa.jsonl                     one QAItem per question
    <out>/transcripts/<video_id>.json  [{"start","end","text"}]
    <out>/transcript_meta.json         per video: language, auto-generated?, error
    <out>/manifest.json

Splits: synthetic_train -> official "train"; synthetic_test and
real_world_test -> official "test" (kept locked: experiment_split "test").
Our dev split is hashed by video INSIDE official train by datasets.py
(experiment_split left "unassigned" for train rows).

The question's "<timestamp>" placeholder is filled with the CSV timestamp,
as a student would see it. That time is also stored as a 40 s evidence
interval [t-30, t+10] with evidence_type "inferred": the student asks about
what was just said. This is an automatic proposal, audited later, not a
human span.

    python scripts/research/fetch_eduvidqa.py --raw data/eduvidqa_raw --out data/eduvidqa
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from videoqa.schemas import QAItem  # noqa: E402

LICENSE = "EduVidQA questions (MIT repo, sourjyadip/eduvidqa-emnlp25); videos/transcripts: YouTube (NPTEL), research use"
FILES = {"synthetic_train": "train", "synthetic_test": "test", "real_world_test": "test"}


def parse_ts(ts: str) -> float:
    parts = [float(p) for p in ts.strip().split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def load_rows(raw: Path) -> list[QAItem]:
    items: list[QAItem] = []
    for stem, official in FILES.items():
        for i, r in enumerate(csv.DictReader(open(raw / f"{stem}.csv", encoding="utf-8"))):
            vid = r.get("vid_id") or r.get("id")
            q = r.get("final_question") or r.get("question")
            a = r.get("final_answer") or r.get("answer")
            ts = r.get("timestamp", "")
            t = parse_ts(ts) if ts else None
            q = q.replace("<timestamp>", ts) if ts else q
            items.append(QAItem(
                qa_id=f"{stem}:{vid}:{i}", video_id=vid, question=q, gold_answer=a,
                evidence_intervals_s=[(max(0.0, t - 30.0), t + 10.0)] if t is not None else [],
                source_dataset=f"EduVidQA/{stem}", source_split="unassigned" if official == "train" else "test",
                official_split=official, experiment_split="unassigned" if official == "train" else "test",
                provenance_and_license=LICENSE, evidence_type="inferred"))
    return items


def fetch(video_id: str):
    from youtube_transcript_api import YouTubeTranscriptApi  # noqa: PLC0415

    api = YouTubeTranscriptApi()
    tl = api.list(video_id)
    try:
        tr = tl.find_manually_created_transcript(["en"])
    except Exception:  # noqa: BLE001 - fall back to YouTube's ASR track
        tr = tl.find_generated_transcript(["en"])
    data = tr.fetch().to_raw_data()
    segs = [{"start": d["start"], "end": d["start"] + d["duration"], "text": d["text"].replace("\n", " ")}
            for d in data if d["text"].strip()]
    return segs, {"language": tr.language_code, "auto_generated": tr.is_generated, "segments": len(segs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/eduvidqa_raw")
    ap.add_argument("--out", default="data/eduvidqa")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    items = load_rows(raw)

    # A video must not appear in both official train and test.
    by_split: dict[str, set[str]] = {}
    for it in items:
        by_split.setdefault(it.official_split, set()).add(it.video_id)
    overlap = sorted(by_split.get("train", set()) & by_split.get("test", set()))

    meta_p = out / "transcript_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    videos = sorted({it.video_id for it in items}, key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    for n, vid in enumerate(videos):
        if (out / "transcripts" / f"{vid}.json").exists():
            continue
        try:
            segs, m = fetch(vid)
            (out / "transcripts" / f"{vid}.json").write_text(json.dumps(segs), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - recorded, never silently dropped
            m = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        meta[vid] = m
        meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[{n + 1}/{len(videos)}] {vid} {m}", flush=True)
        time.sleep(a.sleep)

    with open(out / "qa.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(dataclasses.asdict(it)) + "\n")
    ok = [v for v in videos if "error" not in meta.get(v, {"error": 1})]
    manifest = {
        "dataset": "EduVidQA", "questions": len(items),
        "questions_by_file": {s: sum(it.source_dataset.endswith(s) for it in items) for s in FILES},
        "videos": len(videos), "videos_with_transcript": len(ok),
        "auto_generated_transcripts": sum(bool(meta[v].get("auto_generated")) for v in ok),
        "train_test_video_overlap": overlap, "license": LICENSE,
        "evidence_interval_rule": "[t-30, t+10] around the question timestamp, evidence_type=inferred",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
