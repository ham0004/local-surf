"""Dataset loading in one simple local layout.

Every dataset (synthetic or converted from a public benchmark by an adapter)
is materialised as:

    <root>/qa.jsonl                     one QAItem per line (schemas.QAItem fields)
    <root>/videos/<video_id>.mp4        (or .mkv / .webm)
    <root>/transcripts/<video_id>.json  [{"start","end","text"}], or .srt / .vtt

Splits: if an item carries an official split ("train"/"dev"/"test"...), we
keep it; otherwise ("unassigned") we assign one by hashing the video id
(splits.py), BEFORE any transcript variants are generated.  Items whose video
or transcript is missing are reported, never silently dropped.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .schemas import QAItem, Transcript
from .splits import check_no_leakage, split_of
from .transcript import load_transcript

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
TRANSCRIPT_EXTS = (".json", ".srt", ".vtt")


@dataclasses.dataclass
class LoadedItem:
    qa: QAItem
    video_path: str
    transcript: Transcript


@dataclasses.dataclass
class LoadReport:
    loaded: int = 0
    missing_video: list[str] = dataclasses.field(default_factory=list)
    missing_transcript: list[str] = dataclasses.field(default_factory=list)


def _find(folder: Path, stem: str, exts: tuple[str, ...]) -> Path | None:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _qa_from_json(d: dict) -> QAItem:
    d = dict(d)
    d["evidence_intervals_s"] = [tuple(x) for x in d.get("evidence_intervals_s", [])]
    known = {f.name for f in dataclasses.fields(QAItem)}
    return QAItem(**{k: v for k, v in d.items() if k in known})


def load_local_dataset(root: str | Path, splits: tuple[str, ...] | None = None) -> tuple[list[LoadedItem], LoadReport]:
    """Load items, assign hashed splits where none is given, and filter by split."""
    root = Path(root)
    report = LoadReport()
    items: list[LoadedItem] = []
    transcripts: dict[str, Transcript] = {}
    for line in (root / "qa.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qa = _qa_from_json(json.loads(line))
        if qa.source_split in ("", "unknown", "unassigned"):
            qa = dataclasses.replace(qa, source_split=split_of(qa.video_id))
        if splits and qa.source_split not in splits:
            continue
        video = _find(root / "videos", qa.video_id, VIDEO_EXTS)
        if video is None:
            report.missing_video.append(qa.qa_id)
            continue
        if qa.video_id not in transcripts:
            tpath = _find(root / "transcripts", qa.video_id, TRANSCRIPT_EXTS)
            if tpath is None:
                report.missing_transcript.append(qa.qa_id)
                continue
            transcripts[qa.video_id] = load_transcript(tpath, qa.video_id)
        items.append(LoadedItem(qa, str(video), transcripts[qa.video_id]))
    report.loaded = len(items)
    # Defensive: official splits could still leak a video across splits.
    by_split: dict[str, list[QAItem]] = {}
    for it in items:
        by_split.setdefault(it.qa.source_split, []).append(it.qa)
    check_no_leakage(by_split)
    return items, report
