"""Transcript processing: parsing, retrieval units, and reliability features.

Order of operations in the pipeline matters:

    original segments  --damage.py-->  observed segments  --build_units-->  units

Damage is applied to the ORIGINAL fine-grained segments (so "delete one
subtitle line" means exactly that), and retrieval units are built afterwards
from whatever text survived.  This mirrors deployment, where the system only
ever sees the imperfect transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import Transcript, TranscriptSegment

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches "00:01:02,345" (SRT) and "00:01:02.345" / "01:02.345" (VTT).
_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE_RE = re.compile(rf"{_TS}\s*-->\s*{_TS}")


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    # Pad milliseconds on the right: "5" means 500 ms, not 5 ms.
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_or_vtt(text: str, video_id: str, source: str = "subtitle_file") -> Transcript:
    """Parse SRT or WebVTT text into a :class:`Transcript`.

    Cue numbers, the WEBVTT header, and inline tags such as ``<i>`` are ignored.
    Empty cues are dropped because they carry no retrievable text.
    """
    segments: list[TranscriptSegment] = []
    # Cues are separated by blank lines in both formats.
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            match = _CUE_RE.search(line)
            if not match:
                continue
            g = match.groups()
            start = _to_seconds(g[0], g[1], g[2], g[3])
            end = _to_seconds(g[4], g[5], g[6], g[7])
            body = " ".join(lines[i + 1 :])
            body = re.sub(r"<[^>]+>", "", body).strip()  # strip <i>, <c.color> ...
            if body and end >= start:
                segments.append(
                    TranscriptSegment(id=f"s{len(segments):05d}", start_s=start, end_s=end, text=body)
                )
            break
    return Transcript(video_id=video_id, segments=segments, source=source)


def parse_json_segments(items: list[dict], video_id: str, source: str = "json") -> Transcript:
    """Parse ``[{"start": .., "end": .., "text": .., "confidence"?: ..}, ...]``
    (the shape produced by faster-whisper and most dataset subtitle dumps)."""
    segments = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                id=f"s{len(segments):05d}",
                start_s=float(item["start"]),
                end_s=float(item["end"]),
                text=text,
                confidence=item.get("confidence"),
            )
        )
    return Transcript(video_id=video_id, segments=segments, source=source)


def load_transcript(path: str | Path, video_id: str) -> Transcript:
    """Load .srt / .vtt / .json by extension, recording the file as provenance."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        return parse_json_segments(json.loads(raw), video_id, source=f"json:{path.name}")
    return parse_srt_or_vtt(raw, video_id, source=f"subtitle:{path.name}")


# ---------------------------------------------------------------------------
# Retrieval units
# ---------------------------------------------------------------------------


@dataclass
class TranscriptUnit:
    """A retrieval unit: consecutive segments merged to roughly ``unit_seconds``.

    ``segment_ids`` keeps provenance so any retrieved unit can be traced back to
    exact original subtitle lines (and so evidence citations stay verbatim).
    """

    id: str
    segment_ids: list[str]
    start_s: float
    end_s: float
    text: str


def build_units(segments: list[TranscriptSegment], unit_seconds: float = 20.0,
                min_unit_seconds: float = 5.0) -> list[TranscriptUnit]:
    """Greedily merge consecutive segments into units of about ``unit_seconds``.

    A unit is closed when adding the next segment would exceed ``unit_seconds``
    (as long as the unit already spans ``min_unit_seconds``), or when there is a
    silence gap longer than ``unit_seconds`` - a long gap usually means a topic
    or slide change, so we do not bridge it.
    """
    units: list[TranscriptUnit] = []
    current: list[TranscriptSegment] = []

    def flush() -> None:
        if not current:
            return
        units.append(
            TranscriptUnit(
                id=f"u{len(units):04d}",
                segment_ids=[s.id for s in current],
                start_s=current[0].start_s,
                end_s=current[-1].end_s,
                text=" ".join(s.text for s in current),
            )
        )
        current.clear()

    for seg in sorted(segments, key=lambda s: s.start_s):
        if current:
            span_if_added = seg.end_s - current[0].start_s
            gap = seg.start_s - current[-1].end_s
            long_enough = current[-1].end_s - current[0].start_s >= min_unit_seconds
            if (span_if_added > unit_seconds and long_enough) or gap > unit_seconds:
                flush()
        current.append(seg)
    flush()
    return units


# ---------------------------------------------------------------------------
# Reliability features (observable at inference, no labels involved)
# ---------------------------------------------------------------------------


@dataclass
class GapInfo:
    """A stretch of video with no speech.  Silent stretches are where visual-only
    facts (a slide nobody reads aloud) tend to hide."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def find_gaps(segments: list[TranscriptSegment], video_duration_s: float,
              min_gap_s: float = 3.0) -> list[GapInfo]:
    """Return silent stretches of at least ``min_gap_s`` seconds, including the
    video head and tail.  Works on the *observed* (possibly damaged) transcript,
    so a deleted subtitle line shows up here as a new gap."""
    gaps: list[GapInfo] = []
    cursor = 0.0
    for seg in sorted(segments, key=lambda s: s.start_s):
        if seg.start_s - cursor >= min_gap_s:
            gaps.append(GapInfo(cursor, seg.start_s))
        cursor = max(cursor, seg.end_s)
    if video_duration_s - cursor >= min_gap_s:
        gaps.append(GapInfo(cursor, video_duration_s))
    return gaps


def speech_coverage_near(segments: list[TranscriptSegment], time_s: float,
                         radius_s: float = 10.0) -> float:
    """Fraction of ``[time_s - radius, time_s + radius]`` covered by speech.

    Low coverage near a candidate time is an observable hint that the transcript
    may be missing something there - the kind of signal the controller must learn
    to use *only when it matters for the question*.
    """
    lo, hi = time_s - radius_s, time_s + radius_s
    covered = 0.0
    for seg in segments:
        overlap = min(hi, seg.end_s) - max(lo, seg.start_s)
        if overlap > 0:
            covered += overlap
    return min(1.0, covered / (hi - lo))


def unintelligible_fraction(segments: list[TranscriptSegment], mask_token: str = "[inaudible]") -> float:
    """Share of words that are unintelligible markers (ASR/garbled output)."""
    total = masked = 0
    for seg in segments:
        words = seg.text.split()
        total += len(words)
        masked += sum(1 for w in words if w == mask_token)
    return masked / total if total else 0.0
