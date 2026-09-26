"""Adapter for LongVideoBench (longvideobench/LongVideoBench on Hugging Face).

Dataset facts verified in this session (see docs/DATA_FEASIBILITY.md):
  * ``lvb_val.json`` has 1,337 questions over 753 videos, WITH gold answers
    (``correct_choice``). The Hub "test" split has no gold answers, so it is
    unusable for our label generation / scoring and is not read here.
  * Subtitles ship as ``subtitles/<id>_en.json``: a list of
    ``{"start": "HH:MM:SS.mmm", "end": "...", "line": "..."}`` — auto-generated
    and often noisy (song lyrics, "[Music]" tags, garbled words). We do not
    clean this; it is exactly the kind of imperfect transcript the research
    question is about.
  * Videos ship as ONE 161.6 GB tar split into 31 parts, with no per-video
    download option in the Hub API. ``scripts/download_longvideobench_subset.py``
    implements HTTP Range–based selective extraction so a small subset can be
    fetched without downloading the whole archive.

Evidence extraction
--------------------
Many "T*" question categories (T2A, T2O, T2E, T3O, T3E) anchor the question on
an exact quoted subtitle span, e.g. "When the subtitle says 'temples scattered
throughout the', what colour...". We extract that quoted span with a regex and
fuzzy-match it against the video's subtitle lines to find the answer-relevant
time interval. This interval is what ``videoqa.damage`` targets — the visual
ANSWER (a colour, an object) is not in the transcript, but the transcript is
what a retrieval system uses to find *when* to look, so damaging that exact
subtitle line is a direct, dataset-native test of the research hypothesis.

Coverage measured on lvb_val.json T* categories: 350/367 (95%) quotes found by
the regex; the extractor reports whether an interval was found, and
``videoqa.damage.make_triple`` already skips items with no relevant segment
rather than fabricate one.
"""

from __future__ import annotations

import dataclasses
import difflib
import re

from ..schemas import QAItem, Transcript, TranscriptSegment

# Straight and "smart" quotes seen in the questions (U+2018/2019 measured directly
# in the data; straight ' and " cover the rest).
_QUOTE_RE = re.compile(r"['‘’\"]([^'‘’\"]{2,200})['‘’\"]")

T_STAR_CATEGORIES = frozenset({"T2A", "T2O", "T2E", "T3O", "T3E", "TOS"})


def parse_timestamp(ts: str) -> float:
    """"HH:MM:SS.mmm" -> seconds. LongVideoBench always uses this format."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def subtitles_to_transcript(video_id: str, subtitle_entries: list[dict]) -> Transcript:
    """Convert the raw subtitle JSON list into our Transcript, in time order.

    A few entries have identical (degenerate) start/end timestamps in the raw
    data (e.g. "51.709" for both) — TranscriptSegment requires end >= start, so
    those are kept but with end nudged to start (zero-duration point events,
    not dropped: dropping would silently remove real, if brief, speech).
    """
    segs = []
    for i, e in enumerate(sorted(subtitle_entries, key=lambda e: parse_timestamp(e["start"]))):
        text = e["line"].strip()
        if not text:
            continue
        start, end = parse_timestamp(e["start"]), parse_timestamp(e["end"])
        segs.append(TranscriptSegment(id=f"s{i:05d}", start_s=start, end_s=max(start, end), text=text))
    return Transcript(video_id=video_id, segments=segs, source="longvideobench_subtitle")


def extract_quoted_span(question: str) -> str | None:
    """First quoted substring in the question, or None if there isn't one."""
    m = _QUOTE_RE.search(question)
    return m.group(1).strip() if m else None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def find_evidence_interval(quote: str, transcript: Transcript,
                           min_ratio: float = 0.6) -> tuple[list[str], float]:
    """Fuzzy-match ``quote`` against transcript segments (and adjacent pairs,
    since a quote can straddle two subtitle lines). Returns
    (matched_segment_ids, match_ratio); empty list if nothing clears
    ``min_ratio``.

    Matching uses :func:`difflib.SequenceMatcher` ratio on normalised text,
    which tolerates the ASR noise (missing punctuation, minor word gaps) seen
    in this dataset without requiring an exact substring.
    """
    q = _normalise(quote)
    segs = transcript.segments
    best_ids: list[str] = []
    best_ratio = 0.0
    for i, seg in enumerate(segs):
        candidates = [([seg.id], seg.text)]
        if i + 1 < len(segs):
            candidates.append(([seg.id, segs[i + 1].id], seg.text + " " + segs[i + 1].text))
        for ids, text in candidates:
            ratio = difflib.SequenceMatcher(None, q, _normalise(text)).ratio()
            if ratio > best_ratio:
                best_ratio, best_ids = ratio, ids
    return (best_ids, best_ratio) if best_ratio >= min_ratio else ([], best_ratio)


@dataclasses.dataclass
class ConvertedItem:
    qa: QAItem
    evidence_match_ratio: float | None   # None if no quote was found in the question
    quoted_span: str | None


def convert_item(record: dict, transcript: Transcript, license_note: str) -> ConvertedItem:
    """Turn one lvb_val.json record + its Transcript into a QAItem.

    ``record`` is a raw dict from ``lvb_val.json`` (see module docstring for
    its keys). Options/gold come straight from ``candidates``/``correct_choice``
    — no relabelling.
    """
    evidence_ids: list[str] = []
    ratio = None
    quote = extract_quoted_span(record["question"])
    if quote:
        evidence_ids, ratio = find_evidence_interval(quote, transcript)
    intervals = [(transcript.by_id()[i].start_s, transcript.by_id()[i].end_s) for i in evidence_ids]

    qa = QAItem(
        qa_id=record["id"],
        video_id=record["video_id"],
        question=record["question"],
        gold_answer=record["candidates"][record["correct_choice"]],
        options=record["candidates"],
        gold_option_index=record["correct_choice"],
        evidence_segment_ids=[],          # ids are local to OUR transcript build, filled by caller if needed
        evidence_intervals_s=intervals,
        source_dataset="LongVideoBench",
        source_split="unassigned",        # we assign our own train/dev/calibration/test by video (splits.py)
        provenance_and_license=license_note,
    )
    return ConvertedItem(qa=qa, evidence_match_ratio=ratio, quoted_span=quote)
