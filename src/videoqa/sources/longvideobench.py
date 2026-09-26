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
# in the data; straight ' and " cover the rest).  The same characters are also
# APOSTROPHES ("there's", "tank's", "I'd"), so an opening quote must not be glued
# to a preceding word character and a closing quote must not be followed by one.
# Measured: without these guards 29/440 T* questions yielded a span starting at an
# apostrophe (e.g. "s a frame with a blue background wall..."), and quotes that
# contain an apostrophe ('I'd be happy ...') were cut short.
_QUOTE_RE = re.compile(r"(?<!\w)['‘\"](.{1,200}?)['’\"](?!\w)")
# Quotes shorter than this cannot be located in a transcript reliably ("e",
# "so", "i" match almost anything); they are still parsed, so they cannot
# swallow the text up to the NEXT quote, but are not used as evidence.
MIN_LOCATABLE_QUOTE_CHARS = 3

T_STAR_CATEGORIES = frozenset({"T2A", "T2O", "T2E", "T3O", "T3E", "TOS"})


def parse_timestamp(ts: str) -> float:
    """"HH:MM:SS.mmm" -> seconds. LongVideoBench always uses this format."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _entry_fields(e: dict) -> tuple[float, float, str] | None:
    """LongVideoBench ships TWO subtitle shapes (both measured directly from
    the archive, not documented in the README):

      YouTube-sourced videos : {"start": "HH:MM:SS.mmm", "end": "...", "line": "..."}
      TikTok-sourced videos  : {"timestamp": [start_s, end_s], "text": "..."}

    This reads whichever shape a given entry uses. Some TikTok-sourced entries
    have a null end timestamp (e.g. {"timestamp": [23.0, None], "text": " The"}
    — measured directly, an artefact of the upstream ASR/segmentation, not
    something we introduce); such an entry is treated as a zero-duration point
    event at ``start`` rather than dropped, consistent with how degenerate
    YouTube-format spans are handled below. A null START cannot be placed in
    time at all, so that entry is skipped (returns None) rather than guessed.
    """
    if "timestamp" in e:
        start, end = e["timestamp"]
        if start is None:
            return None
        return float(start), float(start if end is None else end), e["text"]
    return parse_timestamp(e["start"]), parse_timestamp(e["end"]), e["line"]


def subtitles_to_transcript(video_id: str, subtitle_entries: list[dict], offset_s: float = 0.0,
                            clip_duration_s: float | None = None) -> Transcript:
    """Convert the raw subtitle JSON list into our Transcript, in time order.

    CRITICAL, measured directly from the archive: many LongVideoBench videos
    are short clips extracted from a much longer source video, but the
    subtitle file shipped for that video id covers the FULL original video's
    audio track — e.g. a video whose ``duration`` field is 9.0s came with a
    subtitle file containing speech timestamped up to 553s. The record field
    ``starting_timestamp_for_subtitles`` gives the clip's start on that FULL
    timeline; without subtracting it, "transcript" text would include speech
    from parts of the source video that were never in the file we actually
    have, silently fabricating evidence for every downstream stage.

    ``offset_s`` (pass the record's ``starting_timestamp_for_subtitles``)
    re-bases every timestamp onto the clip's own local time axis
    (local = global - offset_s). ``clip_duration_s`` (pass the record's
    ``duration``) then drops any segment with NO overlap with ``[0,
    clip_duration_s]`` and clamps the rest to that range. Because only
    line-level (not word-level) timestamps exist, a segment straddling a clip
    boundary keeps its full text even though only part of it is actually
    audible in this file — a known imprecision of the source data, not
    something we can resolve without finer timestamps.

    A few entries have identical (degenerate) start/end timestamps in the raw
    data (e.g. "51.709" for both) — TranscriptSegment requires end >= start, so
    those are kept but with end nudged to start (zero-duration point events,
    not dropped: dropping would silently remove real, if brief, speech).
    """
    segs = []
    parsed = sorted((f for e in subtitle_entries if (f := _entry_fields(e)) is not None), key=lambda t: t[0])
    for start, end, text in parsed:
        text = text.strip()
        if not text:
            continue
        start, end = start - offset_s, max(start, end) - offset_s
        if clip_duration_s is not None:
            if end < 0 or start > clip_duration_s:
                continue  # no overlap with the shipped clip at all
            start, end = max(0.0, start), min(clip_duration_s, end)
        segs.append(TranscriptSegment(id=f"s{len(segs):05d}", start_s=start, end_s=max(start, end), text=text))
    return Transcript(video_id=video_id, segments=segs, source="longvideobench_subtitle")


def extract_quoted_spans(question: str) -> list[str]:
    """Every quoted substring in the question, in order.

    Measured: 50 questions quote two or more spans (e.g. "between 'A' and
    'B'"), and some quote an on-screen letter ("the letter 'e'") before the
    subtitle quote; reading only the first quote picked the wrong one.
    """
    return [m.group(1).strip() for m in _QUOTE_RE.finditer(question)]


def extract_quoted_span(question: str) -> str | None:
    """First quoted substring long enough to locate, or None."""
    spans = [q for q in extract_quoted_spans(question) if len(q) >= MIN_LOCATABLE_QUOTE_CHARS]
    return spans[0] if spans else None


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
    in this dataset without requiring an exact substring. An exact
    (normalised) substring always counts as a full match (1.0): ratio() is
    diluted by the line's length, so e.g. 'Winifred' inside a long line scored
    only 0.24 and was missed. Ties prefer fewer segments (a single line over a
    pair).
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
            t = _normalise(text)
            ratio = 1.0 if q and q in t else difflib.SequenceMatcher(None, q, t).ratio()
            if ratio > best_ratio or (ratio == best_ratio and ids and len(ids) < len(best_ids)):
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
    — no relabelling. Evidence = the union of segments matched by EVERY
    locatable quoted span (a "between 'A' and 'B'" question needs both).
    """
    quotes = [q for q in extract_quoted_spans(record["question"]) if len(q) >= MIN_LOCATABLE_QUOTE_CHARS]
    evidence_ids: list[str] = []
    ratios: list[float] = []
    for quote in quotes:
        ids, r = find_evidence_interval(quote, transcript)
        ratios.append(r)
        evidence_ids += [i for i in ids if i not in evidence_ids]
    ratio = max(ratios) if ratios else None
    quote = " | ".join(quotes) if quotes else None
    by_id = transcript.by_id()
    evidence_ids.sort(key=lambda i: by_id[i].start_s)
    intervals = [(by_id[i].start_s, by_id[i].end_s) for i in evidence_ids]

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
