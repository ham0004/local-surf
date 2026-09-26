"""LongVideoBench adapter: timestamp parsing, quote extraction, evidence matching."""

from videoqa.schemas import Transcript, TranscriptSegment
from videoqa.sources.longvideobench import (
    convert_item,
    extract_quoted_span,
    find_evidence_interval,
    parse_timestamp,
    subtitles_to_transcript,
)


def test_parse_timestamp():
    assert parse_timestamp("00:01:04.830") == 64.83
    assert parse_timestamp("01:00:00.000") == 3600.0


def test_subtitles_to_transcript_sorts_by_start_and_fixes_degenerate_spans():
    raw = [{"start": "00:00:10.000", "end": "00:00:12.000", "line": "second"},
          {"start": "00:00:01.000", "end": "00:00:01.000", "line": "first"},   # zero-duration
          {"start": "00:00:05.000", "end": "00:00:05.500", "line": "  "}]      # blank, dropped
    t = subtitles_to_transcript("vid1", raw)
    assert [s.text for s in t.segments] == ["first", "second"]
    assert t.segments[0].start_s == t.segments[0].end_s == 1.0   # kept, not dropped


def test_extract_quoted_span_handles_straight_and_smart_quotes():
    assert extract_quoted_span("the subtitle says 'hello world'") == "hello world"
    assert extract_quoted_span("caption ‘like this’ appears") == "like this"
    assert extract_quoted_span("no quotes here at all") is None


def test_find_evidence_interval_tolerates_asr_noise_and_line_splits():
    segs = [TranscriptSegment("s0", 0, 5, "temples scattered"), TranscriptSegment("s1", 5, 9, "throughout the land"),
            TranscriptSegment("s2", 20, 25, "completely unrelated line about weather")]
    t = Transcript("v", segs)
    ids, ratio = find_evidence_interval("temples scattered throughout the", t)
    assert ids == ["s0", "s1"] and ratio > 0.8


def test_find_evidence_interval_returns_empty_below_threshold():
    segs = [TranscriptSegment("s0", 0, 5, "completely different topic entirely")]
    ids, ratio = find_evidence_interval("temples scattered throughout the ancient city", Transcript("v", segs))
    assert ids == []


def test_convert_item_builds_qa_and_reports_match_confidence():
    segs = [TranscriptSegment("s0", 16.0, 22.0, "temples scattered throughout the land")]
    t = Transcript("v1", segs)
    record = {"id": "v1_0", "video_id": "v1",
              "question": "When the subtitle says 'temples scattered throughout the', what colour is shown?",
              "candidates": ["yellow", "green", "blue", "red"], "correct_choice": 0}
    out = convert_item(record, t, "CC-BY-NC-SA-4.0, longvideobench/LongVideoBench")
    assert out.qa.gold_answer == "yellow" and out.qa.gold_option_index == 0
    assert out.qa.evidence_intervals_s == [(16.0, 22.0)]
    assert out.qa.source_split == "unassigned"
    assert out.evidence_match_ratio > 0.6
    assert out.quoted_span == "temples scattered throughout the"


def test_convert_item_without_a_quote_has_no_evidence_interval():
    t = Transcript("v1", [TranscriptSegment("s0", 0, 5, "anything")])
    record = {"id": "v1_1", "video_id": "v1", "question": "What is the man doing?",
              "candidates": ["dancing", "singing"], "correct_choice": 1}
    out = convert_item(record, t, "license")
    assert out.qa.evidence_intervals_s == [] and out.evidence_match_ratio is None
