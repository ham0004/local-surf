"""LongVideoBench adapter: timestamp parsing, quote extraction, evidence matching."""

from videoqa.schemas import Transcript, TranscriptSegment
from videoqa.sources.longvideobench import (
    extract_quoted_spans,
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


def test_subtitles_to_transcript_handles_the_tiktok_shaped_entries_too():
    # Measured directly from the archive: TikTok-sourced videos use
    # {"timestamp": [start, end], "text": ...} instead of start/end/line.
    raw = [{"timestamp": [8.72, 9.74], "text": " Who you gonna call?"},
          {"timestamp": [0.0, 7.2], "text": " Something strange in the neighborhood."}]
    t = subtitles_to_transcript("tiktok_vid", raw)
    assert [s.text for s in t.segments] == ["Something strange in the neighborhood.", "Who you gonna call?"]
    assert t.segments[0].start_s == 0.0 and t.segments[1].end_s == 9.74


def test_subtitles_to_transcript_handles_null_end_and_null_start():
    # Measured directly from the archive: {"timestamp": [23.0, None], ...}.
    raw = [{"timestamp": [23.0, None], "text": " The"},
          {"timestamp": [None, 5.0], "text": " unplaceable, dropped"},
          {"timestamp": [0.0, 1.0], "text": "first"}]
    t = subtitles_to_transcript("vid", raw)
    assert [s.text for s in t.segments] == ["first", "The"]   # null-start entry dropped
    assert t.segments[1].start_s == t.segments[1].end_s == 23.0   # null end -> zero-duration point


def test_subtitles_to_transcript_rebases_onto_the_clips_own_timeline():
    # Measured directly: a 9.0s clip shipped with a subtitle file containing
    # speech up to 553s, because LongVideoBench trims short clips out of a
    # longer source video but ships the FULL video's subtitle track.
    # starting_timestamp_for_subtitles=417 means clip-local 0 == global 417.
    raw = [{"start": "00:06:59.929", "end": "00:07:02.500", "line": "in this clip"},   # 419.929-422.5 global
          {"start": "00:00:01.000", "end": "00:00:02.000", "line": "long before the clip"}]  # 1-2 global
    t = subtitles_to_transcript("vid", raw, offset_s=417.0, clip_duration_s=9.0)
    assert [s.text for s in t.segments] == ["in this clip"]
    seg = t.segments[0]
    assert abs(seg.start_s - 2.929) < 1e-6 and abs(seg.end_s - 5.5) < 1e-6


def test_subtitles_to_transcript_clamps_a_segment_straddling_the_clip_boundary():
    # A line spanning clip-local -1s to +2s (i.e. starts just before the clip)
    # is kept (some of it is audible) but clamped to [0, clip_duration_s].
    raw = [{"start": "00:00:09.000", "end": "00:00:12.000", "line": "straddles the start"}]  # local -1..2 with offset 10
    t = subtitles_to_transcript("vid", raw, offset_s=10.0, clip_duration_s=5.0)
    assert len(t.segments) == 1
    assert t.segments[0].start_s == 0.0 and t.segments[0].end_s == 2.0


def test_subtitles_to_transcript_ids_have_no_gaps_after_dropping_out_of_range_segments():
    raw = [{"start": "00:00:01.000", "end": "00:00:02.000", "line": "kept one"},
          {"start": "00:01:00.000", "end": "00:01:01.000", "line": "far outside clip, dropped"},
          {"start": "00:00:03.000", "end": "00:00:04.000", "line": "kept two"}]
    t = subtitles_to_transcript("vid", raw, offset_s=0.0, clip_duration_s=5.0)
    assert [s.id for s in t.segments] == ["s00000", "s00001"]


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


def test_extract_quoted_span_ignores_apostrophes():
    # Regression, from real LongVideoBench questions: apostrophes in "there's"
    # were taken as opening quotes (29/440 T* questions affected).
    q = "On the screen, there's a frame with a blue wall. After this man says 'by the way', what appears?"
    assert extract_quoted_span(q) == "by the way"
    q2 = "In a yellow tank's turret, when the caption ‘standards our climate’ appears, what is shown?"
    assert extract_quoted_span(q2) == "standards our climate"


def test_extract_quoted_span_keeps_apostrophes_inside_the_quote():
    q = "When the subtitle mentions 'I'd be happy to improve my channel!', what is he holding?"
    assert extract_quoted_span(q) == "I'd be happy to improve my channel!"


def test_one_letter_quote_does_not_swallow_the_next_quote():
    # Real LongVideoBench question: the on-screen letter 'e' is quoted before
    # the subtitle quote; the 1-char quote used to be skipped, so the span ran
    # from 'e' to the NEXT quote's opening mark.
    q = ("... a red light spot, which stops below the letter 'e'. After the subtitles mention "
         "'out the leftover amount you're gonna,' what object appears in her hand?")
    assert extract_quoted_spans(q) == ["e", "out the leftover amount you're gonna,"]
    assert extract_quoted_span(q) == "out the leftover amount you're gonna,"   # 'e' is too short to locate


def test_exact_substring_counts_even_inside_a_long_line():
    # Real miss: 'Winifred' inside a long line scored ratio 0.24 < 0.6.
    seg = TranscriptSegment("s0", 0, 5, "and then we meet Winifred who has been waiting at the station all day long")
    ids, ratio = find_evidence_interval("Winifred", Transcript("v", [seg]))
    assert ids == ["s0"] and ratio == 1.0


def test_convert_item_uses_every_quote_in_a_two_anchor_question():
    segs = [TranscriptSegment("s0", 1, 2, "first we open the box"), TranscriptSegment("s1", 4, 5, "unrelated chatter"),
           TranscriptSegment("s2", 7, 8, "then we close the lid")]
    record = {"id": "v_0", "video_id": "v", "candidates": ["a", "b"], "correct_choice": 0,
              "question": "What happens between 'first we open the box' and 'then we close the lid'?"}
    out = convert_item(record, Transcript("v", segs), "license")
    assert out.qa.evidence_intervals_s == [(1, 2), (7, 8)]


def test_subtitle_text_is_unescaped_and_single_line():
    # Real example: 'or as an added layer of security&nbsp;\nagainst potential predators.'
    raw = [{"start": "00:00:01.000", "end": "00:00:02.000",
            "line": "or as an added layer of security&nbsp;\nagainst potential predators."}]
    t = subtitles_to_transcript("v", raw)
    assert t.segments[0].text == "or as an added layer of security against potential predators."


def test_quoted_subtitle_is_recorded_as_a_temporal_anchor_not_answer_speech():
    segs = [TranscriptSegment("s0", 1, 2, "the water while increased levels of")]
    record = {"id": "v_0", "video_id": "v", "candidates": ["a", "b"], "correct_choice": 1,
              "question": "What happened after the subtitle 'the water while increased levels of' appeared?"}
    qa = convert_item(record, Transcript("v", segs), "license").qa
    assert qa.evidence_type == "temporal_anchor" and qa.official_split == "validation"
