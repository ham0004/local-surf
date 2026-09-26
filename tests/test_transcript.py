"""Parsing, unit building and reliability features for transcripts."""

from videoqa.schemas import TranscriptSegment
from videoqa.transcript import (
    build_units,
    find_gaps,
    parse_json_segments,
    parse_srt_or_vtt,
    speech_coverage_near,
)

SRT = """1
00:00:01,000 --> 00:00:04,500
Today we study <i>gradient descent</i>.

2
00:00:05,000 --> 00:00:09,000
The learning rate is 0.1.

3
00:00:09,500 --> 00:00:09,600

"""

VTT = """WEBVTT

00:01.5 --> 00:03.000
hello there
"""


def test_parse_srt_strips_tags_and_drops_empty_cues():
    t = parse_srt_or_vtt(SRT, video_id="v")
    assert [s.text for s in t.segments] == ["Today we study gradient descent.", "The learning rate is 0.1."]
    assert t.segments[0].start_s == 1.0 and t.segments[0].end_s == 4.5


def test_parse_vtt_short_timestamps_pad_milliseconds():
    t = parse_srt_or_vtt(VTT, video_id="v")
    # "01.5" is 1.5 s, not 1.005 s
    assert t.segments[0].start_s == 1.5


def test_parse_json_skips_empty_text():
    t = parse_json_segments([{"start": 0, "end": 1, "text": " "}, {"start": 1, "end": 2, "text": "a"}], "v")
    assert len(t.segments) == 1 and t.segments[0].id == "s00000"


def _segs(spans):
    return [TranscriptSegment(id=f"s{i}", start_s=a, end_s=b, text=f"w{i}") for i, (a, b) in enumerate(spans)]


def test_units_keep_segment_provenance_and_respect_length():
    segs = _segs([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)])
    units = build_units(segs, unit_seconds=12, min_unit_seconds=5)
    # every original segment appears in exactly one unit
    flat = [sid for u in units for sid in u.segment_ids]
    assert sorted(flat) == sorted(s.id for s in segs)
    assert all(u.end_s - u.start_s <= 12 for u in units)


def test_units_do_not_bridge_long_silence():
    segs = _segs([(0, 2), (100, 102)])
    units = build_units(segs, unit_seconds=20, min_unit_seconds=0)
    assert len(units) == 2


def test_gaps_include_head_middle_and_tail():
    segs = _segs([(5, 10), (20, 25)])
    gaps = find_gaps(segs, video_duration_s=40, min_gap_s=3)
    assert [(g.start_s, g.end_s) for g in gaps] == [(0, 5), (10, 20), (25, 40)]


def test_speech_coverage_is_fraction_of_window():
    segs = _segs([(0, 10)])
    assert speech_coverage_near(segs, time_s=10, radius_s=10) == 0.5
