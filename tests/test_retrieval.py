"""Temporal retrieval: ranking, window merging/clipping, candidate bounds."""

from videoqa.retrieval import bm25_rank, build_windows, generate_candidates, rrf_fuse, tokenize
from videoqa.schemas import CandidateSource
from videoqa.transcript import TranscriptUnit


def _units():
    texts = ["introduction to the course", "gradient descent update rule", "learning rate schedule warmup",
             "convolution layers and pooling", "summary and questions"]
    return [TranscriptUnit(id=f"u{i}", segment_ids=[f"s{i}"], start_s=20 * i, end_s=20 * i + 18, text=t)
            for i, t in enumerate(texts)]


def test_tokenize_keeps_decimals():
    assert "0.3" in tokenize("rate is 0.3")


def test_bm25_ranks_matching_unit_first_and_drops_zero_scores():
    ranked = bm25_rank("what is the learning rate schedule", _units())
    assert ranked[0][0].id == "u2"
    assert all(score > 0 for _, score in ranked)


def test_windows_expand_merge_and_clip():
    units = _units()
    ranked = [(units[1], 2.0), (units[2], 3.0)]
    windows = build_windows(ranked, units, max_windows=4, neighbour_expansion=1, video_duration_s=70)
    # u1 and u2 expanded by one neighbour overlap -> a single merged window
    assert len(windows) == 1
    w = windows[0]
    assert w.unit_ids == ["u0", "u1", "u2", "u3"]
    assert w.score == 3.0
    assert w.end_s == 70   # clipped to the video duration


def test_candidates_are_in_bounds_separated_and_rescue_is_outside_windows():
    units = _units()
    windows = build_windows([(units[2], 1.0)], units, neighbour_expansion=0, video_duration_s=100)
    cands = generate_candidates(windows, video_duration_s=100, per_window=3, uniform=5, rescue=3)
    times = [c.time_s for c in cands]
    assert all(0 <= t < 100 for t in times)
    assert all(abs(a - b) >= 1.0 for i, a in enumerate(times) for b in times[i + 1:])
    for c in cands:
        if c.source == CandidateSource.GLOBAL_RESCUE:
            assert not (windows[0].start_s <= c.time_s <= windows[0].end_s)
    assert sum(c.source == CandidateSource.TRANSCRIPT_RETRIEVAL for c in cands) == 3


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b"], ["a", "c"]])
    assert fused["a"] > fused["b"] and fused["a"] > fused["c"]
