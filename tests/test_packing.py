"""Excerpt packing keeps verbatim text, respects the budget, and protects numbers."""

from videoqa.packing import format_excerpt, pack_excerpt
from videoqa.retrieval import Window
from videoqa.schemas import TranscriptSegment


def _segs():
    texts = ["we now define the rate", "the learning rate is not 0.3 but 0.03",
             "some unrelated chatter about the weather today", "more filler words here"]
    return [TranscriptSegment(id=f"s{i}", start_s=10 * i, end_s=10 * i + 5, text=t) for i, t in enumerate(texts)]


def test_excerpt_is_verbatim_ordered_and_keeps_numbers_and_negation():
    segs = _segs()
    w = Window(id="w0", unit_ids=["u0"], start_s=0, end_s=40, score=1)
    out = pack_excerpt("what is the learning rate", segs, [w], {"u0": ["s0", "s1", "s2", "s3"]}, max_words=16)
    texts = [s.text for s in out]
    assert "the learning rate is not 0.3 but 0.03" in texts   # verbatim, negation intact
    assert "we now define the rate" in texts                  # preceding context kept
    assert [s.start_s for s in out] == sorted(s.start_s for s in out)
    assert sum(len(t.split()) for t in texts) <= 16


def test_format_includes_ids_and_times():
    line = format_excerpt(_segs()[:1])
    assert line == "[s0 0.0-5.0s] we now define the rate"


def test_excerpt_falls_back_to_whole_transcript_when_retrieval_finds_nothing():
    # Regression: with no retrieval window the excerpt used to be EMPTY, so the
    # answerer silently saw no speech and transcript damage had no effect.
    segs = _segs()
    out = pack_excerpt("completely unrelated question", segs, windows=[], unit_to_segments={}, max_words=100)
    assert out and [s.id for s in out] == [s.id for s in sorted(segs, key=lambda s: s.start_s)]
