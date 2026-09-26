"""The matched-control guarantees that make the paired experiment fair.

If any of these fail, targeted-vs-control comparisons are not interpretable.
"""

import pytest

from videoqa.damage import (
    ANNOTATION_RELEVANCE,
    MASK_TOKEN,
    asr_style_noise,
    auto_min_distance_s,
    derive_relevant_segments,
    make_triple,
    select_matched_control,
)
from videoqa.schemas import DamageType, QAItem, Transcript, TranscriptCondition, TranscriptSegment


def _lecture() -> Transcript:
    """A synthetic 3-minute lecture: one segment every 10 s, one of which states
    the answer ("learning rate 0.3")."""
    lines = [
        "welcome to the lecture on optimisation",
        "first we define the loss function carefully",
        "we plot the loss against training steps",
        "the learning rate we chose is 0.3 for this run",   # s03: answer-bearing
        "notice how the curve flattens after a while",
        "momentum can help escape shallow regions",
        "batch size 32 was used in every experiment",
        "we will now look at a second example",
        "this example uses a convolutional network",
        "the network has 5 layers in total",
        "training took about two hours",
        "we evaluate on a held out split",
        "accuracy improved over the baseline",
        "questions are welcome at the end",
        "thank you all for attending today",
        "see you next week everyone",
    ]
    segs = [TranscriptSegment(id=f"s{i:02d}", start_s=10 * i, end_s=10 * i + 8, text=t)
            for i, t in enumerate(lines)]
    return Transcript("lec", segs, source="fixture")


def _qa(**kw) -> QAItem:
    base = dict(qa_id="q1", video_id="lec", question="What learning rate was chosen?", gold_answer="0.3")
    base.update(kw)
    return QAItem(**base)


def test_relevance_prefers_annotations_then_falls_back_to_answer_overlap():
    t = _lecture()
    assert derive_relevant_segments(_qa(evidence_segment_ids=["s05"]), t).method == "annotation_ids"
    assert derive_relevant_segments(_qa(evidence_intervals_s=[(31, 33)]), t).segment_ids == ["s03"]
    rel = derive_relevant_segments(_qa(), t)
    assert rel.method == "answer_overlap" and rel.segment_ids == ["s03"]


@pytest.mark.parametrize("dtype", [DamageType.DELETE, DamageType.MASK_WORDS, DamageType.TIME_SHIFT])
def test_triple_is_matched_in_count_and_type(dtype):
    triple = make_triple(_qa(), _lecture(), dtype=dtype, seed=3)
    assert triple is not None
    t_rec = triple.records[TranscriptCondition.TARGETED_DAMAGE]
    c_rec = triple.records[TranscriptCondition.CONTROL_DAMAGE]
    # same damage type and same number of damaged segments
    assert t_rec.damage_type == c_rec.damage_type == dtype
    assert len(t_rec.damaged_segment_ids) == len(c_rec.damaged_segment_ids)
    # control never touches the answer-relevant segment
    assert set(c_rec.damaged_segment_ids).isdisjoint(t_rec.damaged_segment_ids)
    # word counts matched closely (fixture lines are 6-10 words)
    assert abs(t_rec.words_affected - c_rec.words_affected) <= 3


def test_control_is_far_from_relevant_and_answer_free():
    t = _lecture()
    ctrl = select_matched_control(t, ["s03"], DamageType.DELETE, "0.3", seed=0, min_distance_s=10)
    seg = t.by_id()[ctrl[0]]
    assert "0.3" not in seg.text
    assert ctrl[0] not in {"s02", "s03", "s04"}   # neighbours are within 10 s


def test_number_swap_control_requires_a_number():
    triple = make_triple(_qa(), _lecture(), dtype=DamageType.NUMBER_SWAP, seed=1)
    c_ids = triple.records[TranscriptCondition.CONTROL_DAMAGE].damaged_segment_ids
    by_id = _lecture().by_id()
    assert any(ch.isdigit() for ch in by_id[c_ids[0]].text) or "two" in by_id[c_ids[0]].text


def test_targeted_delete_removes_answer_but_clean_is_untouched():
    t = _lecture()
    triple = make_triple(_qa(), t, dtype=DamageType.DELETE, seed=0)
    assert "0.3" not in " ".join(s.text for s in triple.targeted.segments)
    assert "0.3" in " ".join(s.text for s in triple.control.segments)
    assert triple.clean is t and len(t.segments) == 16   # original not mutated


def test_mask_leaves_marker_tokens():
    triple = make_triple(_qa(), _lecture(), dtype=DamageType.MASK_WORDS, seed=0)
    assert MASK_TOKEN in " ".join(s.text for s in triple.targeted.segments)


def test_triple_is_deterministic_for_a_seed():
    a = make_triple(_qa(), _lecture(), seed=7)
    b = make_triple(_qa(), _lecture(), seed=7)
    assert a.records == b.records


def test_no_triple_when_nothing_is_answer_relevant():
    assert make_triple(_qa(gold_answer="zebra"), _lecture()) is None


def test_asr_noise_keeps_times_valid():
    noisy = asr_style_noise(_lecture(), word_error_rate=0.3, seed=0)
    assert all(0 <= s.start_s <= s.end_s for s in noisy.segments)


def test_auto_min_distance_caps_at_10s_for_a_lecture_length_span():
    # The full-lecture fixture spans 160s, well past where 15% of the span
    # would exceed the cap -> identical to the old hardcoded default, so
    # every existing lecture-based test keeps its original behaviour.
    assert auto_min_distance_s(_lecture()) == 10.0


def test_auto_min_distance_scales_down_for_a_short_clip():
    # Regression driver: a 9s clip with a 7s covered span (measured directly
    # from a real LongVideoBench video) must NOT get the 10s lecture default,
    # or almost no segment could ever qualify as a control.
    segs = [TranscriptSegment("s0", 0.0, 1.0, "a"), TranscriptSegment("s1", 6.0, 7.0, "b")]
    t = Transcript("clip", segs)
    d = auto_min_distance_s(t)
    assert 1.0 <= d < 10.0
    assert d == pytest.approx(7.0 * 0.15)


def test_auto_min_distance_has_a_floor_for_a_near_instantaneous_span():
    t = Transcript("v", [TranscriptSegment("s0", 0.0, 0.1, "a"), TranscriptSegment("s1", 0.2, 0.3, "b")])
    assert auto_min_distance_s(t) == 1.0


def test_make_triple_recovers_a_short_clip_that_a_fixed_10s_distance_would_reject():
    # 9s clip, 3 short segments - representative of the real failure mode.
    segs = [TranscriptSegment("s0", 0.0, 1.0, "welcome to the shop"),
           TranscriptSegment("s1", 2.0, 3.0, "eggs are on the bottom shelf"),
           TranscriptSegment("s2", 6.0, 7.0, "thanks for watching")]
    t = Transcript("clip", segs)
    qa = QAItem(qa_id="q", video_id="clip", question="what is on the shelf",
               gold_answer="eggs", evidence_segment_ids=["s1"])
    assert make_triple(qa, t) is not None                          # auto distance recovers it
    assert make_triple(qa, t, min_distance_s=10.0) is None          # the old fixed default still fails it


def test_annotation_only_relevance_rejects_the_answer_overlap_fallback():
    # _qa() has no annotations, so relevance falls back to answer-word overlap.
    assert make_triple(_qa(), _lecture()) is not None
    assert make_triple(_qa(), _lecture(), relevance_methods=ANNOTATION_RELEVANCE) is None
    annotated = _qa(evidence_intervals_s=[(31.0, 33.0)])
    assert make_triple(annotated, _lecture(), relevance_methods=ANNOTATION_RELEVANCE) is not None
