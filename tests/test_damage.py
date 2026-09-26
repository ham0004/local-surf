"""The matched-control guarantees that make the paired experiment fair.

If any of these fail, targeted-vs-control comparisons are not interpretable.
"""

import pytest

from videoqa.damage import (
    MASK_TOKEN,
    asr_style_noise,
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
