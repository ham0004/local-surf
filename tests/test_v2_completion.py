"""CPU tests for the fourth-frame completion experiment: leakage, fallback, label checks."""
import copy

import numpy as np
import pytest

from videoqa.schemas import TranscriptSegment
from videoqa.v2.completion import (anchor_and_rest, build_questions, completion_design,
                                   completion_feature_names, fit_ridge, nested_cv, pick)
from videoqa.v2.records import FrameCandidate, QuestionPool


def fixture(n_videos=4):
    """Pools of 6 candidates; the frame with OCR 'rank 2' completes the answer."""
    pools, rows = {}, []
    for v in range(n_videos):
        q = f"q{v}"
        cands = [FrameCandidate(f"{q}:f{i}", i * 5.0, ("A",) if i < 3 else ("B",), f"d{i}", "0",
                                clip_sim=i / 5, head_a_visual=10 - i,
                                ocr_text="rank 2 matrix" if i == 5 else "vector",
                                near_text="matrix", emb=np.array([1., i + 1.]),
                                ocr_emb=np.array([i + 1., 1.]), near_emb=np.array([1., 1.]))
                 for i in range(6)]
        pools[q] = QuestionPool(q, f"v{v}", "what is the rank?", ["2", "3"], 0,
                                [TranscriptSegment("s", 0., 20., "the matrix")], cands,
                                duration_s=30., question_emb=np.array([1., 0.]))
        anchor, rest = anchor_and_rest(pools[q])
        for r, i in enumerate(rest):
            rows.append({"qa_id": q, "video_id": f"v{v}", "anchor_ids": [cands[j].id for j in anchor],
                         "candidate_id": cands[i].id, "baseline_rank": r, "before": 0.0,
                         "after": float(i == 5), "gain": float(i == 5)})
    return pools, rows


def test_anchor_is_relevance_top3_and_rest_keeps_baseline_order():
    pools, _ = fixture()
    anchor, rest = anchor_and_rest(pools["q0"])
    assert anchor == [0, 1, 2] and rest == [3, 4, 5]


def test_zero_weights_reproduce_the_baseline_fourth_frame():
    pools, rows = fixture()
    for q in build_questions(pools, rows):
        assert pick(q, np.zeros(len(completion_feature_names()))) == 0
        assert pick(q, fit_ridge([q], None)) == 0


def test_gold_index_never_changes_features():
    pools, _ = fixture()
    a = pools["q0"]
    b = copy.deepcopy(a)
    b.gold_option_index = 1
    anchor, rest = anchor_and_rest(a)
    np.testing.assert_array_equal(completion_design(a, anchor, rest)[0], completion_design(b, anchor, rest)[0])


def test_held_out_labels_do_not_change_held_out_pick():
    pools, rows = fixture(5)
    flipped = copy.deepcopy(rows)
    for r in flipped:
        if r["video_id"] == "v0":
            r["after"] = 1.0 - r["after"]
    a, folds_a = nested_cv(build_questions(pools, rows))
    b, folds_b = nested_cv(build_questions(pools, flipped))
    assert a["q0"] == b["q0"]
    assert folds_a[0]["weights"] == folds_b[0]["weights"]


def test_learner_can_move_away_from_the_baseline_when_labels_disagree():
    pools, rows = fixture(5)
    picks, _ = nested_cv(build_questions(pools, rows))
    assert all(p == 2 for p in picks.values())        # rest index 2 = candidate f5, the labelled winner


def test_incomplete_or_wrong_anchor_labels_are_rejected():
    pools, rows = fixture()
    with pytest.raises(ValueError, match="cover exactly"):
        build_questions(pools, rows[1:])
    bad = copy.deepcopy(rows)
    for r in bad:
        r["anchor_ids"] = ["x", "y", "z"]
    with pytest.raises(ValueError, match="different anchor"):
        build_questions(pools, bad)


def test_duplicate_and_inconsistent_rows_are_rejected():
    pools, rows = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        build_questions(pools, rows + [dict(rows[0])])
    bad = copy.deepcopy(rows)
    bad[0]["before"] = 1.0                                  # one row disagrees on the shared anchor outcome
    with pytest.raises(ValueError, match="three-frame outcome"):
        build_questions(pools, bad)
    bad = copy.deepcopy(rows)
    bad[0]["video_id"] = "elsewhere"
    with pytest.raises(ValueError, match="video id"):
        build_questions(pools, bad)
