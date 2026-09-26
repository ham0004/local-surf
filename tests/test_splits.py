"""Video-level splitting prevents train/test leakage between transcript variants."""

import pytest

from videoqa.schemas import QAItem
from videoqa.splits import assign_splits, check_no_leakage, split_of


def _items(n_videos=200, per_video=3):
    return [QAItem(qa_id=f"v{v}_q{q}", video_id=f"v{v}", question="q", gold_answer="a")
            for v in range(n_videos) for q in range(per_video)]


def test_split_is_deterministic():
    assert split_of("video_42") == split_of("video_42")


def test_all_questions_of_a_video_share_one_split():
    splits = assign_splits(_items())
    check_no_leakage(splits)   # raises on leakage
    assert set(splits) == {"train", "dev", "calibration", "test"}


def test_fractions_are_roughly_respected():
    splits = assign_splits(_items(n_videos=2000, per_video=1))
    assert 0.55 < len(splits["train"]) / 2000 < 0.65


def test_leakage_is_detected():
    a = QAItem(qa_id="1", video_id="same", question="q", gold_answer="a")
    b = QAItem(qa_id="2", video_id="same", question="q", gold_answer="a")
    with pytest.raises(ValueError):
        check_no_leakage({"train": [a], "test": [b]})


def test_course_level_grouping():
    items = [QAItem(qa_id=str(i), video_id=f"course{i % 3}_lec{i}", question="q", gold_answer="a")
             for i in range(30)]
    by_course = assign_splits(items, group_of=lambda qa: qa.video_id.split("_")[0])
    check_no_leakage(by_course, group_of=lambda qa: qa.video_id.split("_")[0])
