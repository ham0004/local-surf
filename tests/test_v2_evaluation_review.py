"""Prevent invalid benchmark scoring and test paired evaluation accounting."""
import importlib.util
from pathlib import Path

import pytest

from videoqa.v2.records import QuestionPool
from videoqa.v2.teacher import CachedTeacher


def test_open_ended_quality_cannot_silently_succeed_from_none_equals_none(tmp_path):
    pool = QuestionPool("q", "v", "explain this", [], None, [], [])
    teacher = CachedTeacher(None, "fake", tmp_path / "unused.jsonl")
    with pytest.raises(ValueError, match="open-ended"):
        teacher.quality(pool, [])
    assert teacher.calls == teacher.hits == 0


def test_paired_eval_uses_only_questions_observed_in_both_arms():
    path = Path(__file__).resolve().parents[1] / "scripts/v2_residual_eval.py"
    spec = importlib.util.spec_from_file_location("review_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.paired([{"video_id": "v", "a": 1, "b": 0},
                            {"video_id": "v", "a": None, "b": 1}], "a", "b", repeats=20)
    assert result["difference"] == 1
    assert result["paired_questions"] == 1
