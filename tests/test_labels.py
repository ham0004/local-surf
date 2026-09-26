"""Automatic label generation on the fixture lecture (fixture answerer, CPU)."""

import dataclasses

import pytest

from videoqa import fixtures
from videoqa.answerer import FixtureAnswerer
from videoqa.config import budget_from_config, load_config
from videoqa.damage import make_triple
from videoqa.features import CANDIDATE_FEATURES
from videoqa.labels import label_items, pair_rows, read_rows, write_labels
from videoqa.schemas import ActionKind, DamageType
from videoqa.scout import PixelStatsScout

CFG = load_config("configs/cpu.yaml")
BUDGET = dataclasses.replace(budget_from_config(CFG), max_frames=8)


def _qa(name, split="train"):
    qa = {q.qa_id.split("/")[-1]: q for q in fixtures.qa_items()}[name]
    return dataclasses.replace(qa, source_split=split)


@pytest.fixture(scope="module")
def labelled(lecture_video):
    qa = _qa("q_lr")
    triple = make_triple(qa, fixtures.transcript(), dtype=DamageType.DELETE, seed=0)
    return label_items([(qa, str(lecture_video), triple)], CFG, BUDGET, PixelStatsScout(), FixtureAnswerer())


def test_rows_have_features_and_measured_quality(labelled):
    rows, examples, stats = labelled
    assert rows and all(len(r.features) == len(CANDIDATE_FEATURES) for r in rows)
    assert all(0.0 <= r.quality_before <= 1.0 and 0.0 <= r.quality_after <= 1.0 for r in rows)
    assert stats.answer_calls > 0 and stats.cache_hits > 0


def test_targeted_damage_creates_positive_gain_that_clean_does_not(labelled):
    rows, _, _ = labelled
    step0 = [r for r in rows if r.step == 0]
    # Clean transcript already answers q_lr -> no action can improve it.
    assert max(r.gain for r in step0 if r.condition == "clean") == 0.0
    # Targeted damage removed the spoken answer -> looking at the slide helps.
    assert max(r.gain for r in step0 if r.condition == "targeted_damage") == 1.0
    # Matched control damage left the answer intact -> no gain available.
    assert max(r.gain for r in step0 if r.condition == "control_damage") == 0.0


def test_preferred_action_is_stop_unless_targeted(labelled):
    _, examples, _ = labelled
    step0 = {}
    for e in examples:                       # examples are in step order per condition
        step0.setdefault(e.transcript_condition.value, e)
    assert step0["clean"].preferred_action_or_stop.kind == ActionKind.STOP
    assert step0["control_damage"].preferred_action_or_stop.kind == ActionKind.STOP
    assert step0["targeted_damage"].preferred_action_or_stop.kind != ActionKind.STOP


def test_pairs_match_question_kind_and_time(labelled):
    rows, _, _ = labelled
    pairs = pair_rows(rows)
    assert pairs
    for t, c in pairs:
        assert rows[t].condition == "targeted_damage" and rows[c].condition == "control_damage"
        assert rows[t].qa_id == rows[c].qa_id and rows[t].action_kind == rows[c].action_kind
        if rows[t].candidate_time_s is not None:
            assert abs(rows[t].candidate_time_s - rows[c].candidate_time_s) <= 0.5


def test_refuses_to_label_non_train_split(lecture_video):
    qa = _qa("q_lr", split="test")
    triple = make_triple(qa, fixtures.transcript(), seed=0)
    with pytest.raises(ValueError):
        label_items([(qa, str(lecture_video), triple)], CFG, BUDGET, PixelStatsScout(), FixtureAnswerer())


def test_write_and_read_roundtrip(labelled, tmp_path):
    rows, examples, _ = labelled
    write_labels(tmp_path, rows, examples, {"answerer": "fixture"})
    rows2, pairs2 = read_rows(tmp_path)
    assert len(rows2) == len(rows) and pairs2 == pair_rows(rows)


def test_answer_cache_never_shares_answers_across_questions(lecture_video):
    # Regression: the cache key once omitted the question id, so a second
    # question silently reused the first question's answers.
    items = []
    for name in ("q_lr", "q_acc"):
        qa = _qa(name)
        triple = make_triple(qa, fixtures.transcript(), dtype=DamageType.DELETE, seed=0)
        items.append((qa, str(lecture_video), triple))
    _, _, both = label_items(items, CFG, BUDGET, PixelStatsScout(), FixtureAnswerer())
    _, _, one = label_items(items[:1], CFG, BUDGET, PixelStatsScout(), FixtureAnswerer())
    assert both.answer_calls > one.answer_calls
