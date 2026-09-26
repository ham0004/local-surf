"""Tests for the information-access rules encoded in videoqa.schemas."""

import dataclasses

import pytest

from videoqa.schemas import (
    ALLOWED_SCOUT_FIELDS,
    ControllerObservation,
    SceneType,
    ScoutSignals,
    TranscriptSegment,
    to_jsonable,
)


def test_segment_rejects_reversed_interval():
    # end before start must fail loudly instead of producing a broken window later
    with pytest.raises(ValueError):
        TranscriptSegment(id="s0", start_s=5.0, end_s=4.0, text="x")


def test_segment_rejects_negative_time():
    with pytest.raises(ValueError):
        TranscriptSegment(id="s0", start_s=-1.0, end_s=1.0, text="x")


def test_scout_fields_are_exactly_the_allowed_set():
    # If someone adds e.g. an "ocr_text" field to ScoutSignals, this fails.
    names = {f.name for f in dataclasses.fields(ScoutSignals)}
    assert names == ALLOWED_SCOUT_FIELDS


def test_scout_rejects_free_text_scene_type():
    # A string like OCR output must not be accepted in place of the enum.
    with pytest.raises(TypeError):
        ScoutSignals(similarity=0.5, scene_type="x = -3", visual_change=0.1,
                     quality=0.9, uncertainty=0.2)


def test_scout_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        ScoutSignals(similarity=1.5, scene_type=SceneType.SLIDE, visual_change=0.1,
                     quality=0.9, uncertainty=0.2)


def test_controller_observation_has_no_label_fields():
    # The controller must never see gold answers, condition names or utilities.
    forbidden = {"gold_answer", "transcript_condition", "damage", "gold_option_index",
                 "evidence_segment_ids", "answer_quality_after_each_action"}
    names = {f.name for f in dataclasses.fields(ControllerObservation)}
    assert names.isdisjoint(forbidden)


def test_to_jsonable_converts_enums_and_nested_dataclasses():
    sig = ScoutSignals(0.1, SceneType.SLIDE, 0.2, 0.3, 0.4)
    out = to_jsonable({"c0": sig})
    assert out["c0"]["scene_type"] == "slide"
    assert out["c0"]["similarity"] == 0.1
