"""The LLM controller's text input contains only allowlisted observables."""

import dataclasses

from videoqa import fixtures
from videoqa.assay import label_assay
from videoqa.answerer import FixtureAnswerer
from videoqa.config import budget_from_config, load_config
from videoqa.schemas import ActionKind, Candidate, CandidateSource, ControllerObservation
from videoqa.scout import PixelStatsScout
from videoqa.serialize import serialize

CFG = load_config("configs/cpu.yaml")


def _obs(transcript):
    qa = fixtures.qa_items()[1]
    cands = [Candidate("c0", 20.0, CandidateSource.TRANSCRIPT_RETRIEVAL, "w0", 0)]
    return qa, ControllerObservation(qa.question, qa.options, transcript.segments, cands, {}, [], 0, 3, 4, 90.0)


def test_text_is_independent_of_gold_and_labels():
    qa, obs = _obs(fixtures.transcript())
    a = serialize(obs, ActionKind.LOOK_AT_THIS_MOMENT, obs.candidates[0])
    # the observation object carries no gold / condition; changing the QA item's
    # gold answer cannot change the text
    qa2 = dataclasses.replace(qa, gold_answer="SOMETHING ELSE ENTIRELY")
    _, obs2 = _obs(fixtures.transcript())
    assert serialize(obs2, ActionKind.LOOK_AT_THIS_MOMENT, obs2.candidates[0]) == a
    assert qa2.gold_answer not in a
    for word in ("targeted", "control_damage", "clean", "gold", "damage"):
        assert word not in a.lower()


def test_assay_rows_carry_text_without_condition_names(lecture_video):
    qa = [q for q in fixtures.qa_items() if q.qa_id.endswith("q_lr")][0]
    items = [(dataclasses.replace(qa, experiment_split="train"), str(lecture_video), fixtures.transcript())]
    rows = label_assay(items, CFG, budget_from_config(CFG), PixelStatsScout(), FixtureAnswerer())[0]
    assert all(r.obs_text for r in rows)
    for r in rows:
        low = r.obs_text.lower()
        assert "targeted" not in low and "control_damage" not in low and "gold" not in low
