"""Policies obey budgets and never invent candidate ids."""

import numpy as np

from videoqa import fixtures
from videoqa.controller import (
    HeuristicController,
    LearnedController,
    RetrievalPolicy,
    ScoutSimilarityPolicy,
    TranscriptOnlyPolicy,
    UniformPolicy,
)
from videoqa.features import CANDIDATE_FEATURES
from videoqa.heads import UtilityHead
from videoqa.schemas import ActionKind, Candidate, CandidateSource, ControllerObservation, SceneType, ScoutSignals


def _obs(frames_remaining=3, looked=()):
    qa = fixtures.qa_items()[1]
    cands = [Candidate("c000", 20.0, CandidateSource.TRANSCRIPT_RETRIEVAL, "w0"),
             Candidate("c001", 35.0, CandidateSource.UNIFORM),
             Candidate("c002", 5.0, CandidateSource.UNIFORM),
             Candidate("c003", 80.0, CandidateSource.GLOBAL_RESCUE)]
    scout = {c.id: ScoutSignals(0.1 * i, SceneType.SLIDE, 0.0, 0.8, 0.2) for i, c in enumerate(cands)}
    return ControllerObservation(qa.question, qa.options, fixtures.transcript().segments, cands, scout,
                                 list(looked), 0, frames_remaining, 4, 90.0)


def test_transcript_only_always_stops():
    assert TranscriptOnlyPolicy().decide(_obs(), 1, 2).kind == ActionKind.STOP


def test_uniform_takes_earliest_uniform_candidate():
    assert UniformPolicy().decide(_obs(), 1, 2).candidate_id == "c002"


def test_retrieval_and_similarity_policies():
    assert RetrievalPolicy().decide(_obs(), 1, 2).candidate_id == "c000"
    # c003 (0.3) is highest but is a rescue candidate -> not a LOOK_AT action; next is c002 (0.2)
    assert ScoutSimilarityPolicy().decide(_obs(), 1, 2).candidate_id == "c002"


def test_every_policy_stops_when_no_frames_remain():
    for p in (UniformPolicy(), RetrievalPolicy(), ScoutSimilarityPolicy()):
        assert p.decide(_obs(frames_remaining=0), 1, 2).kind == ActionKind.STOP


def test_heuristic_returns_a_legal_candidate_with_reason():
    obs = _obs()
    a = HeuristicController().decide(obs, 1, 2)
    assert a.kind != ActionKind.STOP
    assert a.candidate_id in {c.id for c in obs.candidates} or a.kind == ActionKind.EXPAND_TRANSCRIPT
    assert a.reason


def test_learned_controller_stops_when_predicted_gain_is_low():
    head = UtilityHead(len(CANDIDATE_FEATURES), 4, 0)
    head.w2[:] = 0.0
    head.b2[:] = -1.0          # predicts negative utility for everything
    assert LearnedController(head).decide(_obs(), 1, 2).kind == ActionKind.STOP
    head.b2[:] = 1.0
    assert LearnedController(head).decide(_obs(), 1, 2).kind != ActionKind.STOP
    assert np.isfinite(LearnedController(head).decide(_obs(), 1, 2).predicted_utility)
