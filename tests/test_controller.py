"""Policies choose only legal actions, and utility controllers trade gain for measured cost."""

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
from videoqa.cost_model import DEFAULT
from videoqa.features import CANDIDATE_FEATURES, legal_actions_costed
from videoqa.heads import UtilityHead
from videoqa.schemas import ActionKind, Candidate, CandidateSource, ControllerObservation, SceneType, ScoutSignals


def _obs(frames_remaining=3, looked=()):
    qa = fixtures.qa_items()[1]
    cands = [Candidate("c000", 20.0, CandidateSource.TRANSCRIPT_RETRIEVAL, "w0", rank=0),
             Candidate("c001", 35.0, CandidateSource.UNIFORM),
             Candidate("c002", 5.0, CandidateSource.UNIFORM),
             Candidate("c003", 80.0, CandidateSource.GLOBAL_RESCUE)]
    scout = {c.id: ScoutSignals(0.1 * i, SceneType.SLIDE, 0.0, 0.8, 0.2) for i, c in enumerate(cands)}
    return ControllerObservation(qa.question, qa.options, fixtures.transcript().segments, cands, scout,
                                 list(looked), 0, frames_remaining, 4, 90.0)


def _legal(obs, rescue_left=1, tokens_left=4096, look_ms=100.0, expand_ms=5.0):
    return legal_actions_costed(obs, rescue_left, 2, 64, tokens_left, lambda cid: look_ms, expand_ms)


def test_transcript_only_always_stops_and_needs_no_scout():
    p = TranscriptOnlyPolicy()
    assert p.decide(_obs(), _legal(_obs())).kind == ActionKind.STOP and p.scout_seed == "none"


def test_uniform_spreads_budget_over_the_video():
    # cap 3 over 90 s -> targets 15, 45, 75 s; nearest non-rescue candidates:
    # 15 -> c000 (20 s); then 45 -> c001 (35 s)
    assert UniformPolicy().decide(_obs(), _legal(_obs())).candidate_id == "c000"
    o = _obs(frames_remaining=2, looked=["c000"])
    assert UniformPolicy().decide(o, _legal(o)).candidate_id == "c001"


def test_retrieval_and_similarity_policies():
    assert RetrievalPolicy().decide(_obs(), _legal(_obs())).candidate_id == "c000"
    # c003 (0.3) is highest but is a rescue candidate -> not a LOOK_AT action; next is c002 (0.2)
    assert ScoutSimilarityPolicy().decide(_obs(), _legal(_obs())).candidate_id == "c002"
    assert ScoutSimilarityPolicy.scout_seed == "all"


def test_every_policy_stops_when_no_look_is_legal():
    o = _obs(frames_remaining=0)
    for p in (UniformPolicy(), RetrievalPolicy(), ScoutSimilarityPolicy()):
        assert p.decide(o, _legal(o)).kind == ActionKind.STOP


def test_visual_token_budget_is_checked_before_any_look():
    # 64 tokens per frame but only 50 left: no LOOK may be offered at all.
    legal = _legal(_obs(), tokens_left=50)
    assert all(a.kind not in (ActionKind.LOOK_AT_THIS_MOMENT, ActionKind.LOOK_ELSEWHERE) for a in legal)


def test_heuristic_returns_a_legal_action_with_reason():
    obs = _obs()
    legal = _legal(obs)
    a = HeuristicController().decide(obs, legal)
    assert (a.kind, a.candidate_id) in {(x.kind, x.candidate_id) for x in legal} and a.reason


def _head(bias):
    head = UtilityHead(len(CANDIDATE_FEATURES), 4, 0)
    head.w2[:] = 0.0
    head.b2[:] = bias
    return head


def test_learned_controller_stops_when_gain_does_not_pay_for_cost():
    # Every action predicted to gain 0.05; every action costs 100 ms.
    ctrl = LearnedController(_head(0.05), DEFAULT.with_lambda(1.0))     # 0.05 - 1.0*0.1 < 0 -> STOP
    assert ctrl.decide(_obs(), _legal(_obs(), expand_ms=100.0)).kind == ActionKind.STOP
    cheap = LearnedController(_head(0.05), DEFAULT.with_lambda(0.1))    # 0.05 - 0.01 > 0 -> act
    a = cheap.decide(_obs(), _legal(_obs()))
    assert a.kind != ActionKind.STOP and np.isfinite(a.predicted_utility)


def test_cost_decides_between_equally_useful_actions():
    # Same predicted gain everywhere: the cheapest legal action must win.
    obs = _obs()
    legal = legal_actions_costed(obs, 1, 0, 64, 4096, lambda cid: {"c001": 10.0}.get(cid, 500.0), 5.0)
    a = LearnedController(_head(0.5), DEFAULT.with_lambda(1.0)).decide(obs, legal)
    assert a.candidate_id == "c001"
