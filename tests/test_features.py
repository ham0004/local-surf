"""Observable features must react to targeted damage more than to matched control damage."""

from videoqa import fixtures
from videoqa.damage import make_triple
from videoqa.features import (
    CANDIDATE_FEATURES,
    candidate_features,
    expand_features,
    global_features,
    legal_actions,
)
from videoqa.schemas import (
    ActionKind,
    Candidate,
    CandidateSource,
    ControllerObservation,
    DamageType,
)


def _obs(transcript, qa, looked=(), frames_remaining=3):
    cands = [Candidate("c0", 19.0, CandidateSource.TRANSCRIPT_RETRIEVAL, "w0"),
             Candidate("c1", 70.0, CandidateSource.UNIFORM),
             Candidate("c2", 40.0, CandidateSource.GLOBAL_RESCUE)]
    return ControllerObservation(question=qa.question, options=qa.options, transcript=transcript.segments,
                                 candidates=cands, scout={}, looked_at=list(looked), expansions_used=0,
                                 frames_remaining=frames_remaining, rounds_remaining=4, video_duration_s=90)


def test_vector_length_matches_names():
    qa = fixtures.qa_items()[0]
    obs = _obs(fixtures.transcript(), qa)
    assert len(candidate_features(obs, obs.candidates[0])) == len(CANDIDATE_FEATURES)
    assert len(expand_features(obs)) == len(CANDIDATE_FEATURES)


def test_targeted_damage_lowers_question_coverage_and_local_speech_more_than_control():
    qa = fixtures.qa_items()[0]  # learning-rate question, answer spoken at 16-22 s
    triple = make_triple(qa, fixtures.transcript(), dtype=DamageType.DELETE, seed=0)
    idx = {n: i for i, n in enumerate(CANDIDATE_FEATURES)}
    f = {}
    for name, t in (("clean", triple.clean), ("targeted", triple.targeted), ("control", triple.control)):
        obs = _obs(t, qa)
        f[name] = candidate_features(obs, obs.candidates[0])   # candidate at 19 s
    # local speech around the answer time disappears only under targeted damage
    assert f["targeted"][idx["local_speech_coverage"]] < f["control"][idx["local_speech_coverage"]]
    assert f["control"][idx["local_speech_coverage"]] == f["clean"][idx["local_speech_coverage"]]
    assert f["targeted"][idx["local_question_overlap"]] <= f["control"][idx["local_question_overlap"]]


def test_visual_cue_detected_from_question_wording():
    qa = fixtures.qa_items()[1]  # "...shown on the results slide?"
    assert global_features(_obs(fixtures.transcript(), qa))["question_visual_cue"] == 1.0


def test_legal_actions_respect_budget_and_rescue_reserve():
    qa = fixtures.qa_items()[0]
    obs = _obs(fixtures.transcript(), qa, looked=["c0"])
    acts = legal_actions(obs, rescue_frames_left=0)
    kinds = {(k, c) for k, c in acts}
    assert (ActionKind.STOP, None) in kinds
    assert (ActionKind.LOOK_AT_THIS_MOMENT, "c0") not in kinds        # already looked
    assert not any(k == ActionKind.LOOK_ELSEWHERE for k, _ in acts)   # rescue reserve used up
    no_frames = legal_actions(_obs(fixtures.transcript(), qa, frames_remaining=0), rescue_frames_left=1)
    assert all(k in (ActionKind.STOP, ActionKind.EXPAND_TRANSCRIPT) for k, _ in no_frames)
