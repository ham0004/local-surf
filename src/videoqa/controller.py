"""Controller policies: decide LOOK / LOOK_ELSEWHERE / EXPAND_TRANSCRIPT / STOP.

Every policy has the same signature and sees only a ControllerObservation, so
baselines and the learned controller are compared on equal information.

Baselines (Stage 1 of the research plan)
    TranscriptOnlyPolicy   never looks; answers from transcript alone.
    UniformPolicy          looks at uniform-coverage candidates in time order.
    RetrievalPolicy        looks at transcript-retrieved candidates in order.
    ScoutSimilarityPolicy  looks at the highest scout-similarity candidates.
    HeuristicController    hand-set linear utility over the same features the
                           learned head uses; no training.
Learned
    LearnedController      UtilityHead (heads.py) predicts each action's gain;
                           picks the best net of a cost penalty; STOPs when the
                           best predicted net gain is below a threshold tuned
                           on held-out data.

Note: the spec's eventual controller is a small text LLM (e.g. Qwen3-0.6B +
LoRA) emitting the same typed actions.  The numeric head comes first because
it is cheap, auditable and isolates the paired-loss question; see
docs/progress.md for the LLM-controller status.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .features import CANDIDATE_FEATURES, candidate_features, expand_features, global_features, legal_actions
from .heads import UtilityHead
from .schemas import Action, ActionKind, CandidateSource, ControllerObservation


class Policy(Protocol):
    name: str

    def decide(self, obs: ControllerObservation, rescue_frames_left: int, max_expansions: int) -> Action:
        ...


def _look_actions(obs, rescue_frames_left, max_expansions, sources=None):
    """Legal LOOK / LOOK_ELSEWHERE actions, optionally filtered by candidate source."""
    by_id = {c.id: c for c in obs.candidates}
    out = []
    for kind, cid in legal_actions(obs, rescue_frames_left, max_expansions):
        if cid is None:
            continue
        if sources is None or by_id[cid].source in sources:
            out.append((kind, by_id[cid]))
    return out


class TranscriptOnlyPolicy:
    name = "transcript_only"

    def decide(self, obs, rescue_frames_left, max_expansions):
        return Action(ActionKind.STOP, reason="POLICY_NEVER_LOOKS")


class UniformPolicy:
    name = "uniform"

    def decide(self, obs, rescue_frames_left, max_expansions):
        acts = _look_actions(obs, rescue_frames_left, max_expansions, {CandidateSource.UNIFORM})
        if not acts:
            return Action(ActionKind.STOP, reason="NO_UNIFORM_LEFT")
        kind, cand = min(acts, key=lambda a: a[1].time_s)
        return Action(kind, cand.id, reason="UNIFORM_NEXT")


class RetrievalPolicy:
    """Transcript-similarity timestamp retrieval: look where the transcript matched."""

    name = "retrieval"

    def decide(self, obs, rescue_frames_left, max_expansions):
        acts = _look_actions(obs, rescue_frames_left, max_expansions, {CandidateSource.TRANSCRIPT_RETRIEVAL})
        if not acts:
            return Action(ActionKind.STOP, reason="NO_RETRIEVED_LEFT")
        # candidates are generated best-window-first; keep that order
        kind, cand = min(acts, key=lambda a: a[1].id)
        return Action(kind, cand.id, reason="RETRIEVAL_NEXT")


class ScoutSimilarityPolicy:
    """Visual-similarity frame selection: highest question-frame similarity first."""

    name = "scout_similarity"

    def decide(self, obs, rescue_frames_left, max_expansions):
        acts = [a for a in _look_actions(obs, rescue_frames_left, max_expansions)
                if a[0] == ActionKind.LOOK_AT_THIS_MOMENT]
        if not acts:
            return Action(ActionKind.STOP, reason="NO_CANDIDATES_LEFT")
        kind, cand = max(acts, key=lambda a: (obs.scout[a[1].id].similarity if a[1].id in obs.scout else 0.0,
                                               -a[1].time_s))
        return Action(kind, cand.id, reason="HIGHEST_SIMILARITY")


# ---------------------------------------------------------------------------
# Utility-scoring controllers (heuristic and learned share the decision rule)
# ---------------------------------------------------------------------------


class _UtilityController:
    """Shared decision rule: score every legal action, subtract a cost
    penalty, take the best; STOP if nothing clears ``stop_threshold``."""

    name = "utility"
    stop_threshold = 0.05
    look_cost = 0.02      # penalty per frame, in answer-quality units
    expand_cost = 0.005

    def score(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def decide(self, obs, rescue_frames_left, max_expansions):
        g = global_features(obs)
        acts = legal_actions(obs, rescue_frames_left, max_expansions)
        by_id = {c.id: c for c in obs.candidates}
        rows, keys = [], []
        for kind, cid in acts:
            if kind == ActionKind.STOP:
                continue
            if kind == ActionKind.EXPAND_TRANSCRIPT:
                rows.append(expand_features(obs, g))
            else:
                rows.append(candidate_features(obs, by_id[cid], g))
            keys.append((kind, cid))
        if not rows:
            return Action(ActionKind.STOP, reason="NO_LEGAL_ACTION")
        utils = self.score(np.asarray(rows, dtype=np.float64))
        costs = np.array([self.expand_cost if k == ActionKind.EXPAND_TRANSCRIPT else self.look_cost
                          for k, _ in keys])
        net = utils - costs
        best = int(np.argmax(net))
        if net[best] < self.stop_threshold:
            return Action(ActionKind.STOP, predicted_utility=float(utils[best]), reason="NO_ACTION_WORTH_COST")
        kind, cid = keys[best]
        return Action(kind, cid, predicted_utility=float(utils[best]), reason=_reason(kind, rows[best]))


def _reason(kind: ActionKind, row: list[float]) -> str:
    """Short reason code for logs (derived from features, not from labels)."""
    idx = {n: i for i, n in enumerate(CANDIDATE_FEATURES)}
    if kind == ActionKind.EXPAND_TRANSCRIPT:
        return "DEFINITION_OR_CONTEXT_NEEDED"
    if kind == ActionKind.LOOK_ELSEWHERE:
        return "GLOBAL_RESCUE"
    if row[idx["local_speech_coverage"]] < 0.3:
        return "SPEECH_GAP_NEAR_CANDIDATE"
    if row[idx["question_visual_cue"]] > 0:
        return "QUESTION_ASKS_ABOUT_VISUALS"
    return "RELEVANT_WINDOW"


class HeuristicController(_UtilityController):
    """No-training baseline: a transparent hand-set linear utility.

    Intuition encoded (NOT learned): look more when question terms are missing
    from the transcript, when speech is absent near a retrieved candidate, when
    the question asks about something shown, and at novel, sharp, changing
    frames.  Weights are round numbers on purpose - they are a baseline, not a
    tuned model.
    """

    name = "heuristic"
    _W = {
        "scout_similarity": 0.10, "scout_visual_change": 0.10, "scout_quality": 0.05,
        "src_retrieval": 0.10, "src_rescue": -0.05,
        "local_question_overlap": 0.10, "local_speech_coverage": -0.10,
        "question_term_coverage": -0.20, "question_visual_cue": 0.20,
        "min_dist_to_looked": 0.10, "n_looked_frac": -0.30, "expand_action": -0.05,
        "bias": 0.10,
    }

    def __init__(self, stop_threshold: float = 0.05) -> None:
        self.stop_threshold = stop_threshold
        self.w = np.array([self._W.get(n, 0.0) for n in CANDIDATE_FEATURES])

    def score(self, X):
        return X @ self.w


class LearnedController(_UtilityController):
    name = "learned"

    def __init__(self, head: UtilityHead, stop_threshold: float = 0.05, look_cost: float = 0.02) -> None:
        self.head, self.stop_threshold, self.look_cost = head, stop_threshold, look_cost

    @classmethod
    def from_file(cls, path) -> LearnedController:
        head, meta = UtilityHead.load(path)
        return cls(head, stop_threshold=meta.get("stop_threshold", 0.05), look_cost=meta.get("look_cost", 0.02))

    def score(self, X):
        return self.head.predict(X)


def make_policy(name: str, checkpoint: str | None = None, **kwargs) -> Policy:
    simple = {"transcript_only": TranscriptOnlyPolicy, "uniform": UniformPolicy, "retrieval": RetrievalPolicy,
              "scout_similarity": ScoutSimilarityPolicy}
    if name in simple:
        return simple[name]()
    if name == "heuristic":
        return HeuristicController(**kwargs)
    if name == "learned":
        if not checkpoint:
            raise ValueError("learned policy needs --checkpoint")
        return LearnedController.from_file(checkpoint)
    raise ValueError(f"unknown policy {name!r}")
