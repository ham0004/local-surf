"""Controller policies: decide LOOK / LOOK_ELSEWHERE / EXPAND_TRANSCRIPT / STOP.

Every policy has the same interface: ``decide(obs, legal)``, where ``legal`` is
the list of actions the budget allows right now, each with its measured cost
(``features.legal_actions_costed``; legality is checked in ONE place, not by
each policy). A policy may only return one of those actions or STOP.

Each policy also declares how much visual scouting it needs, so the lazy
acquisition path (acquisition.py) makes it pay for exactly that:
    scout_seed = "none"  no scout calls (transcript-only, uniform, retrieval)
               = "all"   scout every non-rescue candidate (scout similarity)
               = k       scout a small seed set of k candidates (heuristic)

Baselines
    TranscriptOnlyPolicy   never looks.
    UniformPolicy          evenly spaced coverage of the video.
    RetrievalPolicy        where the observed transcript matched, best match first.
    ScoutSimilarityPolicy  highest question-frame similarity first.
    HeuristicController    hand-set linear gain estimate over the same features.

Utility controllers pick the action with the largest
    net = predicted_gain - lambda_per_s * cost_ms / 1000
and STOP when no action has net > stop_threshold (0 by default): gain and cost
are kept separate and combined only here, with the measured CostModel.

Version 1 note: learned controllers (an MLP utility head and a Qwen3-0.6B +
LoRA scorer trained on paired transcript-damage labels) were built and
tested, then removed from this branch because the available benchmarks gave
them no signal to learn (see docs/final_report_controller_study.md; the code
is in git history before the "v1: remove controller training" commit).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .cost_model import DEFAULT, CostModel
from .features import CANDIDATE_FEATURES, LegalAction, candidate_features, expand_features, global_features
from .schemas import Action, ActionKind, CandidateSource, ControllerObservation


class Policy(Protocol):
    name: str
    scout_seed: str | int

    def decide(self, obs: ControllerObservation, legal: list[LegalAction]) -> Action:
        ...


def _looks(legal: list[LegalAction], sources=None, obs: ControllerObservation | None = None) -> list[LegalAction]:
    by_id = {c.id: c for c in obs.candidates} if obs else {}
    out = [a for a in legal if a.kind in (ActionKind.LOOK_AT_THIS_MOMENT, ActionKind.LOOK_ELSEWHERE)]
    if sources is not None:
        out = [a for a in out if by_id[a.candidate_id].source in sources]
    return out


class TranscriptOnlyPolicy:
    name = "transcript_only"
    scout_seed = "none"

    def decide(self, obs, legal):
        return Action(ActionKind.STOP, reason="POLICY_NEVER_LOOKS")


class UniformPolicy:
    """Spread the frame budget evenly over the whole video.

    With a cap of N frames, target times are (k + 0.5) * duration / N.  For each
    target (in order) we take the NEAREST non-rescue candidate, whatever source
    proposed it; a target whose nearest candidate was already acquired counts
    as covered.
    """

    name = "uniform"
    scout_seed = "none"

    def decide(self, obs, legal):
        pool = [c for c in obs.candidates if c.source != CandidateSource.GLOBAL_RESCUE]
        allowed = {a.candidate_id: a for a in _looks(legal)}
        if not pool or not allowed:
            return Action(ActionKind.STOP, reason="NO_CANDIDATES_LEFT")
        n_total = len(obs.looked_at) + obs.frames_remaining          # the frame cap
        for k in range(n_total):
            target = (k + 0.5) * obs.video_duration_s / n_total
            nearest = min(pool, key=lambda c: abs(c.time_s - target))
            if nearest.id in allowed:
                return Action(allowed[nearest.id].kind, nearest.id, reason="UNIFORM_COVERAGE")
        return Action(ActionKind.STOP, reason="UNIFORM_TARGETS_DONE")


class RetrievalPolicy:
    """Transcript-similarity timestamp retrieval: look where the transcript matched."""

    name = "retrieval"
    scout_seed = "none"

    def decide(self, obs, legal):
        acts = _looks(legal, {CandidateSource.TRANSCRIPT_RETRIEVAL}, obs)
        if not acts:
            return Action(ActionKind.STOP, reason="NO_RETRIEVED_LEFT")
        by_id = {c.id: c for c in obs.candidates}
        # best-matching window first (explicit rank), then earliest within it
        best = min(acts, key=lambda a: (by_id[a.candidate_id].rank if by_id[a.candidate_id].rank is not None
                                        else 1 << 30, by_id[a.candidate_id].time_s))
        return Action(best.kind, best.candidate_id, reason="RETRIEVAL_NEXT")


class ScoutSimilarityPolicy:
    """Visual-similarity frame selection: highest question-frame similarity first.
    Needs scout signals for every candidate, and pays for them."""

    name = "scout_similarity"
    scout_seed = "all"

    def decide(self, obs, legal):
        acts = [a for a in _looks(legal) if a.kind == ActionKind.LOOK_AT_THIS_MOMENT]
        if not acts:
            return Action(ActionKind.STOP, reason="NO_CANDIDATES_LEFT")
        by_id = {c.id: c for c in obs.candidates}
        best = max(acts, key=lambda a: (obs.scout[a.candidate_id].similarity if a.candidate_id in obs.scout else 0.0,
                                        -by_id[a.candidate_id].time_s))
        return Action(best.kind, best.candidate_id, reason="HIGHEST_SIMILARITY")


# ---------------------------------------------------------------------------
# Utility-scoring controllers (heuristic and learned share the decision rule)
# ---------------------------------------------------------------------------


class _UtilityController:
    """Score every legal action's gain, subtract lambda * measured cost, take
    the best; STOP if nothing clears ``stop_threshold``."""

    name = "utility"
    scout_seed: str | int = 4
    stop_threshold = 0.0

    def __init__(self, cost_model: CostModel = DEFAULT, stop_threshold: float = 0.0) -> None:
        self.cost_model, self.stop_threshold = cost_model, stop_threshold

    def score(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def decide(self, obs, legal):
        g = global_features(obs)
        by_id = {c.id: c for c in obs.candidates}
        acts = [a for a in legal if a.kind != ActionKind.STOP]
        if not acts:
            return Action(ActionKind.STOP, reason="NO_LEGAL_ACTION")
        rows = [expand_features(obs, g) if a.kind == ActionKind.EXPAND_TRANSCRIPT
                else candidate_features(obs, by_id[a.candidate_id], g) for a in acts]
        gains = self.score(np.asarray(rows, dtype=np.float64))
        net = np.array([self.cost_model.net(float(gain), a.cost_ms) for gain, a in zip(gains, acts, strict=True)])
        best = int(np.argmax(net))
        if net[best] <= self.stop_threshold:
            return Action(ActionKind.STOP, predicted_utility=float(gains[best]), reason="NO_ACTION_WORTH_COST")
        a = acts[best]
        return Action(a.kind, a.candidate_id, predicted_utility=float(gains[best]), reason=_reason(a.kind, rows[best]))


def _reason(kind: ActionKind, row: list[float]) -> str:
    """Short decision summary for logs, derived from the features the decision
    used. It is NOT evidence that the model reasons this way."""
    idx = {n: i for i, n in enumerate(CANDIDATE_FEATURES)}
    if kind == ActionKind.EXPAND_TRANSCRIPT:
        return "MORE_CONTEXT"
    if kind == ActionKind.LOOK_ELSEWHERE:
        return "GLOBAL_RESCUE"
    if row[idx["local_speech_coverage"]] < 0.3:
        return "SPEECH_GAP_NEAR_CANDIDATE"
    if row[idx["question_visual_cue"]] > 0:
        return "QUESTION_ASKS_ABOUT_VISUALS"
    return "RELEVANT_WINDOW"


class HeuristicController(_UtilityController):
    """No-training baseline: a transparent hand-set linear gain estimate.

    Intuition encoded (NOT learned): look more when question terms are missing
    from the transcript, when speech is absent near a retrieved candidate, when
    the question asks about something shown, and at novel, sharp, changing
    frames. Weights are round numbers on purpose.
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

    def __init__(self, cost_model: CostModel = DEFAULT, stop_threshold: float = 0.0) -> None:
        super().__init__(cost_model, stop_threshold)
        self.w = np.array([self._W.get(n, 0.0) for n in CANDIDATE_FEATURES])

    def score(self, X):
        return X @ self.w


def make_policy(name: str, cost_model: CostModel = DEFAULT, **kwargs) -> Policy:
    simple = {"transcript_only": TranscriptOnlyPolicy, "uniform": UniformPolicy, "retrieval": RetrievalPolicy,
              "scout_similarity": ScoutSimilarityPolicy}
    if name in simple:
        return simple[name]()
    if name == "heuristic":
        return HeuristicController(cost_model, **kwargs)
    raise ValueError(f"unknown policy {name!r}")
