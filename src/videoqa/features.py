"""Turn a ControllerObservation into numeric features for each possible action.

Everything here is computed from what a deployed system can observe - the
question, the (possibly damaged) transcript, candidate times, scout signals and
the remaining budget.  No labels, no condition names.

Why these features?  The research question is whether the controller can tell
"answer-relevant speech is missing" (TARGETED) from "some unrelated speech is
missing" (CONTROL).  Two observable cues can separate them in principle:

  * question_term_coverage - how many of the question's content words still
    appear anywhere in the transcript.  Targeted damage tends to remove the
    lines that matched the question; control damage usually does not.
  * local speech coverage around a candidate time - a deleted line leaves a
    hole in time.  A hole next to a question-relevant window is informative;
    a hole somewhere unrelated is not.

Whether a learned head actually exploits these cues better with paired training
is exactly what the experiments test.  Nothing here assumes the answer.
"""

from __future__ import annotations

import math

from .damage import MASK_TOKEN
from .retrieval import tokenize
from .schemas import ActionKind, Candidate, CandidateSource, ControllerObservation
from .transcript import speech_coverage_near

# Words that hint a question is about something *shown* rather than said.
_VISUAL_CUES = frozenset(
    "shown show shows slide screen colour color look looks appear appears diagram plot figure "
    "chart graph turn turns visible written board display displayed".split()
)

# Names of the per-candidate feature vector, in order (useful for inspection).
CANDIDATE_FEATURES = (
    "bias",
    "scout_similarity", "scout_visual_change", "scout_quality", "scout_uncertainty",
    "scene_slide", "scene_board", "scene_code", "scene_natural", "scene_blank", "scene_unknown",
    "src_retrieval", "src_uniform", "src_scene_change", "src_rescue",
    "local_speech_coverage",       # speech around this time (observed transcript)
    "local_question_overlap",      # question terms spoken within +/-15 s
    "question_term_coverage",      # global: share of question terms still in transcript
    "masked_fraction",             # global: share of [inaudible] tokens
    "question_visual_cue",         # question mentions shown/slide/colour/...
    "question_has_number_or_option_numbers",
    "frames_remaining_frac", "rounds_remaining_frac",
    "n_looked_frac",
    "min_dist_to_looked",          # temporal novelty vs. frames already acquired
    "expand_action",               # 1 for EXPAND_TRANSCRIPT pseudo-candidate
)


def _question_terms(obs: ControllerObservation) -> set[str]:
    return set(tokenize(obs.question))


def global_features(obs: ControllerObservation) -> dict[str, float]:
    """Features shared by every action at this step."""
    q_terms = _question_terms(obs)
    transcript_terms = set()
    total_words = masked = 0
    for seg in obs.transcript:
        transcript_terms.update(tokenize(seg.text))
        words = seg.text.split()
        total_words += len(words)
        masked += sum(w == MASK_TOKEN for w in words)
    coverage = len(q_terms & transcript_terms) / len(q_terms) if q_terms else 1.0
    opts = " ".join(obs.options or [])
    return {
        "question_term_coverage": coverage,
        "masked_fraction": masked / total_words if total_words else 0.0,
        "question_visual_cue": float(bool(q_terms & _VISUAL_CUES)),
        "question_has_number_or_option_numbers": float(any(c.isdigit() for c in obs.question + opts)),
        "frames_remaining_frac": obs.frames_remaining / max(1, obs.frames_remaining + len(obs.looked_at)),
        "rounds_remaining_frac": min(1.0, obs.rounds_remaining / 6.0),
        "n_looked_frac": min(1.0, len(obs.looked_at) / 8.0),
    }


def candidate_features(obs: ControllerObservation, cand: Candidate, g: dict[str, float] | None = None) -> list[float]:
    """Feature vector (ordered as CANDIDATE_FEATURES) for looking at ``cand``."""
    g = g or global_features(obs)
    sig = obs.scout.get(cand.id)
    scout_vec = sig.as_vector() if sig else [0.5, 0.0, 0.5, 1.0, 0, 0, 0, 0, 0, 1]
    q_terms = _question_terms(obs)
    near = [s for s in obs.transcript if abs(0.5 * (s.start_s + s.end_s) - cand.time_s) <= 15.0]
    near_terms = set(t for s in near for t in tokenize(s.text))
    local_overlap = len(q_terms & near_terms) / len(q_terms) if q_terms else 0.0

    looked_times = [c.time_s for c in obs.candidates if c.id in obs.looked_at]
    if looked_times:
        d = min(abs(cand.time_s - t) for t in looked_times)
        novelty = 1.0 - math.exp(-d / 30.0)
    else:
        novelty = 1.0

    return [
        1.0,
        *scout_vec,
        float(cand.source == CandidateSource.TRANSCRIPT_RETRIEVAL),
        float(cand.source == CandidateSource.UNIFORM),
        float(cand.source == CandidateSource.SCENE_CHANGE),
        float(cand.source == CandidateSource.GLOBAL_RESCUE),
        speech_coverage_near(obs.transcript, cand.time_s, radius_s=10.0),
        local_overlap,
        g["question_term_coverage"],
        g["masked_fraction"],
        g["question_visual_cue"],
        g["question_has_number_or_option_numbers"],
        g["frames_remaining_frac"],
        g["rounds_remaining_frac"],
        g["n_looked_frac"],
        novelty,
        0.0,
    ]


def expand_features(obs: ControllerObservation, g: dict[str, float] | None = None) -> list[float]:
    """Feature vector for the EXPAND_TRANSCRIPT pseudo-action (no scout, no time)."""
    g = g or global_features(obs)
    vec = [0.0] * len(CANDIDATE_FEATURES)
    idx = {n: i for i, n in enumerate(CANDIDATE_FEATURES)}
    vec[idx["bias"]] = 1.0
    for name in ("question_term_coverage", "masked_fraction", "question_visual_cue",
                 "question_has_number_or_option_numbers", "frames_remaining_frac",
                 "rounds_remaining_frac", "n_looked_frac"):
        vec[idx[name]] = g[name]
    vec[idx["expand_action"]] = 1.0
    return vec


def legal_actions(obs: ControllerObservation, rescue_frames_left: int,
                  max_expansions: int = 2) -> list[tuple[ActionKind, str | None]]:
    """Actions allowed under the budget right now (STOP is always legal).

    LOOK_ELSEWHERE is only offered for rescue candidates and only while the
    reserved rescue share of the frame budget is not used up.
    """
    actions: list[tuple[ActionKind, str | None]] = [(ActionKind.STOP, None)]
    if obs.rounds_remaining <= 0:
        return actions
    if obs.frames_remaining > 0:
        for c in obs.candidates:
            if c.id in obs.looked_at:
                continue
            if c.source == CandidateSource.GLOBAL_RESCUE:
                if rescue_frames_left > 0:
                    actions.append((ActionKind.LOOK_ELSEWHERE, c.id))
            else:
                actions.append((ActionKind.LOOK_AT_THIS_MOMENT, c.id))
    if obs.expansions_used < max_expansions:
        actions.append((ActionKind.EXPAND_TRANSCRIPT, None))
    return actions
