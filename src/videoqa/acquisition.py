"""Selective search: text-only preparation plus lazy, metered visual work.

The eager pipeline (``pipeline.prepare``) decodes and scouts EVERY candidate
before the controller decides anything, so a STOP saved only the final
answerer's image processing, never the visual preparation (measured on the
saved LongVideoBench run: tuned controller 1,866 ms vs transcript-only
1,302 ms on clean transcripts). This module is the deployed path instead:

  prepare_text   probe container metadata + transcript search + candidate
                 times. Decodes NOTHING.
  VisualCache    decodes one frame / scouts one candidate only when asked, and
                 meters each call separately (decode, scout).
  run_lazy       the controller loop. Before the loop, the policy's declared
                 scout seed is paid for ("none", "all", or k seed candidates);
                 in the loop, LOOK decodes the chosen frame (probe-and-promote
                 combined in this first implementation: looking at a moment
                 both observes it and adds it to the answerer's evidence).

Consequences the tests check: a transcript-only run, or a controller that
STOPs at once, executes zero decode and zero scout calls; uniform and
retrieval baselines never pay for the scout; only what is looked at is
decoded.

Scout visual change is measured against a fixed frame 1 s earlier (1 s later
at t < 1 s), so a frame's scout signals do not depend on which other
candidates happen to exist. That makes them identical across transcript
conditions, which the paired experiment needs.

Stated simplification: there is no separate PROBE action yet (an observation
that informs later choices without adding evidence). Its value could not be
labelled by immediate answer gain, which is zero by construction; it needs
bounded follow-up rollouts and is left for later.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from PIL import Image

from .answerer import Answerer
from .cost_model import DEFAULT, CostModel
from .costs import CostMeter
from .features import legal_actions_costed
from .frames import VideoInfo, decode_at, probe
from .pipeline import EvidenceState, RunResult, answer_with, apply_action, observation
from .retrieval import Window, bm25_rank, build_windows, generate_candidates
from .schemas import Action, ActionKind, Budget, Candidate, CandidateSource, Frame, ScoutSignals, Transcript
from .scout import Scout, change_between
from .transcript import TranscriptUnit, build_units

NEIGHBOUR_S = 1.0


@dataclasses.dataclass
class LazyPrep:
    """Same fields as ``pipeline.Prepared``, but ``frames`` and ``scout`` start
    empty and are filled only by :class:`VisualCache` on demand."""

    video_path: str
    info: VideoInfo
    question: str
    options: list[str] | None
    transcript: Transcript
    units: list[TranscriptUnit]
    windows: list[Window]
    candidates: list[Candidate]
    frames: dict[str, Frame] = dataclasses.field(default_factory=dict)
    scout: dict[str, ScoutSignals] = dataclasses.field(default_factory=dict)
    frame_size: tuple[int, int] = (0, 0)      # size every decoded frame is resized to


def resized_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    if not max_side or max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_text(video_path: str | Path, transcript: Transcript, question: str, options: list[str] | None,
                 cfg: dict, meter: CostMeter, candidates: list[Candidate] | None = None) -> LazyPrep:
    """Everything that needs no pixels. ``candidates`` may be supplied (the
    controlled assay uses a documented common pool); otherwise they come from
    the OBSERVED transcript only."""
    tcfg, rcfg, ccfg = cfg["transcript"], cfg["retrieval"], cfg["candidates"]
    with meter.stage("probe"):
        info = probe(video_path)                       # container metadata, no decoding
    with meter.stage("retrieval"):
        units = build_units(transcript.segments, tcfg["unit_seconds"], tcfg["min_unit_seconds"])
        ranked = bm25_rank(question + " " + " ".join(options or []), units, rcfg["bm25_top_k"])
        windows = build_windows(ranked, units, rcfg["max_windows"], rcfg["neighbour_expansion"], info.duration_s)
        if candidates is None:
            candidates = generate_candidates(windows, info.duration_s, ccfg["per_window"], ccfg["uniform"],
                                             ccfg["rescue"])
    size = resized_size(info.width, info.height, cfg["answerer"].get("frame_max_side"))
    return LazyPrep(str(video_path), info, question, options, transcript, units, windows, candidates,
                    frame_size=size)


class VisualCache:
    """Lazy, metered frames and scout signals for one prepared question."""

    def __init__(self, prep: LazyPrep, scout: Scout, meter: CostMeter, max_side: int | None,
                 video_id: str = "") -> None:
        self.prep, self.scout_model, self.meter, self.max_side = prep, scout, meter, max_side
        self.video_id = video_id or prep.transcript.video_id
        self._by_id = {c.id: c for c in prep.candidates}

    def ensure_frame(self, cid: str) -> Frame:
        if cid not in self.prep.frames:
            t = self._by_id[cid].time_s
            with self.meter.stage("decode") as rec:
                res = decode_at(self.prep.video_path, [t], max_side=self.max_side, id_prefix=f"{cid}-",
                                video_id=self.video_id)
                rec.decoded_frames = res.frames_visited
            self.prep.frames[cid] = res.frames[0]
        return self.prep.frames[cid]

    def ensure_scout(self, cid: str) -> ScoutSignals:
        if cid not in self.prep.scout:
            t = self._by_id[cid].time_s
            nb_t = t - NEIGHBOUR_S if t >= NEIGHBOUR_S else min(self.prep.info.duration_s - 1e-3, t + NEIGHBOUR_S)
            with self.meter.stage("decode") as rec:
                times = [nb_t] if cid in self.prep.frames else [nb_t, t]
                res = decode_at(self.prep.video_path, times, max_side=self.max_side, id_prefix=f"{cid}-n",
                                video_id=self.video_id)
                rec.decoded_frames = res.frames_visited
            neighbour = res.frames[0]
            if cid not in self.prep.frames:
                self.prep.frames[cid] = res.frames[1]
            frame = self.prep.frames[cid]
            if hasattr(self.scout_model, "ensure_loaded"):
                with self.meter.stage("scout_load"):       # cold one-off cost, reported apart
                    self.scout_model.ensure_loaded()
            with self.meter.stage("scout") as rec:
                sig = self.scout_model.score(self.prep.question, [frame])[frame.id]
                rec.scored_frames = 1
            # Fixed-neighbour visual change: independent of the candidate pool.
            self.prep.scout[cid] = dataclasses.replace(sig, visual_change=change_between(neighbour.image, frame.image))
        return self.prep.scout[cid]


def seed_candidates(prep: LazyPrep, seed: str | int) -> list[str]:
    """Which candidates a policy scouts before deciding. Uses only what is
    observable: retrieval rank first, then uniform coverage in time order."""
    non_rescue = [c for c in prep.candidates if c.source != CandidateSource.GLOBAL_RESCUE]
    if seed == "none" or seed == 0:
        return []
    if seed == "all":
        return [c.id for c in non_rescue]
    ordered = sorted(non_rescue, key=lambda c: (c.rank if c.rank is not None else 1 << 30,
                                                c.source != CandidateSource.TRANSCRIPT_RETRIEVAL, c.time_s))
    return [c.id for c in ordered[: int(seed)]]


def tokens_per_frame(answerer: Answerer, size: tuple[int, int]) -> int:
    """Exact visual tokens of one frame of this video (all share one size)."""
    if hasattr(answerer, "visual_tokens"):
        blank = Frame(id="size-probe", requested_s=0.0, decoded_pts_s=0.0, width=size[0], height=size[1],
                      image=Image.new("RGB", size))
        return int(answerer.visual_tokens([blank]))
    return 256


def run_lazy(prep: LazyPrep, cache: VisualCache, policy, answerer: Answerer, budget: Budget, meter: CostMeter,
             cost_model: CostModel = DEFAULT, max_words: int = 120) -> RunResult:
    """Seed scouting (as the policy declares), the decision loop, one answer."""
    for cid in seed_candidates(prep, getattr(policy, "scout_seed", "none")):
        cache.ensure_scout(cid)

    state = EvidenceState()
    actions: list[Action] = []
    tpf = tokens_per_frame(answerer, prep.frame_size)
    cap = min(budget.max_frames, budget.max_visual_tokens // max(1, tpf))
    rescue_cap = max(0, round(budget.rescue_fraction * budget.max_frames))
    violations: list[str] = []
    expand_ms = cost_model.expand_ms(40)

    for round_idx in range(budget.max_rounds):
        # Best-effort time target (no cancellation mechanism, so not a guarantee).
        if meter.total().elapsed_ms / 1000.0 > 0.8 * budget.max_seconds:
            actions.append(Action(ActionKind.STOP, reason="TIME_RESERVE_FOR_ANSWER"))
            break
        obs = observation(prep, state, budget, budget.max_rounds - round_idx, cap)
        legal = legal_actions_costed(
            obs, rescue_cap - state.rescue_used, budget.max_expansions, tpf,
            budget.max_visual_tokens - tpf * len(state.looked_at),
            lambda cid: cost_model.look_ms(tpf, decoded=cid in prep.frames, scouted=cid in prep.scout),
            expand_ms)
        with meter.stage("controller"):
            action = policy.decide(obs, legal)
        actions.append(action)
        if action.kind == ActionKind.STOP:
            break
        if (action.kind, action.candidate_id) not in {(a.kind, a.candidate_id) for a in legal}:
            violations.append(f"illegal_action:{action.kind.value}:{action.candidate_id}")
            break
        if action.candidate_id is not None:
            cache.ensure_frame(action.candidate_id)
        apply_action(prep, state, action)

    answer, excerpt, frames = answer_with(prep, state, answerer, meter, max_excerpt_words=max_words)
    if len(frames) > budget.max_frames:
        violations.append("max_frames")
    if meter.total().elapsed_ms / 1000.0 > budget.max_seconds:
        violations.append("max_seconds (best-effort target)")
    return RunResult(policy.name, answer, actions, state, excerpt, frames, meter, violations)
