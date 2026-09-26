"""End-to-end pipeline for one question on one video.

    prepare()      transcript -> units -> BM25 windows -> candidates
                   -> decode candidate frames -> scout signals        (metered)
    run_policy()   controller loop under hard caps -> selected frames
                   -> verbatim excerpt -> frozen answerer (once)       (metered)
    save_run()     result.json (answer, citations, evidence ledger, action log,
                   cost trace) + contact_sheet.png of the selected frames

Cost honesty: every candidate frame decoded for the scout is charged, even if
the controller never looks at it.  "Visual tokens sent to the answerer" is
only part of the bill; decoded and scored frames are reported separately.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .answerer import Answerer, AnswerRequest
from .controller import Policy
from .costs import CostMeter
from .frames import VideoInfo, decode_at, probe
from .packing import pack_excerpt
from .retrieval import Window, bm25_rank, build_windows, generate_candidates
from .schemas import (
    Action,
    ActionKind,
    Answer,
    Budget,
    Candidate,
    CandidateSource,
    ControllerObservation,
    Frame,
    ScoutSignals,
    Transcript,
    to_jsonable,
)
from .scout import Scout
from .transcript import TranscriptUnit, build_units

# ---------------------------------------------------------------------------
# Preparation (query-dependent, but independent of the controller policy)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Prepared:
    video_path: str
    info: VideoInfo
    question: str
    options: list[str] | None
    transcript: Transcript                  # the OBSERVED (possibly damaged) transcript
    units: list[TranscriptUnit]
    windows: list[Window]
    candidates: list[Candidate]
    frames: dict[str, Frame]                # candidate_id -> decoded frame
    scout: dict[str, ScoutSignals]          # candidate_id -> signals


def prepare(video_path: str | Path, transcript: Transcript, question: str, options: list[str] | None,
            cfg: dict, scout: Scout, meter: CostMeter) -> Prepared:
    """Retrieval, candidate generation, decoding and scouting for one question."""
    tcfg, rcfg, ccfg = cfg["transcript"], cfg["retrieval"], cfg["candidates"]
    with meter.stage("probe"):
        info = probe(video_path)

    with meter.stage("retrieval"):
        units = build_units(transcript.segments, tcfg["unit_seconds"], tcfg["min_unit_seconds"])
        # Options are part of the question for retrieval: they often carry the key terms.
        query = question + " " + " ".join(options or [])
        ranked = bm25_rank(query, units, rcfg["bm25_top_k"])
        windows = build_windows(ranked, units, rcfg["max_windows"], rcfg["neighbour_expansion"], info.duration_s)
        candidates = generate_candidates(windows, info.duration_s, ccfg["per_window"], ccfg["uniform"], ccfg["rescue"])

    # Decode every candidate ONCE at answerer resolution; the scout downsizes
    # internally.  All of these frames are charged here.
    with meter.stage("decode") as rec:
        res = decode_at(video_path, [c.time_s for c in candidates],
                        max_side=cfg["answerer"].get("frame_max_side"), id_prefix="f",
                        video_id=transcript.video_id)
        frames = {c.id: f for c, f in zip(candidates, res.frames, strict=True)}
        rec.decoded_frames = res.frames_visited

    with meter.stage("scout") as rec:
        by_frame = scout.score(question, list(frames.values()))
        signals = {cid: by_frame[f.id] for cid, f in frames.items()}
        rec.scored_frames = len(frames)

    return Prepared(str(video_path), info, question, options, transcript, units, windows, candidates, frames, signals)


# ---------------------------------------------------------------------------
# Evidence state and answering
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvidenceState:
    looked_at: list[str] = dataclasses.field(default_factory=list)       # candidate ids, acquisition order
    extra_segment_ids: list[str] = dataclasses.field(default_factory=list)
    expansions_used: int = 0
    rescue_used: int = 0


def expand_transcript(prep: Prepared, state: EvidenceState) -> None:
    """EXPAND_TRANSCRIPT: add the unit just before and just after each window."""
    index = {u.id: i for i, u in enumerate(prep.units)}
    for w in prep.windows:
        lo, hi = index[w.unit_ids[0]] - 1 - state.expansions_used, index[w.unit_ids[-1]] + 1 + state.expansions_used
        for j in (lo, hi):
            if 0 <= j < len(prep.units):
                state.extra_segment_ids.extend(prep.units[j].segment_ids)
    state.expansions_used += 1


def answer_with(prep: Prepared, state: EvidenceState, answerer: Answerer, meter: CostMeter,
                max_excerpt_words: int = 120, stage: str = "answer") -> tuple[Answer, list, list[Frame]]:
    """Pack evidence for ``state`` and call the frozen answerer once."""
    unit_to_segments = {u.id: u.segment_ids for u in prep.units}
    excerpt = pack_excerpt(prep.question, prep.transcript.segments, prep.windows, unit_to_segments,
                           max_words=max_excerpt_words, extra_segment_ids=state.extra_segment_ids)
    frames = sorted((prep.frames[c] for c in state.looked_at), key=lambda f: f.decoded_pts_s)
    with meter.stage(stage) as rec:
        answer, usage = answerer.answer(AnswerRequest(prep.question, prep.options, excerpt, frames))
        rec.visual_tokens, rec.text_tokens, rec.selected_frames = usage.visual_tokens, usage.text_tokens, len(frames)
    return answer, excerpt, frames


# ---------------------------------------------------------------------------
# Controller loop
# ---------------------------------------------------------------------------


def observation(prep: Prepared, state: EvidenceState, budget: Budget, rounds_left: int,
                frames_cap: int) -> ControllerObservation:
    """Assemble exactly what the controller may see (no labels)."""
    return ControllerObservation(
        question=prep.question, options=prep.options, transcript=prep.transcript.segments,
        candidates=prep.candidates, scout=prep.scout, looked_at=list(state.looked_at),
        expansions_used=state.expansions_used, frames_remaining=max(0, frames_cap - len(state.looked_at)),
        rounds_remaining=rounds_left, video_duration_s=prep.info.duration_s,
    )


def frame_cap(budget: Budget, answerer: Answerer) -> int:
    """Max frames allowed by BOTH the frame cap and the visual-token cap."""
    per_frame = max(1, int(getattr(answerer, "visual_tokens_per_frame", 256)))
    return min(budget.max_frames, budget.max_visual_tokens // per_frame)


def apply_action(prep: Prepared, state: EvidenceState, action: Action) -> None:
    if action.kind in (ActionKind.LOOK_AT_THIS_MOMENT, ActionKind.LOOK_ELSEWHERE):
        if action.candidate_id in state.looked_at:
            raise ValueError(f"duplicate acquisition of {action.candidate_id}")  # never charge twice
        state.looked_at.append(action.candidate_id)
        if action.kind == ActionKind.LOOK_ELSEWHERE:
            state.rescue_used += 1
    elif action.kind == ActionKind.EXPAND_TRANSCRIPT:
        expand_transcript(prep, state)


@dataclasses.dataclass
class RunResult:
    policy: str
    answer: Answer
    actions: list[Action]
    state: EvidenceState
    excerpt: list
    frames: list[Frame]
    meter: CostMeter
    cap_violations: list[str]


def run_policy(prep: Prepared, policy: Policy, answerer: Answerer, budget: Budget, meter: CostMeter) -> RunResult:
    """Run the acquisition loop, then answer once."""
    state = EvidenceState()
    actions: list[Action] = []
    cap = frame_cap(budget, answerer)
    rescue_cap = max(0, round(budget.rescue_fraction * budget.max_frames))
    violations: list[str] = []

    for round_idx in range(budget.max_rounds):
        # Reserve time for the final answer call: stop acquiring at 80% of the time cap.
        if meter.total().elapsed_ms / 1000.0 > 0.8 * budget.max_seconds:
            actions.append(Action(ActionKind.STOP, reason="TIME_RESERVE_FOR_ANSWER"))
            break
        obs = observation(prep, state, budget, budget.max_rounds - round_idx, cap)
        with meter.stage("controller"):
            action = policy.decide(obs, rescue_cap - state.rescue_used, budget.max_expansions)
        actions.append(action)
        if action.kind == ActionKind.STOP:
            break
        apply_action(prep, state, action)

    answer, excerpt, frames = answer_with(prep, state, answerer, meter)

    total = meter.total()
    if len(frames) > budget.max_frames:
        violations.append("max_frames")
    if total.elapsed_ms / 1000.0 > budget.max_seconds:
        violations.append("max_seconds")
    if meter.by_stage().get("answer") and meter.by_stage()["answer"].visual_tokens > budget.max_visual_tokens:
        violations.append("max_visual_tokens")
    return RunResult(policy.name, answer, actions, state, excerpt, frames, meter, violations)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def evidence_ledger(prep: Prepared, result: RunResult) -> list[dict]:
    """Text spans + real frames with times and provenance, for auditing."""
    ledger = [{"type": "text", "segment_id": s.id, "start_s": s.start_s, "end_s": s.end_s, "text": s.text}
              for s in result.excerpt]
    by_frame = {f.id: cid for cid, f in prep.frames.items()}
    cands = {c.id: c for c in prep.candidates}
    for f in result.frames:
        c = cands[by_frame[f.id]]
        ledger.append({"type": "frame", "candidate_id": c.id, "source": c.source.value,
                       "requested_s": f.requested_s, "decoded_pts_s": f.decoded_pts_s,
                       "width": f.width, "height": f.height})
    return ledger


def contact_sheet(frames: list[Frame], thumb: int = 240) -> Image.Image:
    """Grid of selected frames, each labelled with its true timestamp."""
    if not frames:
        return Image.new("RGB", (thumb, 40), "white")
    cols = min(4, len(frames))
    rows = (len(frames) + cols - 1) // cols
    th = int(thumb * frames[0].height / frames[0].width)
    sheet = Image.new("RGB", (cols * thumb, rows * (th + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for i, f in enumerate(frames):
        x, y = (i % cols) * thumb, (i // cols) * (th + 20)
        sheet.paste(f.image.resize((thumb, th)), (x, y))
        draw.text((x + 4, y + th + 2), f"{f.decoded_pts_s:.2f}s", fill="black", font=font)
    return sheet


def save_run(prep: Prepared, result: RunResult, out_dir: str | Path, extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": result.policy,
        "question": prep.question,
        "options": prep.options,
        "answer": to_jsonable(result.answer),
        "evidence": evidence_ledger(prep, result),
        "actions": to_jsonable(result.actions),
        "candidates": to_jsonable(prep.candidates),
        "scout": to_jsonable(prep.scout),
        "cost_by_stage": to_jsonable(result.meter.by_stage()),
        "cost_total": to_jsonable(result.meter.total()),
        "cap_violations": result.cap_violations,
        "transcript_source": prep.transcript.source,
        **(extra or {}),
    }
    (out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    contact_sheet(result.frames).save(out / "contact_sheet.png")
    return out / "result.json"


# Candidate sources that count as "rescue" for reporting.
RESCUE_SOURCES = {CandidateSource.GLOBAL_RESCUE}
