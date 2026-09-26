"""Automatic action-usefulness labels (TRAIN videos only).

For one training question and each transcript condition (clean / targeted /
control):

    1. prepare() the observed transcript -> candidates, frames, scout signals
    2. q_before = quality of the frozen answerer with the current evidence
    3. for every legal action a:  q_after(a) = quality with evidence + a
       measured cost(a) = incremental answer time + visual tokens + decode share
    4. preferred action = argmax (gain - look_cost), or STOP if nothing clears
       the threshold
    5. apply the preferred action and repeat (greedy rollout, <= max_steps)

This is "imitation of the best observed feasible action": every label is an
actually measured before/after answer, never an extrapolated value.  The gold
answer is used only to SCORE answers here; it never enters the controller's
features.

Pairs for the paired loss: at step 0 the evidence state is empty under every
condition, so a row under TARGETED and a row under CONTROL with the same
question, same action kind and same candidate time (within 0.5 s) differ ONLY
in the transcript.  Those are the rows the paired term compares.

Labelling cost is real and reported: every answerer call is counted.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from pathlib import Path

from .answerer import Answerer, answer_quality
from .costs import CostMeter
from .damage import TranscriptTriple
from .features import candidate_features, expand_features, global_features, legal_actions
from .pipeline import EvidenceState, Prepared, answer_with, apply_action, frame_cap, observation, prepare
from .schemas import (
    Action,
    ActionKind,
    Budget,
    CostRecord,
    QAItem,
    TrainingExample,
    TranscriptCondition,
    to_jsonable,
)
from .scout import Scout


@dataclasses.dataclass
class LabelRow:
    """One (state, action) measurement - a row of the utility-head training set."""

    qa_id: str
    video_id: str
    condition: str
    step: int
    action_kind: str
    candidate_id: str | None
    candidate_time_s: float | None
    features: list[float]
    quality_before: float
    quality_after: float
    gain: float
    cost_ms: float
    cost_visual_tokens: int
    preferred: bool


@dataclasses.dataclass
class LabelStats:
    answer_calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0


class _AnswerCache:
    """Memoise answerer calls on the evidence state.  Identical states under the
    same condition are answered once, which roughly halves labelling cost."""

    def __init__(self, answerer: Answerer, stats: LabelStats) -> None:
        self.answerer, self.stats, self._memo = answerer, stats, {}

    def quality(self, prep: Prepared, state: EvidenceState, qa: QAItem, key_prefix: str) -> tuple[float, CostRecord]:
        # The key MUST include the question: evidence states of different
        # questions (or videos) are never interchangeable.
        key = (qa.qa_id, qa.video_id, key_prefix, tuple(sorted(state.looked_at)), state.expansions_used)
        if key in self._memo:
            self.stats.cache_hits += 1
            return self._memo[key]
        meter = CostMeter()
        ans, _, _ = answer_with(prep, state, self.answerer, meter)
        self.stats.answer_calls += 1
        rec = meter.by_stage()["answer"]
        self._memo[key] = (answer_quality(ans, qa), rec)
        return self._memo[key]


def _action_key(kind: ActionKind, cid: str | None) -> str:
    return f"{kind.value}:{cid}" if cid else kind.value


def label_condition(qa: QAItem, video_path: str, condition: TranscriptCondition, triple: TranscriptTriple,
                    cfg: dict, budget: Budget, scout: Scout, cache: _AnswerCache, max_steps: int = 2,
                    look_cost: float = 0.02, stop_threshold: float = 0.05) -> tuple[list[LabelRow], list[TrainingExample]]:
    transcript = triple.by_condition()[condition]
    prep_meter = CostMeter()
    prep = prepare(video_path, transcript, qa.question, qa.options, cfg, scout, prep_meter)
    # Share of the decode+scout cost attributable to one candidate frame.
    stages = prep_meter.by_stage()
    per_frame_ms = (stages["decode"].elapsed_ms + stages["scout"].elapsed_ms) / max(1, len(prep.candidates))

    cap = frame_cap(budget, cache.answerer)
    rescue_cap = max(0, round(budget.rescue_fraction * budget.max_frames))
    state = EvidenceState()
    rows: list[LabelRow] = []
    examples: list[TrainingExample] = []
    by_id = {c.id: c for c in prep.candidates}

    for step in range(max_steps):
        obs = observation(prep, state, budget, budget.max_rounds - step, cap)
        g = global_features(obs)
        q0, base_cost = cache.quality(prep, state, qa, condition.value)
        step_rows: list[LabelRow] = []
        before, after, costs = {}, {}, {}

        for kind, cid in legal_actions(obs, rescue_cap - state.rescue_used, budget.max_expansions):
            if kind == ActionKind.STOP:
                continue
            nxt = copy.deepcopy(state)
            apply_action(prep, nxt, Action(kind, cid))
            q1, cost = cache.quality(prep, nxt, qa, condition.value)
            is_look = kind != ActionKind.EXPAND_TRANSCRIPT
            inc = CostRecord(stage=_action_key(kind, cid),
                             elapsed_ms=max(0.0, cost.elapsed_ms - base_cost.elapsed_ms) + (per_frame_ms if is_look else 0.0),
                             decoded_frames=1 if is_look else 0,
                             visual_tokens=max(0, cost.visual_tokens - base_cost.visual_tokens),
                             text_tokens=max(0, cost.text_tokens - base_cost.text_tokens))
            feats = candidate_features(obs, by_id[cid], g) if cid else expand_features(obs, g)
            key = _action_key(kind, cid)
            before[key], after[key], costs[key] = q0, q1, inc
            step_rows.append(LabelRow(qa.qa_id, qa.video_id, condition.value, step, kind.value, cid,
                                      by_id[cid].time_s if cid else None, feats, q0, q1, q1 - q0,
                                      inc.elapsed_ms, inc.visual_tokens, False))

        # Preferred action: best measured gain net of a per-frame cost penalty.
        best = max(step_rows, key=lambda r: r.gain - (look_cost if r.candidate_id else 0.0), default=None)
        if best is not None and best.gain - (look_cost if best.candidate_id else 0.0) >= stop_threshold:
            best.preferred = True
            preferred = Action(ActionKind(best.action_kind), best.candidate_id, reason="BEST_MEASURED_GAIN")
        else:
            preferred = Action(ActionKind.STOP, reason="NO_MEASURED_GAIN")
        rows.extend(step_rows)
        examples.append(TrainingExample(
            video_id=qa.video_id, qa_id=qa.qa_id, question=qa.question, options=qa.options,
            gold_answer=qa.gold_answer, timed_transcript=transcript.segments, transcript_condition=condition,
            damage=triple.records[condition], candidate_timestamps=prep.candidates,
            frozen_scout_signals_for_each_candidate=prep.scout, available_budget=budget,
            answer_quality_before_each_action=before, answer_quality_after_each_action=after,
            measured_action_cost=costs, preferred_action_or_stop=preferred,
            source_dataset=qa.source_dataset, source_split=qa.source_split,
            provenance_and_license=qa.provenance_and_license))
        if preferred.kind == ActionKind.STOP:
            break
        apply_action(prep, state, preferred)
    return rows, examples


def pair_rows(rows: list[LabelRow]) -> list[tuple[int, int]]:
    """Indices (targeted_row, control_row) for step-0 rows of the same question,
    same action kind and same candidate time (+/- 0.5 s)."""
    ctrl: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        if r.step == 0 and r.condition == TranscriptCondition.CONTROL_DAMAGE.value:
            ctrl.setdefault((r.qa_id, r.action_kind), []).append(i)
    pairs = []
    for i, r in enumerate(rows):
        if r.step != 0 or r.condition != TranscriptCondition.TARGETED_DAMAGE.value:
            continue
        for j in ctrl.get((r.qa_id, r.action_kind), []):
            c = rows[j]
            same_time = (r.candidate_time_s is None and c.candidate_time_s is None) or (
                r.candidate_time_s is not None and c.candidate_time_s is not None
                and abs(r.candidate_time_s - c.candidate_time_s) <= 0.5)
            if same_time:
                pairs.append((i, j))
                break
    return pairs


def write_labels(out_dir: str | Path, rows: list[LabelRow], examples: list[TrainingExample],
                 manifest: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
    with (out / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(to_jsonable(e)) + "\n")
    pairs = pair_rows(rows)
    (out / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    manifest = {**manifest, "n_rows": len(rows), "n_examples": len(examples), "n_pairs": len(pairs)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def read_rows(label_dir: str | Path) -> tuple[list[LabelRow], list[tuple[int, int]]]:
    d = Path(label_dir)
    rows = [LabelRow(**json.loads(line)) for line in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    pairs = [tuple(p) for p in json.loads((d / "pairs.json").read_text(encoding="utf-8"))]
    return rows, pairs


def label_items(items: list[tuple[QAItem, str, TranscriptTriple]], cfg: dict, budget: Budget, scout: Scout,
                answerer: Answerer, **kw) -> tuple[list[LabelRow], list[TrainingExample], LabelStats]:
    """Label every (qa, video_path, triple) under all three conditions."""
    stats = LabelStats()
    cache = _AnswerCache(answerer, stats)
    t0 = time.perf_counter()
    rows, examples = [], []
    for qa, path, triple in items:
        if qa.source_split not in ("train", "unassigned"):
            raise ValueError(f"refusing to label {qa.qa_id} from split {qa.source_split!r}: train only")
        for cond in TranscriptCondition:
            r, e = label_condition(qa, path, cond, triple, cfg, budget, scout, cache, **kw)
            rows += r
            examples += e
    stats.seconds = time.perf_counter() - t0
    return rows, examples, stats
