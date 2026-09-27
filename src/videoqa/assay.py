"""Controlled paired assay labels (version "assay-v2").

This replaces the pairing in labels.py for the paired-training experiment.
labels.py is kept unchanged so the historical label sets and results
reproduce. The defects it fixes (docs/corrections/2026-09-28.md):

  * pairs were matched on (question, action kind, candidate time +/- 0.5 s)
    between two SEPARATELY prepared conditions, so candidate pools, scout
    neighbours and histories could differ inside a "pair";
  * visual_change depended on which other candidates existed;
  * EXPAND rows were paired as if they were visual actions;
  * labels used a cost that inference did not use.

What one question produces here
-------------------------------
1. Conditions: CLEAN / TARGETED / CONTROL transcripts from build_triple
   (severity-matched effective damage; drop reason recorded otherwise).
2. A COMMON candidate pool: the union of the times each condition's own
   retrieval proposes, plus the uniform and rescue grids, merged at 1 s. It is
   written to pool.jsonl with, per condition, whether the time falls in that
   condition's retrieved windows and at what rank. Every condition sees every
   pool time; only the observable transcript-derived flags differ.
3. Shared visual state: each pool frame is decoded ONCE (exact PTS and pixel
   digest) and scored by the frozen scout once, against a fixed 1 s neighbour.
   Which frames a condition has scouted follows the deployed rule
   (acquisition.seed_candidates with the learned controller's seed size), so
   training features match inference features.
4. Histories: h0 = nothing looked at; h1 = the pool frame nearest the video
   midpoint (a transcript-independent rule), identical in all conditions.
5. For every legal action in every (condition, history), the frozen answerer
   is called with and without the action; raw answer text, parsed option,
   abstention flags and quality are all saved. Cost is the SAME costed
   legality run_lazy uses (cost_model + exact visual tokens).
6. Canonical action key = hash(video, decoded PTS, pixel digest, frame size,
   action class, history fingerprint). A triplet is the three condition rows
   with one key; its gains are what V1/V2 training compares. EXPAND rows are
   kept for the unpaired term but never enter visual triplets.

Splits: only experiment_split "train" (fitting) and "dev" (calibration /
lambda tuning) may be labelled; "test" is refused. No relabelling to
"unassigned". check_media_overlap refuses runs where one media file appears
under two experiment splits.

The gold answer and damage records are used only to SCORE answers and to
document the assay; they never reach features (features.py) or any serialized
controller input.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from .acquisition import LazyPrep, VisualCache, prepare_text, seed_candidates, tokens_per_frame
from .answerer import Answerer, answer_quality
from .cost_model import DEFAULT, CostModel
from .costs import CostMeter
from .damage import TranscriptTriple, build_triple
from .features import candidate_features, expand_features, global_features, legal_actions_costed
from .frames import decode_at
from .pipeline import EvidenceState, answer_with, apply_action, observation
from .retrieval import Window
from .schemas import (
    Action,
    ActionKind,
    Budget,
    Candidate,
    CandidateSource,
    DamageType,
    QAItem,
    Transcript,
    TranscriptCondition,
    to_jsonable,
)
from .scout import Scout
from .serialize import serialize

ASSAY_VERSION = "assay-v2"
CONDITIONS = (TranscriptCondition.CLEAN, TranscriptCondition.TARGETED_DAMAGE, TranscriptCondition.CONTROL_DAMAGE)
LABELLABLE_SPLITS = ("train", "dev")
MERGE_S = 1.0
LEARNED_SCOUT_SEED = 4        # must equal controller._UtilityController.scout_seed


# ---------------------------------------------------------------------------
# Split and media guards
# ---------------------------------------------------------------------------


def check_split(qa: QAItem) -> None:
    if qa.experiment_split not in LABELLABLE_SPLITS:
        raise ValueError(f"refusing to label {qa.qa_id}: experiment_split={qa.experiment_split!r} "
                         f"(only {LABELLABLE_SPLITS}; test is locked)")


def media_fingerprint(path: str | Path, chunk: int = 4 << 20) -> str:
    """sha256 of (size, first 4 MB, last 4 MB). A full hash of every video is
    slow; this catches re-encoded copies only if bytes match, which is the
    leakage case that matters here (the same file under two names)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with p.open("rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(chunk, size - chunk))
            h.update(fh.read(chunk))
    return h.hexdigest()


def check_media_overlap(items: list[tuple[QAItem, str]]) -> dict[str, list[str]]:
    """Raise if one media fingerprint appears under two experiment splits.
    Returns fingerprint -> splits for the manifest."""
    seen: dict[str, set[str]] = defaultdict(set)
    for qa, path in items:
        seen[media_fingerprint(path)].add(qa.experiment_split)
    clash = {h: sorted(s) for h, s in seen.items() if len(s) > 1}
    if clash:
        raise ValueError(f"media shared across experiment splits: {clash}")
    return {h: sorted(s) for h, s in seen.items()}


# ---------------------------------------------------------------------------
# Common pool
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PoolEntry:
    id: str
    time_s: float
    proposed_by: list[str]                 # "clean:transcript_retrieval", "all:uniform", ...
    source: dict[str, str]                 # condition -> CandidateSource value in that condition's view
    rank: dict[str, int | None]            # condition -> retrieval rank in that condition


def _window_hit(t: float, windows: list[Window]) -> tuple[Window | None, int | None]:
    rank_of = {w.id: r for r, w in enumerate(sorted(windows, key=lambda w: -w.score))}
    hits = [w for w in windows if w.start_s - 1e-6 <= t <= w.end_s + 1e-6]
    if not hits:
        return None, None
    best = min(hits, key=lambda w: rank_of[w.id])
    return best, rank_of[best.id]


def build_common_pool(preps: dict[TranscriptCondition, LazyPrep]) -> tuple[list[PoolEntry], dict[str, list[Candidate]]]:
    """Union of every condition's proposed times, merged at MERGE_S.

    Retrieval proposals are merged first (in condition order) so a time the
    transcript points at is kept at its exact position; uniform and rescue
    grids fill the rest. Returns the documented pool and, per condition, the
    Candidate view of it (source and rank as that condition's transcript
    would give them).
    """
    proposals: list[tuple[int, int, float, str]] = []
    prio = {CandidateSource.TRANSCRIPT_RETRIEVAL: 0, CandidateSource.UNIFORM: 1, CandidateSource.SCENE_CHANGE: 2,
            CandidateSource.GLOBAL_RESCUE: 3}
    for ci, cond in enumerate(CONDITIONS):
        for c in preps[cond].candidates:
            proposals.append((prio[c.source], ci, c.time_s, f"{cond.value}:{c.source.value}"))
    kept: list[tuple[float, list[str]]] = []
    for _, _, t, tag in sorted(proposals):
        near = [k for k in kept if abs(k[0] - t) < MERGE_S]
        if near:
            if tag not in near[0][1]:
                near[0][1].append(tag)
        else:
            kept.append((t, [tag]))
    kept.sort(key=lambda k: k[0])

    uniform_times = {c.time_s for c in preps[CONDITIONS[0]].candidates if c.source == CandidateSource.UNIFORM}
    pool, views = [], {cond.value: [] for cond in CONDITIONS}
    for i, (t, tags) in enumerate(kept):
        pid = f"p{i:03d}"
        src, rk = {}, {}
        for cond in CONDITIONS:
            w, r = _window_hit(t, preps[cond].windows)
            if w is not None:
                s = CandidateSource.TRANSCRIPT_RETRIEVAL
            elif any(abs(t - u) < MERGE_S for u in uniform_times):
                s = CandidateSource.UNIFORM
            else:
                s = CandidateSource.GLOBAL_RESCUE      # outside this condition's windows
            src[cond.value], rk[cond.value] = s.value, r
            views[cond.value].append(Candidate(pid, t, s, w.id if w else None, r))
        pool.append(PoolEntry(pid, t, tags, src, rk))
    return pool, views


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AssayRow:
    assay_version: str
    qa_id: str
    video_id: str
    official_split: str
    experiment_split: str
    condition: str
    history_id: str                        # "h0" | "h1"
    history_fp: str
    action_kind: str
    action_class: str                      # "look" | "expand"
    candidate_id: str | None
    candidate_time_s: float | None
    decoded_pts_s: float | None
    pixel_digest: str | None
    frame_size: list[int] | None
    canonical_key: str
    features: list[float]
    scouted: bool
    source: str | None
    quality_before: float
    quality_after: float
    gain: float
    cost_ms: float
    visual_tokens: int
    answer_before: str
    answer_after: str
    option_before: int | None
    option_after: int | None
    abstained_after: bool
    forced_choice_after: bool
    expand_segment_ids: list[str]
    obs_text: str = ""                     # serialize.serialize(obs, action): the LLM controller's input


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def history_fingerprint(frames: dict, looked: list[str], expansions: int) -> str:
    return _sha([sorted(frames[c].digest for c in looked), expansions])


def canonical_key(video_id: str, frame, action_class: str, history_fp: str) -> str:
    if frame is None:
        return _sha([video_id, action_class, history_fp])
    return _sha([video_id, round(frame.decoded_pts_s, 3), frame.digest, [frame.width, frame.height],
                 action_class, history_fp])


@dataclasses.dataclass
class AssayStats:
    answer_calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    drops: Counter = dataclasses.field(default_factory=Counter)


class _Answers:
    """Memoised frozen-answerer calls keyed on (question, condition, evidence)."""

    def __init__(self, answerer: Answerer, stats: AssayStats) -> None:
        self.answerer, self.stats, self._memo = answerer, stats, {}

    def get(self, prep: LazyPrep, state: EvidenceState, qa: QAItem, cond: str):
        key = (qa.qa_id, cond, tuple(sorted(state.looked_at)), tuple(state.extra_segment_ids))
        if key in self._memo:
            self.stats.cache_hits += 1
            return self._memo[key]
        ans, _, _ = answer_with(prep, state, self.answerer, CostMeter())
        self.stats.answer_calls += 1
        self._memo[key] = (ans, answer_quality(ans, qa))
        return self._memo[key]


def label_question(qa: QAItem, video_path: str, triple: TranscriptTriple, cfg: dict, budget: Budget,
                   scout: Scout, answers: _Answers, cost_model: CostModel = DEFAULT) -> tuple[list[AssayRow], list[PoolEntry]]:
    check_split(qa)
    max_side = cfg["answerer"].get("frame_max_side")
    meter = CostMeter()                                  # labelling cost is not a policy cost
    transcripts = triple.by_condition()
    preps = {c: prepare_text(video_path, transcripts[c], qa.question, qa.options, cfg, meter) for c in CONDITIONS}
    pool, views = build_common_pool(preps)

    # Shared visual state: decode every pool frame once, score once.
    master = dataclasses.replace(preps[CONDITIONS[0]], candidates=views[CONDITIONS[0].value], frames={}, scout={})
    res = decode_at(video_path, [p.time_s for p in pool], max_side=max_side, id_prefix="p", video_id=qa.video_id)
    master.frames.update({p.id: f for p, f in zip(pool, res.frames, strict=True)})
    cache = VisualCache(master, scout, meter, max_side, qa.video_id)

    views_prep: dict[str, LazyPrep] = {}
    scouted_by: dict[str, set[str]] = {}
    for cond in CONDITIONS:
        v = dataclasses.replace(preps[cond], candidates=views[cond.value], frames=master.frames, scout={})
        seeds = seed_candidates(v, LEARNED_SCOUT_SEED)
        v.scout = {cid: cache.ensure_scout(cid) for cid in seeds}
        views_prep[cond.value], scouted_by[cond.value] = v, set(seeds)

    tpf = tokens_per_frame(answers.answerer, master.frame_size)
    cap = min(budget.max_frames, budget.max_visual_tokens // max(1, tpf))
    rescue_cap = max(0, round(budget.rescue_fraction * budget.max_frames))
    mid = master.info.duration_s / 2
    h1_id = min(pool, key=lambda p: (abs(p.time_s - mid), p.time_s)).id
    histories = {"h0": [], "h1": [h1_id]}

    rows: list[AssayRow] = []
    for cond in CONDITIONS:
        prep, scouted = views_prep[cond.value], scouted_by[cond.value]
        by_id = {c.id: c for c in prep.candidates}
        for hid, looked in histories.items():
            state = EvidenceState()
            for cid in looked:
                # Applied as a plain LOOK in every condition so the history
                # (including the rescue allowance left) is identical.
                apply_action(prep, state, Action(ActionKind.LOOK_AT_THIS_MOMENT, cid))
            hfp = history_fingerprint(prep.frames, state.looked_at, state.expansions_used)
            obs = observation(prep, state, budget, budget.max_rounds - len(looked), cap)
            g = global_features(obs)
            decoded = scouted | set(state.looked_at)
            legal = legal_actions_costed(
                obs, rescue_cap - state.rescue_used, budget.max_expansions, tpf,
                budget.max_visual_tokens - tpf * len(state.looked_at),
                lambda cid, d=decoded: cost_model.look_ms(tpf, decoded=cid in d, scouted=cid in scouted),
                cost_model.expand_ms(40))
            a0, q0 = answers.get(prep, state, qa, cond.value)
            for act in legal:
                if act.kind == ActionKind.STOP:
                    continue
                nxt = copy.deepcopy(state)
                apply_action(prep, nxt, Action(act.kind, act.candidate_id))
                a1, q1 = answers.get(prep, nxt, qa, cond.value)
                is_look = act.candidate_id is not None
                frame = prep.frames[act.candidate_id] if is_look else None
                cls = "look" if is_look else "expand"
                feats = candidate_features(obs, by_id[act.candidate_id], g) if is_look else expand_features(obs, g)
                rows.append(AssayRow(
                    ASSAY_VERSION, qa.qa_id, qa.video_id, qa.official_split, qa.experiment_split, cond.value,
                    hid, hfp, act.kind.value, cls, act.candidate_id,
                    by_id[act.candidate_id].time_s if is_look else None,
                    frame.decoded_pts_s if frame else None, frame.digest if frame else None,
                    [frame.width, frame.height] if frame else None,
                    canonical_key(qa.video_id, frame, cls, hfp), feats,
                    bool(is_look and act.candidate_id in scouted),
                    by_id[act.candidate_id].source.value if is_look else None,
                    q0, q1, q1 - q0, act.cost_ms, act.visual_tokens,
                    a0.text, a1.text, a0.option_index, a1.option_index, a1.abstained, a1.forced_choice,
                    [s for s in nxt.extra_segment_ids if s not in state.extra_segment_ids],
                    serialize(obs, act.kind, by_id[act.candidate_id] if is_look else None)))
    return rows, pool


# ---------------------------------------------------------------------------
# Triplets and validation
# ---------------------------------------------------------------------------


def build_triplets(rows: list[AssayRow]) -> tuple[list[dict], Counter]:
    """Exact-key visual triplets: {clean, targeted, control} row indices.
    EXPAND never pairs. Returns triplets and drop counts by reason."""
    groups: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        if r.action_class == "look":
            groups[(r.qa_id, r.canonical_key)][r.condition].append(i)
    out, drops = [], Counter()
    for (qa_id, key), by_cond in groups.items():
        missing = [c.value for c in CONDITIONS if c.value not in by_cond]
        if missing:
            drops[f"missing_{'+'.join(missing)}"] += 1
            continue
        if any(len(v) > 1 for v in by_cond.values()):
            drops["duplicate_key"] += 1
            continue
        idx = {c.value: by_cond[c.value][0] for c in CONDITIONS}
        rs = [rows[i] for i in idx.values()]
        # By construction these must hold; a failure is a bug, not a drop.
        assert len({(r.decoded_pts_s, r.pixel_digest, r.history_fp) for r in rs}) == 1, (qa_id, key)
        out.append({"qa_id": qa_id, "canonical_key": key, "history_id": rs[0].history_id,
                    "clean": idx["clean"], "targeted": idx["targeted_damage"], "control": idx["control_damage"]})
    return out, drops


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def label_assay(items: list[tuple[QAItem, str, Transcript]], cfg: dict, budget: Budget, scout: Scout,
                answerer: Answerer, cost_model: CostModel = DEFAULT, dtype: DamageType = DamageType.DELETE,
                seed: int = 0, relevance_methods: frozenset[str] | None = None, log=None):
    stats = AssayStats()
    answers = _Answers(answerer, stats)
    t0 = time.perf_counter()
    rows: list[AssayRow] = []
    pools, questions = {}, []
    for n, (qa, path, transcript) in enumerate(items):
        check_split(qa)
        triple, reason = build_triple(qa, transcript, dtype=dtype, seed=seed, relevance_methods=relevance_methods)
        if triple is None:
            stats.drops[f"triple:{reason}"] += 1
            continue
        r, pool = label_question(qa, path, triple, cfg, budget, scout, answers, cost_model)
        rows += r
        pools[qa.qa_id] = [dataclasses.asdict(p) for p in pool]
        recs = triple.records
        questions.append({
            "qa_id": qa.qa_id, "video_id": qa.video_id, "evidence_type": qa.evidence_type,
            "relevance_method": triple.relevance_method, "pool_size": len(pool),
            "targeted": to_jsonable(recs[TranscriptCondition.TARGETED_DAMAGE]),
            "control": to_jsonable(recs[TranscriptCondition.CONTROL_DAMAGE]),
        })
        if log:
            log(f"[{n + 1}/{len(items)}] {qa.qa_id}: {len(r)} rows, pool {len(pool)}, "
                f"calls {stats.answer_calls}, {time.perf_counter() - t0:.0f}s")
    stats.seconds = time.perf_counter() - t0
    return rows, pools, questions, stats


def write_assay(out_dir: str | Path, rows: list[AssayRow], pools: dict, questions: list[dict], stats: AssayStats,
                manifest: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
    (out / "pools.json").write_text(json.dumps(pools, indent=1), encoding="utf-8")
    (out / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    triplets, drops = build_triplets(rows)
    (out / "triplets.json").write_text(json.dumps(triplets), encoding="utf-8")
    t_gain = [rows[t["targeted"]].gain - rows[t["control"]].gain for t in triplets]
    report = {
        "assay_version": ASSAY_VERSION, "git_commit": _git_commit(),
        "questions_labelled": len(questions), "rows": len(rows),
        "look_rows": sum(r.action_class == "look" for r in rows),
        "expand_rows": sum(r.action_class == "expand" for r in rows),
        "triplets": len(triplets), "triplet_drops": dict(drops), "question_drops": dict(stats.drops),
        "triplets_with_nonzero_targeted_minus_control": sum(d != 0 for d in t_gain),
        "mean_targeted_minus_control_gain": (sum(t_gain) / len(t_gain)) if t_gain else None,
        "answer_calls": stats.answer_calls, "cache_hits": stats.cache_hits,
        "labelling_seconds": round(stats.seconds, 1),
        **manifest,
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_assay(label_dir: str | Path) -> tuple[list[AssayRow], list[dict]]:
    d = Path(label_dir)
    rows = [AssayRow(**json.loads(x)) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return rows, json.loads((d / "triplets.json").read_text(encoding="utf-8"))
