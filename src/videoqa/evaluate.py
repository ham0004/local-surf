"""Evaluate policies across transcript conditions, with cost and hypothesis metrics.

Conditions per held-out question:
    clean, targeted_damage, control_damage   (from make_triple; gold evidence is
                                              used only to BUILD the damage)
    asr_noise                                (untargeted synthetic noise; NOT
                                              a substitute for natural ASR errors)

Per (policy, condition) we report mean answer quality, frames acquired, visual
tokens, decoded frames and wall time.  Per policy we report the hypothesis
metrics:

    targeted_response  = frames(targeted) - frames(clean)     want: > 0 when it helps
    control_overspend  = frames(control)  - frames(clean)     want: ~ 0
    selectivity        = targeted_response - control_overspend

Uncertainty: paired cluster bootstrap over VIDEOS (questions of one video are
resampled together), giving 95% intervals for policy differences.

Cost fairness (changed 2026-09-28): every (policy, condition) run executes
the lazy selective-search path (acquisition.py) on its own: its own text
preparation, its own frame cache and its own meter. A policy is therefore
charged exactly the decode and scout work it actually caused; nothing is
shared or charged back. Policy order is rotated per question so no policy
always runs first (file-system cache). ``total_ms`` includes one-off model
loading if it happened inside the run; ``warm_ms`` excludes every
``*_load`` stage and is the per-query cost to compare. The old eager path
(decode + scout everything, then charge back) is kept with ``eager=True``
only to reproduce historical results.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .answerer import Answerer, answer_quality
from .controller import Policy, TranscriptOnlyPolicy
from .costs import CostMeter
from .damage import asr_style_noise, make_triple
from .acquisition import VisualCache, prepare_text, run_lazy
from .cost_model import DEFAULT, CostModel
from .pipeline import prepare, run_policy
from .schemas import Budget, DamageType, QAItem, Transcript
from .scout import Scout

CONDITIONS = ("clean", "targeted_damage", "control_damage", "asr_noise")
_TEXT_ONLY_STAGES = {"probe", "retrieval"}


@dataclasses.dataclass
class EvalRecord:
    qa_id: str
    video_id: str
    policy: str
    condition: str
    quality: float
    frames: int
    visual_tokens: int
    decoded_frames: int
    total_ms: float
    cap_violations: list[str]
    warm_ms: float = 0.0          # total_ms without one-off model loading
    scouted_frames: int = 0
    answer: str = ""              # raw answer text (trace)
    option: int | None = None


def _conditions(qa: QAItem, transcript: Transcript, dtype: DamageType, seed: int,
                asr_wer: float, relevance_methods: frozenset[str] | None = None) -> dict[str, Transcript] | None:
    triple = make_triple(qa, transcript, dtype=dtype, seed=seed, relevance_methods=relevance_methods)
    if triple is None:
        return None
    return {"clean": triple.clean, "targeted_damage": triple.targeted, "control_damage": triple.control,
            "asr_noise": asr_style_noise(transcript, asr_wer, seed)}


def evaluate(items: list[tuple[QAItem, str, Transcript]], policies: list[Policy], cfg: dict, budget: Budget,
             scout: Scout, answerer: Answerer, dtype: DamageType = DamageType.DELETE, seed: int = 0,
             asr_wer: float = 0.2,
             relevance_methods: frozenset[str] | None = None, cost_model: CostModel = DEFAULT,
             eager: bool = False) -> tuple[list[EvalRecord], int]:
    """Run every policy on every (question, condition).  Returns records and the
    number of questions skipped because no fair triple could be built (no
    answer-relevant segment from an allowed relevance source, or no matched
    control)."""
    records: list[EvalRecord] = []
    skipped = 0
    max_side = cfg["answerer"].get("frame_max_side")
    for q_idx, (qa, video_path, transcript) in enumerate(items):
        conds = _conditions(qa, transcript, dtype, seed, asr_wer, relevance_methods)
        if conds is None:
            skipped += 1
            continue
        for cond_name, observed in conds.items():
            if eager:
                prep_meter = CostMeter()
                shared = prepare(video_path, observed, qa.question, qa.options, cfg, scout, prep_meter)
            k = q_idx % len(policies)
            for policy in policies[k:] + policies[:k]:          # rotated order
                meter = CostMeter()
                if eager:
                    text_only = isinstance(policy, TranscriptOnlyPolicy)
                    meter.records = [copy.copy(r) for r in prep_meter.records
                                     if not text_only or r.stage in _TEXT_ONLY_STAGES]
                    res = run_policy(shared, policy, answerer, budget, meter, cost_model)
                else:
                    prep = prepare_text(video_path, observed, qa.question, qa.options, cfg, meter)
                    cache = VisualCache(prep, scout, meter, max_side, qa.video_id)
                    res = run_lazy(prep, cache, policy, answerer, budget, meter, cost_model)
                total = meter.total()
                warm = sum(r.elapsed_ms for r in meter.records if not r.stage.endswith("_load"))
                records.append(EvalRecord(qa.qa_id, qa.video_id, policy.name, cond_name,
                                          answer_quality(res.answer, qa), len(res.frames), total.visual_tokens,
                                          total.decoded_frames, total.elapsed_ms, res.cap_violations,
                                          warm_ms=warm, scouted_frames=total.scored_frames,
                                          answer=res.answer.text, option=res.answer.option_index))
    return records, skipped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(records: list[EvalRecord]) -> dict[str, dict[str, dict[str, float]]]:
    """policy -> condition -> mean metrics."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r.policy][r.condition].append(r)
    out: dict = {}
    for pol, by_cond in acc.items():
        out[pol] = {}
        for cond, rs in by_cond.items():
            out[pol][cond] = {
                "n": len(rs),
                "quality": float(np.mean([r.quality for r in rs])),
                "frames": float(np.mean([r.frames for r in rs])),
                "visual_tokens": float(np.mean([r.visual_tokens for r in rs])),
                "decoded_frames": float(np.mean([r.decoded_frames for r in rs])),
                "total_ms": float(np.mean([r.total_ms for r in rs])),
                "warm_ms": float(np.mean([r.warm_ms for r in rs])),
                "scouted_frames": float(np.mean([r.scouted_frames for r in rs])),
                "p95_ms": float(np.percentile([r.total_ms for r in rs], 95)),
                "cap_violations": int(sum(bool(r.cap_violations) for r in rs)),
            }
        c = out[pol]
        if {"clean", "targeted_damage", "control_damage"} <= c.keys():
            tr = c["targeted_damage"]["frames"] - c["clean"]["frames"]
            co = c["control_damage"]["frames"] - c["clean"]["frames"]
            c["hypothesis"] = {"targeted_response": tr, "control_overspend": co, "selectivity": tr - co}
    return out


def cluster_bootstrap_diff(records: list[EvalRecord], policy_a: str, policy_b: str, condition: str,
                           metric: str = "quality", n_boot: int = 2000, seed: int = 0) -> dict[str, float] | None:
    """Paired difference (a - b) in ``metric`` with a 95% interval, resampling
    whole videos.  Returns None if the policies share no questions."""
    a = {r.qa_id: (r.video_id, getattr(r, metric)) for r in records if r.policy == policy_a and r.condition == condition}
    b = {r.qa_id: getattr(r, metric) for r in records if r.policy == policy_b and r.condition == condition}
    shared = [q for q in a if q in b]
    if not shared:
        return None
    by_video: dict[str, list[float]] = defaultdict(list)
    for q in shared:
        by_video[a[q][0]].append(a[q][1] - b[q])
    videos = list(by_video)
    rng = np.random.default_rng(seed)
    point = float(np.mean([d for v in videos for d in by_video[v]]))
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(len(videos), size=len(videos), replace=True)
        diffs = [d for i in sample for d in by_video[videos[i]]]
        boots.append(np.mean(diffs))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": point, "ci95_low": float(lo), "ci95_high": float(hi), "n_videos": len(videos),
            "n_questions": len(shared)}


def selectivity_bootstrap(records: list[EvalRecord], policy_a: str, policy_b: str | None = None,
                          n_boot: int = 2000, seed: int = 0) -> dict[str, float] | None:
    """Per-question selectivity = frames(targeted) - frames(control) (the clean
    condition cancels), averaged, with a 95% interval from resampling whole
    videos. With ``policy_b`` it is the paired difference sel(a) - sel(b):
    the exact quantity the hypothesis is about ("looks more when relevant
    speech is lost than when equal unrelated speech is lost")."""
    def per_question(policy: str) -> dict[str, tuple[str, float]]:
        t = {r.qa_id: r for r in records if r.policy == policy and r.condition == "targeted_damage"}
        c = {r.qa_id: r for r in records if r.policy == policy and r.condition == "control_damage"}
        return {q: (t[q].video_id, t[q].frames - c[q].frames) for q in t if q in c}

    a = per_question(policy_a)
    b = per_question(policy_b) if policy_b else None
    by_video: dict[str, list[float]] = defaultdict(list)
    for q, (vid, sel) in a.items():
        if b is None:
            by_video[vid].append(sel)
        elif q in b:
            by_video[vid].append(sel - b[q][1])
    if not by_video:
        return None
    videos = list(by_video)
    rng = np.random.default_rng(seed)
    point = float(np.mean([d for v in videos for d in by_video[v]]))
    boots = [np.mean([d for i in rng.choice(len(videos), len(videos)) for d in by_video[videos[i]]])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"selectivity" if b is None else "selectivity_diff": point, "ci95_low": float(lo),
            "ci95_high": float(hi), "n_videos": len(videos),
            "n_questions": sum(len(v) for v in by_video.values())}


def to_markdown(summary: dict) -> str:
    """Compact table: one row per (policy, condition)."""
    lines = ["| policy | condition | n | quality | frames | visual tok | decoded | scouted | warm ms | mean ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol, conds in summary.items():
        for cond in CONDITIONS:
            if cond in conds:
                m = conds[cond]
                lines.append(f"| {pol} | {cond} | {m['n']} | {m['quality']:.3f} | {m['frames']:.2f} | "
                             f"{m['visual_tokens']:.0f} | {m['decoded_frames']:.0f} | "
                             f"{m.get('scouted_frames', 0):.1f} | {m.get('warm_ms', 0):.0f} | {m['total_ms']:.0f} | "
                             f"{m['p95_ms']:.0f} |")
    lines += ["", "| policy | targeted_response | control_overspend | selectivity |", "|---|---|---|---|"]
    for pol, conds in summary.items():
        if "hypothesis" in conds:
            h = conds["hypothesis"]
            lines.append(f"| {pol} | {h['targeted_response']:+.2f} | {h['control_overspend']:+.2f} | "
                         f"{h['selectivity']:+.2f} |")
    return "\n".join(lines)


def write_report(out_dir: str | Path, records: list[EvalRecord], summary: dict, extra: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("\n".join(json.dumps(dataclasses.asdict(r)) for r in records) + "\n",
                                       encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"summary": summary, **extra}, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(to_markdown(summary) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Primary endpoint: recovery under important-speech failure, with its cost
# ---------------------------------------------------------------------------


def speech_failure_stratum(records: list[EvalRecord], baseline: str = "transcript_only") -> set[str]:
    """Questions where the transcript mattered and the damage broke it:
    the transcript-only baseline is right on CLEAN and wrong on TARGETED."""
    q = {(r.qa_id, r.condition): r.quality for r in records if r.policy == baseline}
    return {qa for (qa, c), v in q.items() if c == "clean" and v >= 1.0 and q.get((qa, "targeted_damage"), 1.0) < 1.0}


def recovery_table(records: list[EvalRecord], baseline: str = "transcript_only", n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Per policy, on the stratum: recovery = mean quality under TARGETED,
    with its warm cost and a 95% interval resampling whole videos. Also the
    same policy's CONTROL-condition cost on those questions (overspend).
    Compare policies at EQUAL cost (Pareto), not by recovery alone."""
    stratum = speech_failure_stratum(records, baseline)
    out: dict = {"stratum_questions": len(stratum), "policies": {}}
    if not stratum:
        return out
    rng = np.random.default_rng(seed)
    for pol in sorted({r.policy for r in records}):
        t = [r for r in records if r.policy == pol and r.condition == "targeted_damage" and r.qa_id in stratum]
        c = [r for r in records if r.policy == pol and r.condition == "control_damage" and r.qa_id in stratum]
        if not t:
            continue
        by_video: dict[str, list[float]] = defaultdict(list)
        for r in t:
            by_video[r.video_id].append(r.quality)
        vids = list(by_video)
        boots = [np.mean([x for i in rng.choice(len(vids), len(vids)) for x in by_video[vids[i]]])
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["policies"][pol] = {
            "recovery": float(np.mean([r.quality for r in t])), "ci95": [float(lo), float(hi)],
            "n_questions": len(t), "n_videos": len(vids),
            "targeted_warm_ms": float(np.mean([r.warm_ms for r in t])),
            "control_warm_ms": float(np.mean([r.warm_ms for r in c])) if c else None,
            "targeted_frames": float(np.mean([r.frames for r in t])),
            "control_frames": float(np.mean([r.frames for r in c])) if c else None,
        }
    return out
