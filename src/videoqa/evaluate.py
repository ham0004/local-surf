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

Cost fairness: all policies share one prepare() per (question, condition) for
speed, but each run is CHARGED the prepare stages it would need on its own.
The transcript-only baseline needs no decoding or scouting, so it is charged
only probe + retrieval.
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


def _conditions(qa: QAItem, transcript: Transcript, dtype: DamageType, seed: int,
                asr_wer: float) -> dict[str, Transcript] | None:
    triple = make_triple(qa, transcript, dtype=dtype, seed=seed)
    if triple is None:
        return None
    return {"clean": triple.clean, "targeted_damage": triple.targeted, "control_damage": triple.control,
            "asr_noise": asr_style_noise(transcript, asr_wer, seed)}


def evaluate(items: list[tuple[QAItem, str, Transcript]], policies: list[Policy], cfg: dict, budget: Budget,
             scout: Scout, answerer: Answerer, dtype: DamageType = DamageType.DELETE, seed: int = 0,
             asr_wer: float = 0.2) -> tuple[list[EvalRecord], int]:
    """Run every policy on every (question, condition).  Returns records and the
    number of questions skipped because no fair matched control existed."""
    records: list[EvalRecord] = []
    skipped = 0
    for qa, video_path, transcript in items:
        conds = _conditions(qa, transcript, dtype, seed, asr_wer)
        if conds is None:
            skipped += 1
            continue
        for cond_name, observed in conds.items():
            prep_meter = CostMeter()
            prep = prepare(video_path, observed, qa.question, qa.options, cfg, scout, prep_meter)
            for policy in policies:
                meter = CostMeter()
                text_only = isinstance(policy, TranscriptOnlyPolicy)
                meter.records = [copy.copy(r) for r in prep_meter.records
                                 if not text_only or r.stage in _TEXT_ONLY_STAGES]
                res = run_policy(prep, policy, answerer, budget, meter)
                total = meter.total()
                records.append(EvalRecord(qa.qa_id, qa.video_id, policy.name, cond_name,
                                          answer_quality(res.answer, qa), len(res.frames), total.visual_tokens,
                                          total.decoded_frames, total.elapsed_ms, res.cap_violations))
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


def to_markdown(summary: dict) -> str:
    """Compact table: one row per (policy, condition)."""
    lines = ["| policy | condition | n | quality | frames | visual tok | decoded | mean ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pol, conds in summary.items():
        for cond in CONDITIONS:
            if cond in conds:
                m = conds[cond]
                lines.append(f"| {pol} | {cond} | {m['n']} | {m['quality']:.3f} | {m['frames']:.2f} | "
                             f"{m['visual_tokens']:.0f} | {m['decoded_frames']:.0f} | {m['total_ms']:.0f} | "
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
