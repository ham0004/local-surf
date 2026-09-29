"""Command-line interface: `uv run videoqa <command> ...`.

Commands (each maps to one pipeline stage; see README for full examples):

    make-synthetic    write N synthetic lectures (plumbing smoke data)
    ask               answer one question on one video, export evidence + cost trace
    evaluate          compare policies on a split (clean transcript, or clean /
                      targeted / control damage conditions)
    profile           cold vs warm per-stage cost of one question on this device
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from . import fixtures
from .answerer import make_answerer
from .config import budget_from_config, load_config
from .controller import make_policy
from .costs import CostMeter
from .damage import ANNOTATION_RELEVANCE, make_triple
from .datasets import load_local_dataset
from .evaluate import (
    cluster_bootstrap_diff,
    evaluate,
    recovery_table,
    selectivity_bootstrap,
    summarise,
    to_markdown,
    write_report,
)
from .acquisition import VisualCache, prepare_text, run_lazy
from .cost_model import cost_model_from_config
from .pipeline import save_run
from .schemas import DamageType, to_jsonable
from .scout import make_scout
from .transcript import load_transcript


def _scout(cfg):
    backend = cfg["scout"]["backend"]
    if backend == "open_clip":
        cache = str(Path(cfg.get("paths", {}).get("cache_dir", "cache")) / "open_clip")
        return make_scout(backend, device=cfg.get("device", "cpu"), cache_dir=cache)
    return make_scout(backend, device=cfg.get("device", "cpu"))


def cmd_make_synthetic(a) -> None:
    qa_path = fixtures.write_dataset(a.out, a.n, a.seed0)
    print(f"wrote {a.n} synthetic lectures -> {qa_path}")


def cmd_ask(a) -> None:
    cfg = load_config(a.config)
    transcript = load_transcript(a.transcript, Path(a.video).stem)
    options = a.options.split("|") if a.options else None
    meter = CostMeter()
    cm = cost_model_from_config(cfg)
    policy = make_policy(a.policy, cm)
    res, prep = _run_lazy(a.video, transcript, a.question, options, cfg, _scout(cfg), make_answerer(cfg),
                          policy, meter, cm)
    out = save_run(prep, res, a.out, {"config": a.config})
    print(json.dumps({"answer": to_jsonable(res.answer), "frames_s": [f.decoded_pts_s for f in res.frames],
                      "total_ms": round(meter.total().elapsed_ms), "result": str(out)}, indent=2))


def _run_lazy(video, transcript, question, options, cfg, scout, answerer, policy, meter, cm):
    """The deployed selective-search path (same code the evaluator runs)."""
    prep = prepare_text(video, transcript, question, options, cfg, meter)
    cache = VisualCache(prep, scout, meter, cfg["answerer"].get("frame_max_side"))
    return run_lazy(prep, cache, policy, answerer, budget_from_config(cfg), meter, cm), prep


RELEVANCE_CHOICES = {"any": None, "annotation": ANNOTATION_RELEVANCE}


def _triples(items, dtype, seed, relevance_methods=None):
    out, skipped = [], 0
    for it in items:
        triple = make_triple(it.qa, it.transcript, dtype=dtype, seed=seed, relevance_methods=relevance_methods)
        if triple is None:
            skipped += 1
            continue
        out.append((it.qa, it.video_path, triple))
    return out, skipped


def parse_policy_spec(spec: str, cost_model=None):
    """A policy name: transcript_only | uniform | retrieval | scout_similarity | heuristic."""
    from .cost_model import DEFAULT  # noqa: PLC0415
    return make_policy(spec.strip(), cost_model or DEFAULT)


def cmd_evaluate(a) -> None:
    cfg = load_config(a.config)
    items, report = load_local_dataset(a.data, splits=tuple(a.split.split(",")))
    items = items[: a.limit] if a.limit else items
    if getattr(a, "open_ended", False):
        # Free-answer mode: hide the options so the model must write its own
        # answer; it is scored later against the gold answer text (token F1 in
        # the records, LLM judge in scripts/judge_open_ended.py).
        items = [dataclasses.replace(it, qa=dataclasses.replace(it.qa, options=None, gold_option_index=None))
                 for it in items]
    cm = cost_model_from_config(cfg)
    policies = [parse_policy_spec(spec, cm) for spec in a.policies.split(",")]
    records, skipped = evaluate([(it.qa, it.video_path, it.transcript) for it in items], policies, cfg,
                                budget_from_config(cfg), _scout(cfg), make_answerer(cfg),
                                dtype=DamageType(a.damage), seed=a.seed,
                                relevance_methods=RELEVANCE_CHOICES[a.relevance], cost_model=cm,
                                eager=getattr(a, "eager", False), clean_only=getattr(a, "clean_only", False))
    summary = summarise(records)
    names = [p.name for p in policies]
    comparisons = {}
    for i, pa in enumerate(names):
        for pb in names[i + 1:]:
            for cond in ("clean", "targeted_damage", "control_damage", "asr_noise"):
                for metric in ("quality", "frames"):
                    comparisons[f"{pa} - {pb} | {cond} | {metric}"] = cluster_bootstrap_diff(
                        records, pa, pb, cond, metric=metric)
    selectivity = {pa: selectivity_bootstrap(records, pa) for pa in names}
    selectivity_diffs = {f"{pa} - {pb}": selectivity_bootstrap(records, pa, pb)
                         for i, pa in enumerate(names) for pb in names[i + 1:]}
    recovery = recovery_table(records)
    write_report(a.out, records, summary, {"primary_recovery": recovery,
                                           "comparisons": comparisons, "selectivity": selectivity,
                                           "selectivity_diffs": selectivity_diffs,
                                           "skipped_no_fair_triple": skipped, "relevance": a.relevance,
                                           "missing_video": len(report.missing_video), "config": a.config,
                                           "split": a.split, "data": str(a.data)})
    print(to_markdown(summary))
    print(json.dumps({"primary_recovery": recovery}, indent=2))


def cmd_profile(a) -> None:
    """Cold vs warm cost on one device: run the same question ``repeats`` times
    in ONE process.  Run 1 includes model loading (cold); later runs reuse the
    loaded models (warm).  Reports per-stage ms and peak memory."""
    import platform
    import statistics

    import psutil

    cfg = load_config(a.config)
    transcript = load_transcript(a.transcript, Path(a.video).stem)
    options = a.options.split("|") if a.options else None
    scout, answerer, policy = _scout(cfg), make_answerer(cfg), make_policy(a.policy)
    runs = []
    for _ in range(a.repeats):
        meter = CostMeter()
        res, _ = _run_lazy(a.video, transcript, a.question, options, cfg, scout, answerer, policy, meter,
                           cost_model_from_config(cfg))
        runs.append({k: to_jsonable(v) for k, v in meter.by_stage().items()} | {"_answer": res.answer.text})
    stages = sorted({k for r in runs for k in r if not k.startswith("_")})
    warm = runs[1:] or runs

    def med(stage, field):
        return statistics.median(r.get(stage, {}).get(field, 0) for r in warm)

    report = {
        "config": a.config, "policy": a.policy, "repeats": a.repeats,
        "device": {"platform": platform.platform(), "cpu": platform.processor(),
                   "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)},
        "cold_ms": {s: round(runs[0].get(s, {}).get("elapsed_ms", 0), 1) for s in stages},
        "warm_median_ms": {s: round(med(s, "elapsed_ms"), 1) for s in stages},
        "cold_total_ms": round(sum(runs[0].get(s, {}).get("elapsed_ms", 0) for s in stages), 1),
        "warm_total_ms": round(sum(med(s, "elapsed_ms") for s in stages), 1),
        "peak_vram_gb": round(max(r.get(s, {}).get("peak_vram_bytes", 0) for r in runs for s in stages) / 1e9, 2),
        "peak_ram_gb": round(max(r.get(s, {}).get("peak_ram_bytes", 0) for r in runs for s in stages) / 1e9, 2),
        "answers": [r["_answer"] for r in runs],
        "runs": runs,
    }
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            report["device"]["gpu"] = torch.cuda.get_device_name(0)
            report["device"]["torch"] = torch.__version__
    except ImportError:
        pass
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="videoqa", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-synthetic")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed0", type=int, default=0)
    s.set_defaults(fn=cmd_make_synthetic)

    s = sub.add_parser("ask")
    s.add_argument("--video", required=True)
    s.add_argument("--transcript", required=True, help=".srt/.vtt/.json")
    s.add_argument("--question", required=True)
    s.add_argument("--options", help="multiple-choice options separated by |")
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--policy", default="heuristic")
    s.add_argument("--out", default="runs/ask")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("evaluate")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="test", help="one split or a comma list, e.g. train,dev,calibration,test")
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--policies", default="transcript_only,uniform,retrieval,scout_similarity,heuristic",
                   help="comma list of policy names")
    s.add_argument("--damage", default="delete", choices=[d.value for d in DamageType])
    s.add_argument("--relevance", default="any", choices=sorted(RELEVANCE_CHOICES),
                   help="which relevance sources may define targeted damage; use 'annotation' when "
                        "answers are visual (e.g. LongVideoBench)")
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--clean-only", action="store_true",
                   help="benchmark mode: every question on its original transcript only (no damage "
                        "conditions, so no question is skipped for lacking a fair triple)")
    s.add_argument("--open-ended", action="store_true",
                   help="hide the multiple-choice options: the model writes a free answer, scored against the "
                        "gold answer text (token F1 here; LLM judge via scripts/judge_open_ended.py)")
    s.add_argument("--eager", action="store_true",
                   help="legacy path: decode+scout every candidate up front, then charge back (historical runs only)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("profile")
    s.add_argument("--video", required=True)
    s.add_argument("--transcript", required=True)
    s.add_argument("--question", required=True)
    s.add_argument("--options")
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--policy", default="heuristic")
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--out", default="reports/profile.json")
    s.set_defaults(fn=cmd_profile)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
