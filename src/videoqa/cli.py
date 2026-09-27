"""Command-line interface: `uv run videoqa <command> ...`.

Commands (each maps to one pipeline stage; see README for full examples):

    make-synthetic    write N synthetic lectures (plumbing smoke data)
    ask               answer one question on one video, export evidence + cost trace
    build-labels      automatic action-usefulness labels for a split (train/dev)
    train-controller  train paired and unpaired utility heads, tune STOP on dev
    evaluate          compare policies across transcript conditions on a split
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
from .evaluate import cluster_bootstrap_diff, evaluate, selectivity_bootstrap, summarise, to_markdown, write_report
from .heads import TrainConfig
from .labels import label_items, write_labels
from .acquisition import VisualCache, prepare_text, run_lazy
from .cost_model import cost_model_from_config
from .pipeline import save_run
from .schemas import DamageType, to_jsonable
from .scout import make_scout
from .train import train_paired_and_unpaired
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
    policy = make_policy(a.policy, a.checkpoint, cm)
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


def cmd_build_labels(a) -> None:
    cfg = load_config(a.config)
    items, report = load_local_dataset(a.data, splits=(a.split,))
    items = items[: a.limit] if a.limit else items
    dtype = DamageType(a.damage)
    triples, skipped = _triples(items, dtype, a.seed, RELEVANCE_CHOICES[a.relevance])
    # train labels fit heads; dev labels only tune thresholds. The split is
    # carried as loaded (no relabelling); labels.py refuses anything else.
    answerer = make_answerer(cfg)
    rows, examples, stats = label_items(triples, cfg, budget_from_config(cfg), _scout(cfg), answerer,
                                        max_steps=a.max_steps)
    manifest = {"data": str(a.data), "split": a.split, "config": a.config, "answerer": answerer.name,
                "answerer_model": cfg["answerer"].get("model_id"), "answerer_revision": cfg["answerer"].get("revision"),
                "damage_type": dtype.value, "relevance": a.relevance, "seed": a.seed,
                "questions": len(triples), "skipped_no_fair_triple": skipped, "missing_video": len(report.missing_video),
                "answer_calls": stats.answer_calls, "cache_hits": stats.cache_hits,
                "labelling_seconds": round(stats.seconds, 2),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_labels(a.out, rows, examples, manifest)
    print(json.dumps(manifest, indent=2))


def cmd_build_assay(a) -> None:
    """Controlled paired assay labels (assay.py): common pool, exact keys."""
    from .assay import check_media_overlap, label_assay, write_assay  # noqa: PLC0415

    cfg = load_config(a.config)
    splits = tuple(a.split.split(","))
    items, report = load_local_dataset(a.data, splits=splits)
    items = items[: a.limit] if a.limit else items
    media = check_media_overlap([(it.qa, it.video_path) for it in items])
    answerer, cm = make_answerer(cfg), cost_model_from_config(cfg)
    rows, pools, questions, stats = label_assay(
        [(it.qa, it.video_path, it.transcript) for it in items], cfg, budget_from_config(cfg), _scout(cfg),
        answerer, cm, dtype=DamageType(a.damage), seed=a.seed, relevance_methods=RELEVANCE_CHOICES[a.relevance],
        log=lambda m: print(m, flush=True))
    manifest = {"data": str(a.data), "split": a.split, "config": a.config, "damage_type": a.damage,
                "relevance": a.relevance, "seed": a.seed, "missing_video": len(report.missing_video),
                "answerer": getattr(answerer, "fingerprint", lambda: {"name": answerer.name})(),
                "scout": {"backend": cfg["scout"]["backend"], **{k: v for k, v in cfg["scout"].items() if k != "backend"}},
                "cost_model": dataclasses.asdict(cm), "media_fingerprints": len(media),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(write_assay(a.out, rows, pools, questions, stats, manifest), indent=2))


def cmd_train_assay(a) -> None:
    """V0/V1/V2 heads on assay-v2 labels; decision lambda calibrated on dev."""
    from .train_assay import train_assay  # noqa: PLC0415

    price = a.price_per_s if a.price_per_s is not None else cost_model_from_config(load_config(a.config)).lambda_per_s
    res = train_assay(a.train_labels, a.dev_labels, a.out, TrainConfig(epochs=a.epochs, hidden=a.hidden),
                      price_per_s=price, seeds=tuple(int(x) for x in a.seeds.split(",")))
    brief = {v: [{"seed": s["seed"], "lambda": s["lambda"], **s["dev"],
                  "dev_utility": s["dev_at_lambda"]["all"]["utility"]} for s in runs]
             for v, runs in res["variants"].items()}
    print(json.dumps({"train": res["train"], "dev": res["dev"], "price_per_s": price,
                      "reference": {k: v["all"] for k, v in res["reference"].items()}, "variants": brief}, indent=2))


def cmd_train_llm(a) -> None:
    """Qwen3-0.6B + LoRA gain scorer on assay labels (V0/V1/V2), lambda on dev."""
    from .llm_controller import LLMTrainConfig, train_llm  # noqa: PLC0415

    price = a.price_per_s if a.price_per_s is not None else cost_model_from_config(load_config(a.config)).lambda_per_s
    out = {}
    for v in a.variants.split(","):
        for seed in (int(x) for x in a.seeds.split(",")):
            meta = train_llm(a.train_labels, a.dev_labels, a.out,
                             LLMTrainConfig(variant=v, seed=seed, steps=a.steps), price_per_s=price)
            out[f"{v}_seed{seed}"] = {k: meta[k] for k in ("lambda_per_s", "trainable_params", "train_tokens",
                                                            "train_seconds", "peak_vram_gb",
                                                            "dev_infer_ms_per_action_batched", "dev")} | {
                "dev_utility": meta["dev_at_lambda"]["all"]["utility"]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "train_llm_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def cmd_train(a) -> None:
    res = train_paired_and_unpaired(a.train_labels, a.dev_labels, a.out,
                                    TrainConfig(epochs=a.epochs, hidden=a.hidden, seed=a.seed),
                                    lambda_pair=a.lambda_pair, look_cost=a.look_cost)
    print(json.dumps(res, indent=2))


def parse_policy_spec(spec: str, cost_model=None):
    """``name`` | ``learned=<head.npz>`` | ``learned=<head.npz>@<stop_threshold>``.

    The ``@threshold`` form overrides the dev-tuned STOP threshold, so paired
    and unpaired heads can be compared along the same accuracy-vs-frames curve
    instead of at two different, separately tuned thresholds (which confounded
    earlier comparisons)."""
    from .cost_model import DEFAULT  # noqa: PLC0415
    cost_model = cost_model or DEFAULT
    name, _, rest = spec.strip().partition("=")
    if name == "llm":
        from .llm_controller import LLMController  # noqa: PLC0415
        return LLMController(rest, cost_model)
    ckpt, _, thr = rest.partition("@")
    pol = make_policy(name, ckpt or None, cost_model)
    if ckpt:
        pol.name = f"learned[{Path(ckpt).stem}]"
    if thr:
        pol.stop_threshold = float(thr)
        pol.name = f"{pol.name}@{float(thr):g}"
    return pol


def cmd_evaluate(a) -> None:
    cfg = load_config(a.config)
    items, report = load_local_dataset(a.data, splits=(a.split,))
    items = items[: a.limit] if a.limit else items
    cm = cost_model_from_config(cfg)
    policies = [parse_policy_spec(spec, cm) for spec in a.policies.split(",")]
    records, skipped = evaluate([(it.qa, it.video_path, it.transcript) for it in items], policies, cfg,
                                budget_from_config(cfg), _scout(cfg), make_answerer(cfg),
                                dtype=DamageType(a.damage), seed=a.seed,
                                relevance_methods=RELEVANCE_CHOICES[a.relevance], cost_model=cm,
                                eager=getattr(a, "eager", False))
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
    write_report(a.out, records, summary, {"comparisons": comparisons, "selectivity": selectivity,
                                           "selectivity_diffs": selectivity_diffs,
                                           "skipped_no_fair_triple": skipped, "relevance": a.relevance,
                                           "missing_video": len(report.missing_video), "config": a.config,
                                           "split": a.split, "data": str(a.data)})
    print(to_markdown(summary))


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
    scout, answerer, policy = _scout(cfg), make_answerer(cfg), make_policy(a.policy, a.checkpoint)
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
    s.add_argument("--checkpoint")
    s.add_argument("--out", default="runs/ask")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("build-labels")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="train", choices=["train", "dev"])
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--damage", default="delete", choices=[d.value for d in DamageType])
    s.add_argument("--relevance", default="any", choices=sorted(RELEVANCE_CHOICES),
                   help="which relevance sources may define targeted damage; use 'annotation' when "
                        "answers are visual (e.g. LongVideoBench)")
    s.add_argument("--max-steps", type=int, default=2)
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_labels)

    s = sub.add_parser("train-controller")
    s.add_argument("--train-labels", required=True)
    s.add_argument("--dev-labels")
    s.add_argument("--out", default="checkpoints/controller")
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--hidden", type=int, default=32)
    s.add_argument("--lambda-pair", type=float, default=1.0)
    s.add_argument("--look-cost", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("build-assay", help="controlled paired assay labels (common pool, exact pair keys)")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="train", help="train, dev or train,dev (test is refused)")
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--damage", default="delete", choices=[d.value for d in DamageType])
    s.add_argument("--relevance", default="any", choices=sorted(RELEVANCE_CHOICES))
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_assay)

    s = sub.add_parser("train-assay", help="V0/V1/V2 heads on assay labels, lambda calibrated on dev")
    s.add_argument("--train-labels", required=True)
    s.add_argument("--dev-labels", required=True)
    s.add_argument("--config", default="configs/cpu.yaml", help="source of the declared price of time")
    s.add_argument("--price-per-s", type=float)
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--hidden", type=int, default=32)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_train_assay)

    s = sub.add_parser("train-llm", help="Qwen3-0.6B + LoRA controller on assay labels")
    s.add_argument("--train-labels", required=True)
    s.add_argument("--dev-labels", required=True)
    s.add_argument("--config", default="configs/gpu_12gb.yaml")
    s.add_argument("--price-per-s", type=float)
    s.add_argument("--variants", default="V0,V1,V2")
    s.add_argument("--seeds", default="0")
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_train_llm)

    s = sub.add_parser("evaluate")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--policies", default="transcript_only,uniform,retrieval,scout_similarity,heuristic",
                   help="comma list; learned policies as learned=path/to/head.npz or learned=path@threshold")
    s.add_argument("--damage", default="delete", choices=[d.value for d in DamageType])
    s.add_argument("--relevance", default="any", choices=sorted(RELEVANCE_CHOICES),
                   help="which relevance sources may define targeted damage; use 'annotation' when "
                        "answers are visual (e.g. LongVideoBench)")
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
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
    s.add_argument("--checkpoint")
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--out", default="reports/profile.json")
    s.set_defaults(fn=cmd_profile)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
