"""Command-line interface: `uv run videoqa <command> ...`.

Commands (each maps to one pipeline stage; see README for full examples):

    make-synthetic    write N synthetic lectures (plumbing smoke data)
    ask               answer one question on one video, export evidence + cost trace
    build-labels      automatic action-usefulness labels for a split (train/dev)
    train-controller  train paired and unpaired utility heads, tune STOP on dev
    evaluate          compare policies across transcript conditions on a split
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
from .damage import make_triple
from .datasets import load_local_dataset
from .evaluate import cluster_bootstrap_diff, evaluate, summarise, to_markdown, write_report
from .heads import TrainConfig
from .labels import label_items, write_labels
from .pipeline import prepare, run_policy, save_run
from .schemas import DamageType, to_jsonable
from .scout import make_scout
from .train import train_paired_and_unpaired
from .transcript import load_transcript


def _scout(cfg):
    return make_scout(cfg["scout"]["backend"], device=cfg.get("device", "cpu"))


def cmd_make_synthetic(a) -> None:
    qa_path = fixtures.write_dataset(a.out, a.n, a.seed0)
    print(f"wrote {a.n} synthetic lectures -> {qa_path}")


def cmd_ask(a) -> None:
    cfg = load_config(a.config)
    transcript = load_transcript(a.transcript, Path(a.video).stem)
    options = a.options.split("|") if a.options else None
    meter = CostMeter()
    prep = prepare(a.video, transcript, a.question, options, cfg, _scout(cfg), meter)
    policy = make_policy(a.policy, a.checkpoint)
    res = run_policy(prep, policy, make_answerer(cfg), budget_from_config(cfg), meter)
    out = save_run(prep, res, a.out, {"config": a.config})
    print(json.dumps({"answer": to_jsonable(res.answer), "frames_s": [f.decoded_pts_s for f in res.frames],
                      "total_ms": round(meter.total().elapsed_ms), "result": str(out)}, indent=2))


def _triples(items, dtype, seed):
    out, skipped = [], 0
    for it in items:
        triple = make_triple(it.qa, it.transcript, dtype=dtype, seed=seed)
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
    triples, skipped = _triples(items, dtype, a.seed)
    # Label generation is train-only by construction; a dev label set is built
    # the same way but only ever used for threshold tuning.
    triples = [(dataclasses.replace(qa, source_split="train" if a.split == "train" else "unassigned"), p, t)
               for qa, p, t in triples]
    answerer = make_answerer(cfg)
    rows, examples, stats = label_items(triples, cfg, budget_from_config(cfg), _scout(cfg), answerer,
                                        max_steps=a.max_steps)
    manifest = {"data": str(a.data), "split": a.split, "config": a.config, "answerer": answerer.name,
                "answerer_model": cfg["answerer"].get("model_id"), "answerer_revision": cfg["answerer"].get("revision"),
                "damage_type": dtype.value, "seed": a.seed, "questions": len(triples),
                "skipped_no_matched_control": skipped, "missing_video": len(report.missing_video),
                "answer_calls": stats.answer_calls, "cache_hits": stats.cache_hits,
                "labelling_seconds": round(stats.seconds, 2),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_labels(a.out, rows, examples, manifest)
    print(json.dumps(manifest, indent=2))


def cmd_train(a) -> None:
    res = train_paired_and_unpaired(a.train_labels, a.dev_labels, a.out,
                                    TrainConfig(epochs=a.epochs, hidden=a.hidden, seed=a.seed),
                                    lambda_pair=a.lambda_pair, look_cost=a.look_cost)
    print(json.dumps(res, indent=2))


def cmd_evaluate(a) -> None:
    cfg = load_config(a.config)
    items, report = load_local_dataset(a.data, splits=(a.split,))
    items = items[: a.limit] if a.limit else items
    policies = []
    for spec in a.policies.split(","):
        name, _, ckpt = spec.partition("=")          # e.g. learned=runs/ckpt/head_paired.npz
        pol = make_policy(name, ckpt or None)
        if ckpt:
            pol.name = f"learned[{Path(ckpt).stem}]"
        policies.append(pol)
    records, skipped = evaluate([(it.qa, it.video_path, it.transcript) for it in items], policies, cfg,
                                budget_from_config(cfg), _scout(cfg), make_answerer(cfg),
                                dtype=DamageType(a.damage), seed=a.seed)
    summary = summarise(records)
    names = [p.name for p in policies]
    comparisons = {}
    for i, pa in enumerate(names):
        for pb in names[i + 1:]:
            for cond in ("clean", "targeted_damage", "control_damage", "asr_noise"):
                for metric in ("quality", "frames"):
                    comparisons[f"{pa} - {pb} | {cond} | {metric}"] = cluster_bootstrap_diff(
                        records, pa, pb, cond, metric=metric)
    write_report(a.out, records, summary, {"comparisons": comparisons, "skipped_no_matched_control": skipped,
                                           "missing_video": len(report.missing_video), "config": a.config,
                                           "split": a.split, "data": str(a.data)})
    print(to_markdown(summary))


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

    s = sub.add_parser("evaluate")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--config", default="configs/cpu.yaml")
    s.add_argument("--policies", default="transcript_only,uniform,retrieval,scout_similarity,heuristic",
                   help="comma list; learned policies as learned=path/to/head.npz")
    s.add_argument("--damage", default="delete", choices=[d.value for d in DamageType])
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_evaluate)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
