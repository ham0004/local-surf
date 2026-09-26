#!/usr/bin/env bash
# Full real-data run on LongVideoBench: labels -> training (5 seeds) -> evaluation
# with accuracy-vs-frames threshold sweeps. Exact commands behind reports/lvb_full/.
#
# Prerequisite (resumable; skips videos already on disk):
#   HF_TOKEN=... uv run python scripts/download_longvideobench_subset.py --out data/longvideobench_full
#
# Relevance is 'annotation': LongVideoBench answers are visual, so only the
# dataset's own quoted-subtitle evidence may define "answer-relevant speech".
set -euo pipefail

DATA=data/longvideobench_full
CFG=configs/gpu_12gb.yaml
OUT=runs/lvb_full
mkdir -p "$OUT"

uv run --no-sync videoqa build-labels --data "$DATA" --split train --relevance annotation \
    --config "$CFG" --out "$OUT/labels_train"
uv run --no-sync videoqa build-labels --data "$DATA" --split dev --relevance annotation \
    --config "$CFG" --out "$OUT/labels_dev"

# Training is CPU-cheap: 5 seeds give a variance estimate for the paired-vs-unpaired gap.
for seed in 0 1 2 3 4; do
  uv run --no-sync videoqa train-controller --train-labels "$OUT/labels_train" \
      --dev-labels "$OUT/labels_dev" --seed "$seed" --out "$OUT/ckpt_seed$seed"
done

# Evaluate seed 0 heads at their dev-tuned threshold AND along a common threshold
# sweep, so paired and unpaired are compared at matched operating points.
P="$OUT/ckpt_seed0/head_paired.npz"
U="$OUT/ckpt_seed0/head_unpaired.npz"
POLICIES="transcript_only,uniform,retrieval,scout_similarity,heuristic"
POLICIES+=",learned=$P,learned=$U"
for t in 0.0 0.1 0.3 0.6; do POLICIES+=",learned=$P@$t,learned=$U@$t"; done

uv run --no-sync videoqa evaluate --data "$DATA" --split test --relevance annotation \
    --config "$CFG" --policies "$POLICIES" --out "$OUT/eval_test"
