#!/usr/bin/env bash
# v1 benchmark: every LongVideoBench question in data/longvideobench_full on its
# original subtitles, answered by each training-free frame-selection policy.
# Exact command behind reports/benchmark_v1/.
#
# Prerequisite (resumable; skips videos already on disk):
#   HF_TOKEN=... uv run python scripts/download_longvideobench_subset.py --out data/longvideobench_full
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

uv run --no-sync videoqa evaluate --data data/longvideobench_full \
    --split train,dev,calibration,test --clean-only \
    --policies transcript_only,uniform,retrieval,scout_similarity,heuristic \
    --config configs/gpu_12gb.yaml --out runs/bench_lvb_v1

# Open-ended variant: same questions with the options hidden; free answers are
# graded by a local LLM judge (scripts/judge_open_ended.py).
uv run --no-sync videoqa evaluate --data data/longvideobench_full \
    --split train,dev,calibration,test --clean-only --open-ended \
    --policies transcript_only,uniform,retrieval,scout_similarity,heuristic \
    --config configs/gpu_12gb.yaml --out runs/bench_lvb_v1_open
