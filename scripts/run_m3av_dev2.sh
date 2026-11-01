#!/usr/bin/env bash
# Lecture study, dev, after scripts/run_m3av_dev.sh: R5 pool (setting chosen on dev), Head B (S3, the CG-Bench-trained
# ev_set, lam 2, mu 1), then the primary measure for every system.
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
$PY scripts/v3_baseline_cycle.py pools --dataset m3av_dev --pool vsi4
$PY scripts/v3_head_b.py features --dataset m3av_dev --pool hybrid
$PY scripts/v3_head_b.py select --dataset m3av_dev --pool hybrid --train-pool hybrid --method ev_set --lam 2.0 --mu 1.0
$PY scripts/m3av_hits.py --split dev
echo "M3AV DEV2 DONE"
