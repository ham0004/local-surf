#!/usr/bin/env bash
# Lecture study, dev only (docs/v3/m3av_protocol.md): dense features, R5 settings, then the candidate pools of
# every system whose pool needs no dev choice. One GPU job at a time; resumable (finished files are skipped).
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
$PY scripts/m3av_dataset.py --split dev
$PY scripts/v3_features.py --dataset m3av_dev --step 2
$PY scripts/m3av_vsi.py --split dev
for pool in uniform4 dense4 dense4o v2 hybrid hybrid_ahead hybrid_ahead_m3av; do
  $PY scripts/v3_baseline_cycle.py pools --dataset m3av_dev --pool $pool
done
echo "M3AV DEV POOLS DONE"
