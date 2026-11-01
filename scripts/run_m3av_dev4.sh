#!/usr/bin/env bash
# Lecture study, dev: final fixed-K rule pools (h = 0, 96 ranked times; research log step 86), then the measure.
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
$PY scripts/m3av_vsi.py --split dev
for pool in dense4x dense4ox vsi4x; do $PY scripts/v3_baseline_cycle.py pools --dataset m3av_dev --pool $pool; done
$PY scripts/m3av_hits.py --split dev
echo "M3AV DEV4 DONE"
