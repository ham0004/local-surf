#!/usr/bin/env bash
# Lecture study, dev: fixed-K rules that show 4 distinct pictures (research log step 86), then the primary measure.
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
$PY scripts/m3av_vsi.py --split dev                    # ranked 24 times per setting (hit@4 unchanged: first 4)
for pool in uniform4d dense4d0 dense4d3 dense4od0 dense4od3 vsi4d0 vsi4d3; do
  $PY scripts/v3_baseline_cycle.py pools --dataset m3av_dev --pool $pool
done
$PY scripts/m3av_hits.py --split dev
echo "M3AV DEV3 DONE"
