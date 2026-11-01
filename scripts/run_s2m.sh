#!/usr/bin/env bash
# S2m (docs/v3/m3av_protocol.md): Head A retrained on M3AV train lectures with the deployed recipe
# (light_dens, ens features, human labels, width 32, lr 1e-3, 15 epochs, 3 seeds); dev = M3AV dev visible spans.
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HEAD_A_DATA=data/head_a_m3av HEAD_A_RUN=runs/v3_head_a_m3av HEAD_A_OUT=reports/m3av
PY=.venv/Scripts/python
mkdir -p $HEAD_A_RUN
$PY scripts/m3av_head_a_data.py
$PY scripts/v3_head_a_train.py prepare                       # zero-shot MiniLM relevance (also the zs control)
$PY scripts/v3_head_a_train.py zs_dens
$PY scripts/v3_head_a_light.py bge
$PY scripts/v3_head_a_light.py features --feature-set ens
for s in 0 1 2; do
  $PY scripts/v3_head_a_light.py train --method light_dens --feature-set ens --no-weak --epochs 15 --seed $s --save
done
$PY scripts/v3_head_a_light.py report
echo "S2M DONE"
