#!/usr/bin/env bash
# research_log step 66: oracle ceiling (5 placements of 4 pictures inside the human-marked seconds), CG-Bench dev
# and test, with and without the 600-word speech text. $1 = local | api
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
if [ "${1:-local}" = local ]; then
  $PY scripts/v3_ceiling.py answer --oracle --config configs/gpu_12gb.yaml --ledger budget_oracle_2b.json
  $PY scripts/v3_ceiling.py answer --oracle --config configs/gpu_16gb_4b.yaml --ledger budget_oracle_4b.json
else
  $PY scripts/v3_ceiling.py answer --oracle --config configs/api_nemotron_vl.yaml --ledger budget_oracle_api.json
fi
$PY scripts/v3_ceiling.py report > /dev/null
echo "ORACLE ${1:-local} DONE"
