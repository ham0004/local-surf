#!/usr/bin/env bash
# research_log step 54: every combination of answer model {2B, 4B} x speech text {none, 600 words} x
# {best framework, + Head A, + Head B, + Head A + Head B}, on dev and test (the 7 combinations not yet run).
# Settings are unchanged (fixed earlier on dev); test numbers are follow-ups, reported apart from the declared test.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
SETS="cgbench videommmu videommmu_comp cgbench_test videommmu_test videommmu_comp_test"
run() {  # $1 = extra args (answerer), $2 = transcript args
  for ds in $SETS; do
    $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors file:${ds}_hybrid_ev_set --k 4 $2 $1
    $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid_ahead --selectors mmropt,file:${ds}_hybrid_ahead_ev_set --k 4 $2 $1
  done
}
T600="--words 600 --tpolicy frames+hybrid"
T0="--words 0"
run "--ledger budget_grid_2b.json" "$T600"                                                  # 2B + speech text
run "--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_grid_4b.json" "$T600"   # 4B + speech text
run "--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_grid_4b.json" "$T0"     # 4B, no speech text
$PY scripts/v3_baseline_cycle.py report > /dev/null
echo "FULL GRID DONE"
