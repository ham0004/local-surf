#!/usr/bin/env bash
# Nemotron-120B part of the confirmation run (API; needs the pools and selections built by run_confirmation.sh),
# then the follow-up on the existing test sets.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
A="--config configs/api_nemotron_vl.yaml --tag ans=nemo --ledger budget_confirm_api.json"
for D in lvb_fresh cgbench_test videommmu_test videommmu_comp_test; do
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool v2 --selectors clip --k 4 --words 120 $A
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid --selectors mmropt --k 4 --words 0 $A
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid_ahead --selectors mmropt,file:${D}_hybrid_ahead_ev_set --k 4 --words 0 $A
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid --selectors mmropt --k 4 --words 600 --tpolicy frames+hybrid $A
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid_ahead --selectors file:${D}_hybrid_ahead_ev_set --k 4 --words 600 --tpolicy frames+hybrid $A
  if [ "$D" = lvb_fresh ]; then
    $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid --selectors file:${D}_hybrid_ev_set --k 4 --words 0 $A
    $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool uniform4 --selectors clip --k 4 --words 0 $A
    $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool dense4 --selectors clip --k 4 --words 0 $A
  fi
done
echo "CONFIRMATION API DONE"
