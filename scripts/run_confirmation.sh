#!/usr/bin/env bash
# Confirmation run on fresh LongVideoBench questions (docs/v3/confirmation_protocol.md). Run once.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
D=lvb_fresh
$PY scripts/v3_features.py --dataset lvb_fresh --split test
for pool in v2 hybrid hybrid_ahead uniform4 dense4; do $PY scripts/v3_baseline_cycle.py pools --dataset $D --pool $pool; done
for pool in hybrid hybrid_ahead; do
  $PY scripts/v3_head_b.py features --dataset $D --pool $pool
  $PY scripts/v3_head_b.py select --dataset $D --pool $pool --train-pool hybrid --method ev_set --lam 2.0 --mu 1.0
done
run() {  # $1 = answer-model args
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool v2 --selectors clip --k 4 --words 120 $1                       # S0
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid --selectors mmropt,file:${D}_hybrid_ev_set --k 4 --words 0 $1      # S1, S3
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid_ahead --selectors mmropt,file:${D}_hybrid_ahead_ev_set --k 4 --words 0 $1   # S2, S4
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool uniform4 --selectors clip --k 4 --words 0 $1                    # R2
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool dense4 --selectors clip --k 4 --words 0 $1                      # R3
}
runtext() {
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid --selectors mmropt --k 4 --words 600 --tpolicy frames+hybrid $1                       # S1t
  $PY scripts/v3_baseline_cycle.py answer --dataset $D --pool hybrid_ahead --selectors file:${D}_hybrid_ahead_ev_set --k 4 --words 600 --tpolicy frames+hybrid $1   # S4t
}
run "--ledger budget_confirm_2b.json"
runtext "--ledger budget_confirm_2b.json"
run "--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_confirm_4b.json"
runtext "--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_confirm_4b.json"
echo "CONFIRMATION LOCAL DONE"
