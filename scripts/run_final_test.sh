#!/usr/bin/env bash
# Final frozen comparison (docs/v3/final_test_protocol.md). Run once, after test transcripts and features exist.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
L="--ledger budget_test.json"
for ds in cgbench_test videommmu_test videommmu_comp_test; do
  for pool in v2 hybrid hybrid_ahead uniform4 dense4; do $PY scripts/v3_baseline_cycle.py pools --dataset $ds --pool $pool; done
  for pool in hybrid hybrid_ahead; do
    $PY scripts/v3_head_b.py features --dataset $ds --pool $pool
    $PY scripts/v3_head_b.py select --dataset $ds --pool $pool --method optset --lam 1.0 --mu 0.3
    $PY scripts/v3_head_b.py select --dataset $ds --pool $pool --method ev_set --lam 2.0 --mu 1.0
  done
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors none --k 4 --words 0 $L        # R1
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool uniform4 --selectors clip --k 4 --words 0 $L     # R2
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool dense4 --selectors clip --k 4 --words 0 $L       # R3
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool v2 --selectors clip --k 4 --words 120 $L         # S0
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt,file:${ds}_hybrid_ev_set --k 4 --words 0 $L        # S1, S3
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid_ahead --selectors mmropt,file:${ds}_hybrid_ahead_ev_set --k 4 --words 0 $L  # S2, S4
  $PY scripts/v3_baseline_cycle.py verify --dataset $ds --k 4 --sets 4 $L                                        # S5
done
$PY scripts/v3_baseline_cycle.py report
echo "FINAL TEST DONE"
