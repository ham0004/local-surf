#!/usr/bin/env bash
# Stronger frozen answerer (Qwen3-VL-4B), research_log step 51. Same pools and selections as the 2B runs;
# only the answerer changes. Dev (three sets) and test (three sets), five systems; plus the 2B follow-up of the
# step-50 transcript policy on test.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
C4="--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_4b.json"
for ds in cgbench videommmu videommmu_comp cgbench_test videommmu_test videommmu_comp_test; do
  case $ds in *_test) U=uniform4; D=dense4;; *) U=""; D="";; esac
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool v2 --selectors clip --k 4 --words 120 $C4
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt --k 4 --words 0 $C4
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt --k 4 --words 600 --tpolicy frames+hybrid $C4
  [ -n "$D" ] && $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool $D --selectors clip --k 4 --words 0 $C4
  [ -n "$U" ] && $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool $U --selectors clip --k 4 --words 0 $C4
done
for ds in cgbench_test videommmu_test videommmu_comp_test; do   # 2B follow-up: step-50 transcript policy on test
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt --k 4 --words 600 --tpolicy frames+hybrid --ledger budget_test_followup.json
done
$PY scripts/v3_baseline_cycle.py report
echo "4B CHECK DONE"
