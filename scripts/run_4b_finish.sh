#!/usr/bin/env bash
# research_log step 52: finish the 4B test runs on a second ledger (cached answers are reused).
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
C4="--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_4b_2.json"
$PY scripts/v3_baseline_cycle.py answer --dataset videommmu_test --pool hybrid --selectors mmropt --k 4 --words 600 --tpolicy frames+hybrid $C4
for s in "v2 clip 120 bm25" "hybrid mmropt 0 bm25" "hybrid mmropt 600 frames+hybrid"; do set -- $s
  $PY scripts/v3_baseline_cycle.py answer --dataset videommmu_comp_test --pool $1 --selectors $2 --k 4 --words $3 --tpolicy $4 $C4; done
for ds in videommmu_test videommmu_comp_test; do for pool in dense4 uniform4; do
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool $pool --selectors clip --k 4 --words 0 $C4; done; done
echo "4B FINISH DONE"
