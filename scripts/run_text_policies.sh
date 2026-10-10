#!/usr/bin/env bash
# Transcript policies on dev (research_log step 49): frozen baseline frames (hybrid pool, mmropt, K = 4),
# only the transcript changes. Multiple choice on three dev sets.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
L="--ledger budget_text.json"
for ds in cgbench videommmu videommmu_comp; do
  for w in 120 300; do $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt --k 4 --words $w --tpolicy hybrid $L; done
  for w in 300 600; do $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt --k 4 --words $w --tpolicy frames+hybrid $L; done
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid --selectors mmropt --k 4 --words 100000 --tpolicy full $L
done
$PY scripts/v3_baseline_cycle.py report
echo "TEXT POLICIES DONE"
