#!/usr/bin/env bash
# Baseline cycle stage 3 (research_log step 33) on the chosen setting: hybrid pool + mmropt selector.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
for ds in cgbench videommmu; do
  for k in 2 8; do $PY scripts/v3_baseline_cycle.py answer --dataset "$ds" --pool hybrid --selectors mmropt --k $k --words 120; done
  for w in 0 300; do $PY scripts/v3_baseline_cycle.py answer --dataset "$ds" --pool hybrid --selectors mmropt --k 4 --words $w; done
  for pool in hybrid_a3 hybrid_b12; do
    $PY scripts/v3_baseline_cycle.py pools --dataset "$ds" --pool $pool
    $PY scripts/v3_baseline_cycle.py answer --dataset "$ds" --pool $pool --selectors mmropt --k 4 --words 120
  done
done
$PY scripts/v3_baseline_cycle.py report
echo "BASELINE STAGE 3 DONE"
