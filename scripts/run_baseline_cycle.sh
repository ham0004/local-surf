#!/usr/bin/env bash
# Baseline cycle, stages 1-2 (declared in docs/v3/research_log.md step 27). Stage 3 (K and transcript budget)
# is run after stage 1 picks the best pool + selector. Resumable: pools and answers are cached.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
for ds in cgbench videommmu; do
  for pool in v2 balanced video; do
    $PY scripts/v3_baseline_cycle.py pools --dataset "$ds" --pool "$pool" || echo "POOLS FAILED $ds $pool"
    $PY scripts/v3_baseline_cycle.py answer --dataset "$ds" --pool "$pool" \
        --selectors clip,clipopt,mmr,mmropt,relevance --k 4 --words 120 || echo "ANSWER FAILED $ds $pool"
  done
  $PY scripts/v3_baseline_cycle.py answer --dataset "$ds" --pool v2 --selectors A_only,B_only --k 4 --words 120 \
      || echo "ABLATION FAILED $ds"
done
$PY scripts/v3_baseline_cycle.py report
echo "BASELINE STAGES 1-2 DONE"
