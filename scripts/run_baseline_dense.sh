#!/usr/bin/env bash
# Baseline cycle, added trial (research_log step 30): Path B from dense whole-video features ("dense" pool),
# same selectors, K and transcript budget as stages 1-2, plus the path ablation on it.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
for ds in cgbench videommmu; do
  $PY scripts/v3_baseline_cycle.py pools --dataset "$ds" --pool dense || echo "POOLS FAILED $ds dense"
  $PY scripts/v3_baseline_cycle.py answer --dataset "$ds" --pool dense \
      --selectors clip,clipopt,mmr,mmropt,relevance,A_only,B_only --k 4 --words 120 || echo "ANSWER FAILED $ds dense"
done
$PY scripts/v3_baseline_cycle.py report
echo "BASELINE DENSE DONE"
