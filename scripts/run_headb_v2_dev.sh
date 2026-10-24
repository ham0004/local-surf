#!/usr/bin/env bash
# research_log step 60: Head B v2 screen on dev (2B and 4B, no speech text). Head A pools, Head B v2 trained on
# cross-fitted Head A training pools; settings: speech-path cap 4 or 2, time gap 0 / 30 / 60 s.
set -u
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
TAGS="v2g0q4 v2g0q2 v2g30q2 v2g60q2"
for ds in cgbench videommmu videommmu_comp; do
  for t in $TAGS; do g=${t#v2g}; g=${g%q*}; q=${t##*q}
    $PY scripts/v3_head_b.py select --dataset $ds --pool hybrid_ahead --train-pool hybrid_ahead_cf --method ev_set --lam 2.0 --mu 1.0 --gap $g --a-quota $q --tag $t > /dev/null
  done
  S=""; for t in $TAGS; do S="$S,file:${ds}_hybrid_ahead_ev_set_$t"; done
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid_ahead --selectors ${S#,} --k 4 --words 0 --ledger budget_headb_v2.json
  $PY scripts/v3_baseline_cycle.py answer --dataset $ds --pool hybrid_ahead --selectors ${S#,} --k 4 --words 0 --config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_headb_v2_4b.json
done
echo "HEADB V2 DEV DONE"
