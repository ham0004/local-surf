#!/usr/bin/env bash
# Phase 1 checks (research_log steps 74 and 77), Qwen3-VL-4B, K = 4, run once; finished answers come from the cache.
# Order: CG-Bench dev first (where any decision is made), then the fast no-text ablations on the fresh set, then the
# slow 600-word speech arms on the fresh set (exploratory follow-ups: the fresh set's results were already seen).
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
M="--config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_phase1_4b.json"
ans() { $PY scripts/v3_baseline_cycle.py answer --k 4 "$@" $M; }

# 1. CG-Bench dev: Head A with its residual zeroed (same density-voting moments), the broader zero-shot pool,
#    and the matched-budget speech arms (frames+hybrid at 600 words already exists there)
$PY scripts/v3_baseline_cycle.py pools --dataset cgbench --pool hybrid_ahead_r0
ans --dataset cgbench --pool hybrid_ahead_r0 --selectors mmropt --words 0
ans --dataset cgbench --pool hybrid_azs --selectors mmropt --words 0
for tp in bm25full hybrid frames; do ans --dataset cgbench --pool hybrid --selectors mmropt --words 600 --tpolicy $tp; done
# 2. fresh set, no text: Head A residual zeroed; Head B without its option term (lam = 0, set context kept)
D=lvb_fresh
$PY scripts/v3_baseline_cycle.py pools --dataset $D --pool hybrid_ahead_r0
ans --dataset $D --pool hybrid_ahead_r0 --selectors mmropt --words 0
for pool in hybrid hybrid_ahead; do
  $PY scripts/v3_head_b.py select --dataset $D --pool $pool --train-pool hybrid --method ev_set --lam 0.0 --mu 1.0 --tag nolam
  ans --dataset $D --pool $pool --selectors file:${D}_${pool}_ev_set_nolam --words 0
done
# 3. fresh set, 600-word speech arms and the H3 decomposition (slow: about 5 hours each)
for tp in bm25full hybrid frames; do ans --dataset $D --pool hybrid --selectors mmropt --words 600 --tpolicy $tp; done
ans --dataset $D --pool hybrid --selectors mmropt --words 120                 # S1 pictures + v2 BM25 excerpt
ans --dataset $D --pool v2 --selectors clip --words 600 --tpolicy frames+hybrid   # S0 pictures, S1t text policy
echo "PHASE1 DONE"
