#!/usr/bin/env bash
# Lecture study, TEST, primary measure (docs/v3/m3av_protocol.md; frozen in docs/v3/m3av_freeze.json and
# docs/v3/m3av_freeze_supplement.json). Run only after the auditor's go-ahead. Checks the models, code and test
# inputs against the freeze; on the first start, refuses any test artifact that already exists, so every artifact
# is built by this frozen run (a restart resumes it: finished files are skipped); records the artifact hashes
# before anything is scored. One GPU job at a time.
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
DS=m3av_test
$PY scripts/m3av_freeze_supplement.py verify --split test --stage inputs
if [ ! -e runs/m3av/test_started ]; then
  for x in data/m3av_test/features runs/v3_baseline/m3av_test runs/m3av/vsi/test_lam0.5_qo_max.json \
           runs/v3_head_b/features_m3av_test_hybrid.npy runs/v3_head_b/selections/m3av_test_hybrid_ev_set.json \
           docs/v3/m3av_manifest_test_artifacts.json reports/m3av/hits_test.json; do
    [ ! -e "$x" ] || { echo "exists before the frozen run: $x"; exit 1; }
  done
  date -u +%FT%TZ > runs/m3av/test_started
fi
$PY scripts/v3_features.py --dataset $DS --step 2
$PY scripts/m3av_vsi.py --split test --setting lam0.5_qo_max
for pool in v2 hybrid hybrid_ahead hybrid_ahead_m3av uniform4d dense4x dense4ox vsi4x; do
  $PY scripts/v3_baseline_cycle.py pools --dataset $DS --pool $pool
done
$PY scripts/v3_head_b.py features --dataset $DS --pool hybrid
$PY scripts/v3_head_b.py select --dataset $DS --pool hybrid --train-pool hybrid --method ev_set --lam 2.0 --mu 1.0
$PY scripts/m3av_freeze_supplement.py manifest --split test --stage artifacts
$PY scripts/m3av_hits.py --split test
echo "M3AV TEST PRIMARY DONE"
