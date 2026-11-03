#!/usr/bin/env bash
# Lecture study, TEST, secondary answer-model evaluation (docs/v3/m3av_protocol.md; commands and rules sealed in
# docs/v3/m3av_freeze_supplement3.json, auditor P103). Checks the whole freeze chain, the test inputs and artifacts,
# the primary report and the QA code; on the first start, refuses existing test QA outputs. Resumable: a restart
# continues from the saved records. Seals the records before the report. One GPU job at a time.
set -euo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
$PY scripts/m3av_freeze_supplement3.py verify
if [ ! -e runs/m3av/qa_test_started ]; then
  for x in runs/m3av/qa/test docs/v3/m3av_qa_test_outputs.json reports/m3av/qa_test.json; do
    [ ! -e "$x" ] || { echo "exists before the sealed run: $x"; exit 1; }
  done
  date -u +%FT%TZ > runs/m3av/qa_test_started
fi
$PY scripts/m3av_qa.py answer --split test
$PY scripts/m3av_qa.py seal --split test
$PY scripts/m3av_qa.py report --split test
echo "M3AV TEST QA DONE"
