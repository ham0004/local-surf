# video-understanding — when speech misses the evidence

Research prototype for **low-cost, evidence-grounded video question answering**. A small
controller decides *whether and where* to look in a video, using the transcript and cheap
visual hints. A frozen vision-language model then answers from the selected real frames and a
short verbatim transcript excerpt, with timestamp citations and a full cost trace.

**Hypothesis under test (not a result):** training the controller on the *same* video and
question under three transcripts — clean, answer-relevant speech damaged, and an equal amount
of unrelated speech damaged — teaches it to spend visual effort when the missing speech
*matters*, without overspending when unrelated speech is missing. See
[docs/novelty_matrix.md](docs/novelty_matrix.md) for how this differs from the closest prior
work, and [docs/architecture.md](docs/architecture.md) for a plain-language tour of the code.

## Status

| Milestone | State |
|---|---|
| M0 prior-art + runtime audit | done — [literature](docs/literature.md), [novelty](docs/novelty_matrix.md), [environment](docs/environment.md), [models](docs/models.md), [data](docs/DATA_FEASIBILITY.md) |
| M1 frozen baseline end to end | done — synthetic lecture (GPU + CPU profiles in `reports/`) and real LongVideoBench videos |
| M2 measurable acquisition | done: windows, candidates, scout, typed actions, hard caps, rescue, packing, evidence ledger, contact sheet |
| M3 labels + training | done — real frozen-VLM labels on LongVideoBench (330 fair question triples) |
| M4 evaluation on real benchmarks | LongVideoBench done (252 videos / 440 questions): **null result, and the dataset does not satisfy the hypothesis's premise** — see [reports/lvb_full/RESULTS.md](reports/lvb_full/RESULTS.md). EduVidQA / Video-MME not started |
| M5 reproducibility | this README, pinned `uv.lock`, configs; results tables only from logs |

Details, open problems and measured numbers: [docs/progress.md](docs/progress.md).

## Setup

```bash
uv sync --extra dev                    # core + tests (CPU)
uv sync --extra dev --extra models     # + torch (CUDA 12.8), transformers, open_clip
uv run pytest -q            # 148 tests
```

## Commands (all tested)

```bash
# 1. Synthetic lectures for plumbing checks (NOT research data)
uv run videoqa make-synthetic --out data/synthetic --n 40

# 2. Ask one question (writes result.json + contact_sheet.png)
uv run videoqa ask --video data/synthetic/videos/synth_0000.mp4 \
    --transcript data/synthetic/transcripts/synth_0000.json \
    --question "What final accuracy is shown on the results slide?" \
    --options "67%|77%|87%|97%" --config configs/gpu_12gb.yaml --out runs/ask/demo

# 3. Automatic action-usefulness labels (train videos only; dev set for threshold tuning)
uv run videoqa build-labels --data data/synthetic --split train --config configs/gpu_12gb.yaml --out runs/labels/train
uv run videoqa build-labels --data data/synthetic --split dev   --config configs/gpu_12gb.yaml --out runs/labels/dev

# 4. Train paired and unpaired (ablation) utility heads; tune STOP on dev
uv run videoqa train-controller --train-labels runs/labels/train --dev-labels runs/labels/dev --out checkpoints/run1

# 5. Evaluate policies under clean / targeted / control / asr_noise transcripts
uv run videoqa evaluate --data data/synthetic --split test --config configs/gpu_12gb.yaml \
    --policies transcript_only,uniform,retrieval,scout_similarity,heuristic,learned=checkpoints/run1/head_paired.npz,learned=checkpoints/run1/head_unpaired.npz \
    --out runs/eval/run1

# 6. Cold vs warm cost on this device
uv run videoqa profile --video ... --transcript ... --question ... --config configs/gpu_12gb.yaml

# Real data: LongVideoBench (needs `hf auth login` + accepted dataset terms), then the full run
HF_TOKEN=... uv run python scripts/download_longvideobench_subset.py --out data/longvideobench_full
bash scripts/run_lvb_full.sh

# Dataset feasibility audit (live API calls, no downloads)
uv run python scripts/audit_datasets.py --out reports/data_audit.json
```

Configs: `cpu.yaml` (test double, CI only), `cpu_vlm.yaml` (real VLM on CPU, slow),
`gpu_8gb.yaml` (unmeasured), `gpu_12gb.yaml` (measured on an RTX 5060 Ti 16 GB).

## Layout

```
src/videoqa/   one module per stage (see docs/architecture.md)
configs/       hardware profiles (YAML with inheritance)
tests/         148 tests incl. real-mp4 and local-HTTP-server integration tests
scripts/       dataset audit
docs/          research audit, data feasibility, environment, models, architecture, progress
reports/       measured artefacts committed to git (audits, device profiles)
data/ cache/ runs/ checkpoints/   git-ignored
```

## Ground rules this repo follows

* No invented results: every number in `docs/` and `reports/` comes from a logged run.
* The synthetic lectures and the fixture answerer exist for plumbing tests; their numbers are
  never research evidence.
* Splits are by original video before transcript variants are created; labels are built on
  train videos only; the test split is touched once.
