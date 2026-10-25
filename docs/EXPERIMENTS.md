# Experiments index

Every experiment of the project in the order it was done: what it tested, how to rerun it, where the result
is, and what it showed. Detailed numbers and every decision are in [`v3/research_log.md`](v3/research_log.md);
every result, one per row, is in [`report/all_results.csv`](report/all_results.csv).

**Words used here**
- **Answer model:** the frozen vision-language model that reads 4 pictures (and optional text) and picks an
  answer: Qwen3-VL-2B or Qwen3-VL-4B (local), or Nemotron-3.5-Super-VL 120B (NVIDIA API).
- **Best framework:** transcript search (BM25) → speech path (MiniLM on lines inside the matched windows) and
  picture path (half from the windows, half from a dense whole-video MobileCLIP index) → merge → MMR picker on
  question + options → 4 pictures. With **speech text**, the answer model also reads up to 600 words: the speech
  around the 4 pictures plus the best lines by BM25 + MiniLM.
- **Head A:** trained speech-to-moment locator (replaces the speech path). **Head B:** trained frame-set selector
  (replaces the MMR picker).
- **dev:** questions used to choose settings. **test:** questions locked until the end and used once.

## Run the best framework

```bash
# once per dataset: picture index (2 s MobileCLIP fingerprints) and candidate pools
uv run python scripts/v3_features.py --dataset videommmu --split test
uv run python scripts/v3_baseline_cycle.py pools --dataset videommmu_test --pool hybrid
# answer: best framework + 600-word speech text, 4B answer model
uv run python scripts/v3_baseline_cycle.py answer --dataset videommmu_test --pool hybrid --selectors mmropt \
    --k 4 --words 600 --tpolicy frames+hybrid --config configs/gpu_16gb_4b.yaml --tag ans=4b --ledger budget_demo.json
uv run python scripts/v3_baseline_cycle.py report
```

Answer-model configs: `configs/gpu_12gb.yaml` (2B), `configs/gpu_16gb_4b.yaml` (4B),
`configs/api_nemotron_vl.yaml` (Nemotron-120B; key in `~/.nvidia_api_key`). Add Head A with `--pool hybrid_ahead`,
Head B with `--selectors file:<dataset>_<pool>_ev_set` after `scripts/v3_head_b.py select`.

## Experiments

| # | Experiment | Question it answered | Run with | Result file | Outcome |
|---|---|---|---|---|---|
| 1 | v1 benchmark (tag `v1-final`) | Which picture-selection strategy works with a 2B model? | `scripts/run_lvb_benchmark.sh` | `reports/benchmark_v1/RESULTS.md` | MobileCLIP pictures 41.6% vs transcript only 32.5% (LongVideoBench, 440 q) |
| 2 | v1 controller study | Can a controller learn when to look from damaged speech? | see `docs/final_report_controller_study.md` | same file | Not supported |
| 3 | v2 main experiment (tag `v2-main-experiment`) | Do learned heads beat rules on lecture questions? | `docs/v2/README.md` | `docs/v2/main_report.md` | No; zero-shot rule best (self-made MIT questions) |
| 4 | v3 benchmark check | Can pictures change the answer on each benchmark? | `scripts/v3_gate_frames.py`, `v3_gate_videommmu.py`, `v3_gate_cgbench.py` | `docs/v3/gate_*_report.md` | Yes on Video-MMMU and CG-Bench; no on EduVidQA |
| 5 | Baseline tuning (dev) | Which settings make the best rule-based framework? | `scripts/run_baseline_cycle.sh`, `run_baseline_dense.sh`, `run_baseline_stage3.sh` | `reports/v3_baseline/table.json` | Hybrid picture path + MMR on question + options + no excerpt (tag `v3-baseline-frozen`) |
| 6 | Head A | Can a small trained model find the answer moment from speech? | `scripts/v3_head_a_data.py`, `v3_head_a_eval.py`, `v3_head_a_light.py` | `reports/v3_head_a/*.json` | Moment found 36.6% vs 28.0% of new questions (confirmed) |
| 7 | Head B | Can a trained picker beat the MMR rule? | `scripts/v3_head_b.py`, `v3_head_b_labels.py` | research log steps 41–46 | No clear gain |
| 8 | Final locked test (tag `v3-final-test`) | Declared comparison of 9 systems | `scripts/run_final_test.sh` | `docs/v3/final_test_results.md` | Best framework +16.4 on long videos (confirmed); heads no gain |
| 9 | Speech text | How much relevant speech reaches the answer model? | `scripts/run_text_policies.sh` | research log steps 47–50 | 68% kept vs 28%; open-ended +8.5 (dev) |
| 10 | 4B answer model (tag `v3-best-speech-4b`) | Does the framework still help a stronger model? | `scripts/run_4b_check.sh`, `run_4b_finish.sh` | research log steps 51–53 | 55.6% vs 50.3% on 459 test questions (confirmed) |
| 11 | Full grid | Every answer model × speech text × head combination | `scripts/run_full_grid.sh` | research log step 55 | Heads help 4B only without speech text; not confirmed on test |
| 12 | Ceiling | Accuracy with pictures from the human-marked answer seconds | `scripts/v3_ceiling.py`, `run_oracle_ceiling.sh` | `reports/v3_ceiling/summary.json` | CG-Bench test: 2B 61.8, 4B 56.4, Nemotron-120B 69.1 (oracle ceiling running) |
| 13 | Head B v2 | Matched training pools, time gap, speech-path cap | `scripts/run_headb_v2_dev.sh` | research log steps 58–61 | Not better than Head B v1 |
| 14 | Confirmation on fresh data | Do the unconfirmed claims hold on 894 new LongVideoBench questions? | `scripts/run_confirmation.sh`, `run_confirmation_api.sh` | research log steps 62–65; protocol `docs/v3/confirmation_protocol.md` | Running. 4B: Head A + Head B −0.2 (not supported); best framework +7.6 over the starting design (confirmed) but equal to dense MobileCLIP top-4 |

## Where things are

- Pipeline code: `src/videoqa/` (v2 candidate pools in `src/videoqa/v2/`, Head A inference and judges in
  `src/videoqa/v3/`, API answer model in `src/videoqa/answerer_api.py`).
- Experiment runners: `scripts/v3_*.py`; exact command sequences: `scripts/run_*.sh`; report builders:
  `scripts/build_*.py` (PDF and Excel are generated locally).
- Data, run outputs, model caches and checkpoints live in `data/`, `runs/`, `cache/` (not in git).
