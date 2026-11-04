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
| 6 | Head A | Can a small trained model find the answer moment from speech? | `scripts/v3_head_a_data.py`, `v3_head_a_eval.py`, `v3_head_a_light.py` | `reports/v3_head_a/*.json` | One of 6 proposed moments on the marked moment: 36.6% vs 28.0% of 164 CG-Bench dev questions (17 videos); the method family was chosen on these questions, so this is a dev result |
| 7 | Head B | Can a trained picker beat the MMR rule? | `scripts/v3_head_b.py`, `v3_head_b_labels.py` | research log steps 41–46 | No clear gain |
| 8 | Final locked test (tag `v3-final-test`) | Declared comparison of 9 systems | `scripts/run_final_test.sh` | `docs/v3/final_test_results.md` | Best framework +16.4 on CG-Bench test (confirmed, but only 6 test videos); +3.3 over all 459 test questions (not confirmed); heads no gain |
| 9 | Speech text | How much relevant speech reaches the answer model? | `scripts/run_text_policies.sh` | research log steps 47–50 | Share of dev questions with a chosen line within 10 s of the marked moment: higher than v2's excerpt, but at a different word budget (600 vs 120); matched-budget comparison planned. Open-ended +8.5 (dev) |
| 10 | 4B answer model (tag `v3-best-speech-4b`) | Does the framework still help a stronger model? | `scripts/run_4b_check.sh`, `run_4b_finish.sh` | research log steps 51–53 | 55.6% vs 50.3% on 459 test questions, +5.2 (+1.1..+9.4, confirmed); a follow-up after the original 2B comparison. Speech on top of tuned pictures +2.8, not confirmed |
| 11 | Full grid | Every answer model × speech text × head combination | `scripts/run_full_grid.sh` | research log step 55 | Heads help 4B only without speech text; not confirmed on test |
| 12 | Ceiling | Accuracy with pictures from the human-marked answer seconds | `scripts/v3_ceiling.py`, `run_oracle_ceiling.sh` | `reports/v3_ceiling/summary.json` | CG-Bench test: 2B 61.8, 4B 56.4, Nemotron-120B 69.1. Oracle (complete; any of 5 placements inside the marked seconds, no text): test 70.9 / 61.8 / 81.8, dev 70.7 / 72.0 / 79.3; a lenient best-of-five diagnostic, not an attainable ceiling |
| 13 | Head B v2 | Matched training pools, time gap, speech-path cap | `scripts/run_headb_v2_dev.sh` | research log steps 58–61 | Not better than Head B v1 |
| 14 | Confirmation on fresh data | Do the unconfirmed claims hold on 894 new LongVideoBench questions? | `scripts/run_confirmation.sh`, `run_confirmation_api.sh`, `scripts/v3_confirmation_report.py` | `reports/v3_confirmation/summary.md`; research log steps 62–65, 72, 76; protocol `docs/v3/confirmation_protocol.md`; deviations `docs/v3/confirmation_deviations.md` | Final (Holm over H1–H4). Heads − best framework: 4B −0.2 (−3.0..+2.7), Nemotron +2.1 (−0.8..+5.1), not supported. Best framework + speech text − starting design: 4B +8.9 (+5.7..+12.2), Nemotron +8.3 (+4.8..+11.8), supported (p ≈ 0.0002, Holm ≈ 0.0008). No demonstrated advantage over dense MobileCLIP top-4 with any answer model (−0.8, −0.1, −1.1) |
| 15 | Lecture study (M3AV), primary | Does Head A trained on lectures put the source slide among the 4 pictures more often than the best framework and the strongest training-free rule? | `scripts/run_m3av_test.sh` (protocol `docs/v3/m3av_protocol.md`, freeze `docs/v3/m3av_freeze*.json`) | `reports/m3av/hits_test.md`; research log steps 79–90 | Source-slide interval hit@4 (automatic visibility proxy), 853 test questions, 112 lectures, Holm over two: vs best framework +5.3 (+2.7..+7.8, confirmed; the whole Head A package, not the learned part alone); vs VSI-inspired R5 −0.5 (−3.8..+2.9, no confirmed difference). Human review of 100 questions pending |
| 16 | Lecture study (M3AV), answers (secondary) | Do the same systems answer more lecture questions with the 4B answer model? | `scripts/run_m3av_qa_test.sh` (third freeze supplement) | `reports/m3av/qa_test.md`; deviation `docs/v3/m3av_qa_test_deviation.md`; research log steps 91–94 | Descriptive, uncorrected: S2m 87.2% vs best framework 82.5% (+4.7, +2.5..+6.9) and R5 86.5% (+0.7, −1.7..+3.0). No confirmed improvement from lecture-specific training, Head B or speech text. Question-only control 51.2%, transcript-only 62.1%. One control answer re-attempted after a crash (a deviation; full-cohort results and the sensitivity interpretation unchanged; the transcript-only mask has 322 instead of 323 questions) |

## Where things are

- Pipeline code: `src/videoqa/` (v2 candidate pools in `src/videoqa/v2/`, Head A inference and judges in
  `src/videoqa/v3/`, API answer model in `src/videoqa/answerer_api.py`).
- Experiment runners: `scripts/v3_*.py`; exact command sequences: `scripts/run_*.sh`; report builders:
  `scripts/build_*.py` (PDF and Excel are generated locally).
- Data, run outputs, model caches and checkpoints live in `data/`, `runs/`, `cache/` (not in git).
