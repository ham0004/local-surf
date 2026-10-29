# Finding the Right Moments: Local Question Answering on Long and Lecture Videos

[![tests](https://github.com/ham0004/local-surf/actions/workflows/tests.yml/badge.svg?branch=framework-2)](https://github.com/ham0004/local-surf/actions/workflows/tests.yml)

A fully local framework that answers multiple-choice questions about long videos (10–80 min) and
university lectures on **one 16 GB GPU** with small, frozen open models. A small vision-language
model can only look at a few pictures, so the framework's job is to find **the right moments and the
right speech**: transcript search and a dense whole-video visual search propose candidate frames, a
selector chooses four, and the speech around them is passed to the frozen answer model.

Two trained components target the core problem, going from what is *said* to where the answer is
*shown*:

- **Head A, the speech-to-moment locator:** reads every transcript line and votes for the moments where the
  answer should be visible (1,889 trained parameters over frozen MiniLM and BGE rerankers).
- **Head B, the frame-set selector:** chooses the final set of frames from both proposal paths.

## Main results (locked test set)

The original frozen comparison (2B answer model, systems declared in
[`docs/v3/final_test_protocol.md`](docs/v3/final_test_protocol.md)) was run once on 459 test questions. The 4B and
speech-text rows are **follow-ups** run afterwards on the same questions; the speech policy was changed on dev after the
original test had been seen (research log step 50). Question sets: CG-Bench (long everyday videos), 55; Video-MMMU
Perception (lectures, reading the screen), 202; Video-MMMU Comprehension (lectures, understanding), 202.
Accuracy is multiple choice. "Confirmed" means the 95% paired bootstrap interval excludes zero (10,000 resamples of
whole source videos, 208 videos; both Video-MMMU tracks of one lecture count as one video).

| System | Answer model | All 459 | Long videos | Lecture screen-reading | Lecture understanding |
|---|---|---|---|---|---|
| Starting framework (v2 design) | Qwen3-VL-2B | 38.1 | 34.5 | 47.0 | 30.2 |
| Tuned framework | Qwen3-VL-2B | 41.4 | **50.9** | 48.0 | 32.2 |
| Starting framework (v2 design) | Qwen3-VL-4B | 50.3 | 38.2 | 60.9 | 43.1 |
| Tuned framework + selected speech | Qwen3-VL-4B | **55.6** | 49.1 | **67.8** | 45.0 |

- Tuned vs starting framework, long videos, 2B: **+16.4 points (confirmed, but only 6 test videos)**. Over all 459
  questions the 2B gain is +3.3 (not confirmed).
- Tuned framework + selected speech vs starting framework, 4B (follow-up): **+5.2 points over all 459 (confirmed)**;
  lecture screen-reading **+6.9 (confirmed)**. This changes pictures and text together, so it does not isolate the
  picture choice.
- Selected speech (600 words): for 68% of CG-Bench dev questions at least one chosen speech line lies within 10 s of
  the human-marked answer moment, against 28% for the 120-word BM25 excerpt (different word budgets). With 4B it adds
  +2.8 points (interval 0.0 to +5.6: borderline, not confirmed).
- Head A: one of its 6 proposed moments falls on the human-marked answer moment for 36.6% of 164 CG-Bench dev
  questions (17 videos), against 28.0% for its untrained counterpart (confirmed; the method family was chosen on
  these questions). Its effect on accuracy is under confirmation on new data (see
  `docs/v3/research_log.md`).

Index of every experiment with its script and result: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Every result, one per row:
[`docs/report/all_results.csv`](docs/report/all_results.csv). Full chronological record of every
experiment and decision: [`docs/v3/research_log.md`](docs/v3/research_log.md).

## Framework

```
question + options, video, timed transcript
  -> BM25 transcript search (4 windows of ~60 s)
  -> Path A (speech):   speech rule (MiniLM on lines in the windows)  |  Head A (trained, whole transcript)
     Path B (pictures): 3 frames scanned inside the windows + 3 from a dense 2 s MobileCLIP index of the whole video
  -> merge (~10-12 candidates, near-duplicates removed)
  -> selector: MMR on question + options  |  Head B (trained)  |  confidence choice over 4 sets
  -> speech text: speech around the chosen frames + best lines by BM25 + MiniLM fusion (<= 600 words)
  -> frozen answer model (Qwen3-VL-2B / 4B) -> option letter + timestamp citations
```

Design and protocol: [`docs/v3/baseline_framework.md`](docs/v3/baseline_framework.md),
[`docs/v3/heads_design.md`](docs/v3/heads_design.md),
[`docs/v3/final_test_protocol.md`](docs/v3/final_test_protocol.md),
[`docs/v3/final_test_results.md`](docs/v3/final_test_results.md).

## Repository layout

```
src/videoqa/            pipeline modules (transcript, retrieval, frames, encoders, answerer)
src/videoqa/v2/         candidate pool (two paths), selectors, budgets, answer cache
src/videoqa/v3/         Head A inference, open-ended answering, judges, metrics
scripts/v3_*.py         experiment runners (baseline cycle, Head A, Head B, final test, ceiling)
scripts/run_*.sh        exact command sequences of each declared experiment
scripts/build_*.py      report builders (PDF/Excel are generated locally, not tracked)
scripts/research/       dataset download, speech recognition, preparation
configs/                hardware profiles (2B and 4B answer models) and measured costs
reports/                result summaries (JSON/Markdown) per experiment
docs/v3/                research plan, protocols, research log, results
docs/v2/, docs/history/ earlier versions and their reports
tests/                  unit tests (`uv run pytest -q`)
```

## Reproduce

```bash
uv sync --extra dev --extra models
uv run pytest -q
# data (gated datasets need their usage agreements): scripts/research/fetch_*.py, prepare_cgbench.py
bash scripts/run_baseline_cycle.sh        # dev tuning (2B)
bash scripts/run_final_test.sh            # declared test comparison
bash scripts/run_full_grid.sh             # every model x speech-text x head combination
uv run python scripts/build_results_table.py   # docs/report/all_results.{csv,xlsx,pdf}
```

Answers are cached by request key and every run is budgeted (`runs/*/budget*.json`); `data/`, `runs/`,
`cache/` and checkpoints are not tracked.

## Earlier versions

- **v1** (branch `QAframework`, tag `v1-final`): cost-aware controller; LongVideoBench 440 questions, best
  strategy 41.6% (MobileCLIP frames) vs 32.5% transcript only. The controller hypothesis was not
  supported ([`docs/final_report_controller_study.md`](docs/final_report_controller_study.md),
  [`reports/benchmark_v1/RESULTS.md`](reports/benchmark_v1/RESULTS.md)).
- **v2** (tag `v2-main-experiment`): two proposal paths and first learned heads on self-made MIT lecture
  questions; learned selectors did not beat zero-shot rules ([`docs/v2/`](docs/v2/)).

## Licences

Code: unlicensed for now. Qwen3-VL and MobileCLIP: Apache-2.0. CG-Bench (MIT, usage agreement),
Video-MMMU and LongVideoBench (CC BY-NC-SA 4.0) are not redistributed here.
