# Confirmation run on fresh data: protocol (declared before any fresh question is answered)

Written 2026-10-24 (research_log step 62). Purpose: confirm or reject, on questions that played no part in any
decision, the claims that the earlier test left unconfirmed, above all whether the trained heads add accuracy.
Nothing below changes after the first fresh answer.

## Fresh test set
LongVideoBench (NeurIPS 2024 Datasets and Benchmarks; CC BY-NC-SA 4.0), validation split, **every question on a
video not used by v1, v2 or v3**: 501 videos (8 s to 60 min), 897 multiple-choice questions (4–5 options), with the
dataset's subtitles as transcripts (`scripts/research/fetch_lvb_fresh.py`). The dataset's referred moment (the time
span of the subtitle a question quotes) is used to measure whether the shown pictures include that moment.
All framework settings and heads are frozen as of research_log step 61 (no retraining, no tuning on this set).

## Answer models
Qwen3-VL-2B (local), Qwen3-VL-4B (local), Nemotron-3.5-Super-VL 120B (NVIDIA API, `configs/api_nemotron_vl.yaml`).

## Systems (4 pictures each)
| Id | System | Speech path | Picker | Text to the answer model |
|---|---|---|---|---|
| S0 | starting framework (v2 design) | speech rule, windows-only Path B | top match to question | BM25, 120 words |
| S1 | best framework | speech rule | MMR on question + options | none |
| S1t | best framework + speech text | speech rule | MMR | 600-word speech text |
| S2 | + Head A | Head A | MMR | none |
| S3 | + Head B | speech rule | Head B (v1) | none |
| S4 | + Head A + Head B | Head A | Head B (v1) | none |
| S4t | + Head A + Head B + speech text | Head A | Head B (v1) | 600-word speech text |
| R2 | 4 evenly spaced pictures | none | all | none |
| R3 | dense MobileCLIP top-4 | none | all | none |

All systems run with 2B and Nemotron; with 4B all systems run except that the slow 600-word runs (S1t, S4t,
about 28 s per answer) come last and are reported as far as the declared ledger allows.

## Primary hypotheses (two-sided, 95% paired bootstrap over videos, 10,000 resamples, Holm correction over H1-H4)
- **H1 (heads, 4B):** S4 − S1 > 0 with Qwen3-VL-4B, no speech text (dev +4.3, earlier test +2.2).
- **H2 (heads, large model):** S4 − S1 > 0 with Nemotron-120B, no speech text.
- **H3 (framework, 4B):** S1t − S0 > 0 with Qwen3-VL-4B (earlier test +5.2).
- **H4 (framework, large model):** S1t − S0 > 0 with Nemotron-120B.

Secondary (reported, no correction): S2 − S1, S3 − S1, S4t − S1t, S1 − R3, S1 − R2 per answer model; the share of
questions whose shown pictures include the referred moment, per system (S1 vs S2 measures Head A on new data);
time per question.

## Also run with Nemotron-120B (follow-up on the existing sets, not confirmation)
CG-Bench test and Video-MMMU test (459 questions): S0, S1, S1t, S2, S4, S4t; and the ceiling with human-marked
pictures on CG-Bench dev and test.

## Budget
Ledger `runs/v3_baseline/budget_confirm*.json`: 20,000 answer calls / 72 GPU- or API-hours in total.
