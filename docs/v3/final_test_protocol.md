# Final frozen comparison: protocol (declared before any test question is answered)

Written 2026-10-10, after the Head A cycle, the Head B cycle and the combined grid on dev (research_log
steps 27–46). Nothing below is changed after the first test answer. Test questions are answered once per
system; no setting is chosen on test.

## Test sets (never used before)
| Set | Questions | Videos | Notes |
|---|---|---|---|
| CG-Bench test split | 55 | 6 | our declared video split (sha256 rule); small, reported with wide CIs |
| Video-MMMU Perception test | 202 | 202 | official videos, our declared split |
| Video-MMMU Comprehension test | 202 | 202 | same videos |

Transcripts: CG-Bench English subtitles; Video-MMMU our Whisper large-v3-turbo transcripts (same pipeline as
dev). Dense 2 s MobileCLIP features computed for the test videos before any answer.

## Systems (all with the same frozen Qwen3-VL-2B, same prompt, 4 frames)
| # | System | Pool | Selector | Transcript |
|---|---|---|---|---|
| R1 | question only (reference) | — | no frames | none |
| R2 | 4 evenly spaced frames (external rule) | `uniform4` | all | none |
| R3 | dense MobileCLIP top-4 (external rule, as in the gate) | `dense4` | all | none |
| S0 | v2 default (the starting framework) | `v2` | clip | 120 words |
| S1 | **frozen baseline** (tuned rules) | `hybrid` | mmropt | none |
| S2 | + Head A | `hybrid_ahead` | mmropt | none |
| S3 | + Head B (learned evidence + option term) | `hybrid` | ev_set (λ 2, μ 1) | none |
| S4 | Head A + Head B | `hybrid_ahead` | ev_set (λ 2, μ 1) | none |
| S5 | answer-confidence selection over 4 sets (training-free) | both | verify max | none |

Head A and Head B checkpoints are the ones evaluated on dev (3 seeds averaged); no retraining.

## Metrics and tests
- Multiple-choice accuracy per test set and pooled over the three (n = 459).
- Paired differences with 95% bootstrap CIs over videos (10,000 resamples):
  - primary: S1 − S0 (does the tuned framework beat the starting framework?);
  - secondary: S2, S3, S4, S5 − S1; S1 − R3; S1 − R2.
  No correction is claimed for multiple secondary comparisons; they are reported as such.
- Open-ended answers on CG-Bench test for S0, S1, S5's first set and R3 (short-answer judge, as on dev).
- CG-Bench evidence recall of the chosen frames (human intervals).
- Measured seconds per question.

## Dev results these systems were chosen on (pooled over 278 dev questions, vs S1)
S0 −5.8 (−10.2..−1.0); S2 −0.4; S3 −0.4; S4 −0.7; S5 +1.1 (−1.7..+3.8). Expected on test: S1 > S0; heads
and S5 within a few points of S1.

## Budget
One ledger for the whole comparison, `runs/v3_baseline/budget_test.json`: 6,000 answer calls / 4 GPU-hours
(about 12 calls per question × 459 questions: 8 single-set systems + 4 sets for S5). Open-ended answers and
judge verdicts use their existing ledgers.
