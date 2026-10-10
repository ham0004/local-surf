# Final frozen comparison: results (test, run once)

Protocol: `docs/v3/final_test_protocol.md` (declared and pushed before the run). Run 2026-10-10, ledger
`budget_test.json`: 5,370 of 6,000 calls, 6,244 s. Answerer frozen Qwen3-VL-2B, K = 4. 95% CIs: paired bootstrap
over videos, 10,000 resamples. CG-Bench test has only 6 videos, so its CIs are unreliable (6 clusters).

| System | CG-Bench | VMMMU Perception | VMMMU Comprehension | Pooled | CG-Bench evidence |
|---|---|---|---|---|---|
| R1 question only | 20.0 (n=55) | 25.2 (n=202) | 19.8 (n=202) | 22.2 | 0.0 |
| R2 4 evenly spaced frames | 32.7 (n=55) | 40.6 (n=202) | 33.2 (n=202) | 36.4 | 5.5 |
| R3 dense MobileCLIP top-4 | 41.8 (n=55) | 40.6 (n=202) | 31.7 (n=202) | 36.8 | 58.2 |
| S0 v2 default (starting framework) | 34.5 (n=55) | 47.0 (n=202) | 30.2 (n=202) | 38.1 | 34.5 |
| S1 frozen baseline (tuned rules) | 50.9 (n=55) | 48.0 (n=202) | 32.2 (n=202) | 41.4 | 47.3 |
| S2 + Head A | 41.8 (n=55) | 45.0 (n=202) | 33.7 (n=202) | 39.7 | 49.1 |
| S3 + Head B (ev_set) | 47.3 (n=55) | 48.0 (n=202) | 33.2 (n=202) | 41.4 | 50.9 |
| S4 Head A + Head B | 34.5 (n=55) | 52.0 (n=202) | 32.7 (n=202) | 41.4 | 56.4 |
| S5 answer-confidence selection (4 sets) | 47.3 (n=55) | 50.0 (n=202) | 33.2 (n=202) | 42.3 | — |

| Comparison | kind | CG-Bench | VMMMU Perception | VMMMU Comprehension | Pooled |
|---|---|---|---|---|---|
| S1 − S0 | primary | +16.4 (+7.0..+25.9) | +1.0 (-5.0..+6.9) | +2.0 (-3.0..+6.9) | +3.3 (-0.7..+7.0) |
| S2 − S1 | secondary | -9.1 (-15.7..-3.4) | -3.0 (-7.4..+1.5) | +1.5 (-2.5..+5.4) | -1.7 (-4.4..+1.1) |
| S3 − S1 | secondary | -3.6 (-11.5..+4.1) | +0.0 (-5.0..+4.5) | +1.0 (-3.5..+5.4) | +0.0 (-3.0..+3.1) |
| S4 − S1 | secondary | -16.4 (-19.6..-13.0) | +4.0 (-0.5..+8.4) | +0.5 (-4.0..+5.0) | +0.0 (-3.1..+3.3) |
| S5 − S1 | secondary | -3.6 (-7.7..+0.0) | +2.0 (-2.0..+5.9) | +1.0 (-3.0..+5.0) | +0.9 (-1.6..+3.5) |
| S1 − R3 | secondary | +9.1 (+1.9..+15.6) | +7.4 (+2.5..+12.9) | +0.5 (-4.0..+5.4) | +4.6 (+1.3..+7.8) |
| S1 − R2 | secondary | +18.2 (+8.8..+27.3) | +7.4 (+1.5..+13.9) | -1.0 (-6.4..+4.5) | +5.0 (+1.1..+8.9) |

## What the test shows
- **Primary (S1 − S0):** the tuned framework beats the starting v2 framework by +3.3 points pooled over 459
  test questions (CI −0.7..+7.0, just short of significance); the gain is large on long videos (CG-Bench
  +16.4, +7.0..+25.9) and small on lectures (+1.0 / +2.0).
- **External rules:** the tuned framework beats both standard training-free rules with the same answerer:
  dense MobileCLIP top-4 by +4.6 (+1.3..+7.8) and evenly spaced frames by +5.0 (+1.1..+8.9).
- **Heads:** neither learned head improves on the tuned baseline on test (Head B +0.0, Head A −1.7, both
  +0.0, answer-confidence selection +0.9 pooled; all CIs include 0). The heads' evidence localisation does
  improve on CG-Bench (chosen frames inside the human evidence: baseline 47.3%, Head B 50.9%, Head A + B
  56.4%), but with 55 questions on 6 videos this does not show up as accuracy, and Head A + B loses there.
- Dev → test: the rankings of S0 < S1 and "heads ≈ S1" hold; Video-MMMU Perception accuracy is lower on test
  (48–52%) than on dev (62–66%) for every system, a difference between the video sets, not between systems.

## Claims this supports (and does not)
- Supported: on long videos, searching the whole video visually (instead of only BM25-retrieved windows),
  matching frames to question + options, and dropping the BM25 excerpt for a small answerer give a large,
  significant gain over the v2 pipeline; the tuned pipeline significantly beats standard frame-selection
  rules overall.
- Not supported: that the learned Head A or Head B improve end-to-end accuracy. Head A improves moment
  localisation (dev recall +8.5, significant) without an accuracy gain.
