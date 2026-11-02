# Source-slide interval hit@4, M3AV test (853 questions, 112 lectures)

Paired bootstrap over lectures (10,000 resamples, seed 20261101); every frozen system covers every question.

## Primary contrasts (amendment 4; Holm over L1 and L2)

| Contrast | Difference, points (95% interval) | raw p | Holm p | significant at 0.05 |
|---|---|---|---|---|
| L1: S2m Head A (M3AV train) - R5 VSI-inspired (96 ranked) | -0.5 (-3.8..+2.9) | 0.8253 | 0.8253 | no |
| L2: S2m Head A (M3AV train) - S1 best framework (also S1t's frames) | +5.3 (+2.7..+7.8) | 0.0002 | 0.0004 | yes |

## Secondary contrasts (no correction; descriptive)

| Contrast | Difference, points (95% interval) | raw p |
|---|---|---|
| S2m Head A (M3AV train) - R3 dense MobileCLIP top-4 (96 ranked) | +15.2 (+11.3..+19.1) | 0.0002 |
| S2m Head A (M3AV train) - R4 dense option-aware top-4 (96 ranked) | +2.5 (-1.1..+6.0) | 0.178 |
| S2m Head A (M3AV train) - S2c Head A (CG-Bench) | -0.4 (-2.1..+1.5) | 0.7555 |

## Systems

| System | hit@4 (%) | frames shown (mean) | questions with < 4 frames |
|---|---|---|---|
| S0 starting design | 67.4 | 3.95 | 12 |
| S1 best framework (also S1t's frames) | 74.1 | 4.0 | 1 |
| S2c Head A (CG-Bench) | 79.7 | 4.0 | 0 |
| S2m Head A (M3AV train) | 79.4 | 4.0 | 0 |
| S3 Head B | 73.6 | 4.0 | 1 |
| R2 4 evenly spaced | 22.9 | 4.0 | 0 |
| R3 dense MobileCLIP top-4 (96 ranked) | 64.1 | 3.87 | 52 |
| R4 dense option-aware top-4 (96 ranked) | 76.9 | 3.88 | 48 |
| R5 VSI-inspired (96 ranked) | 79.8 | 3.94 | 27 |

## Sensitivity: S-src hit@4 (descriptive, not confirmatory)

Visible spans clipped to the question's source segment; other occurrences ignored; same frames. 3 of 853 questions have no source-visible part and count as misses for every system. A lower S-src score is not worse retrieval in itself: genuine repeats are excluded.

| Contrast | Difference, points (95% interval) |
|---|---|
| L1: S2m Head A (M3AV train) - R5 VSI-inspired (96 ranked) | -1.5 (-4.9..+1.8) |
| L2: S2m Head A (M3AV train) - S1 best framework (also S1t's frames) | +5.0 (+2.4..+7.6) |

| System | S-src hit@4 (%) | primary hit@4 (%) |
|---|---|---|
| S0 starting design | 63.1 | 67.4 |
| S1 best framework (also S1t's frames) | 68.2 | 74.1 |
| S2c Head A (CG-Bench) | 73.4 | 79.7 |
| S2m Head A (M3AV train) | 73.3 | 79.4 |
| S3 Head B | 68.6 | 73.6 |
| R2 4 evenly spaced | 18.6 | 22.9 |
| R3 dense MobileCLIP top-4 (96 ranked) | 60.0 | 64.1 |
| R4 dense option-aware top-4 (96 ranked) | 72.2 | 76.9 |
| R5 VSI-inspired (96 ranked) | 74.8 | 79.8 |
