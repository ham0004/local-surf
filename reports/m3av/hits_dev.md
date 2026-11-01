# Source-slide interval hit@4, M3AV dev (514 questions, 96 lectures)

Paired bootstrap over lectures (10,000 resamples, seed 20261101); every frozen system covers every question. Dev results were used to choose methods; they are not a confirmation.

## Primary contrasts (amendment 4; Holm over L1 and L2)

| Contrast | Difference, points (95% interval) | raw p | Holm p | significant at 0.05 |
|---|---|---|---|---|
| L1: S2m Head A (M3AV train) - R5 VSI-inspired (96 ranked) | +1.0 (-2.9..+4.7) | 0.6399 | 0.6399 | no |
| L2: S2m Head A (M3AV train) - S1 best framework (also S1t's frames) | +6.8 (+3.1..+10.7) | 0.0005999 | 0.0012 | yes |

## Secondary contrasts (no correction; descriptive)

| Contrast | Difference, points (95% interval) | raw p |
|---|---|---|
| S2m Head A (M3AV train) - R3 dense MobileCLIP top-4 (96 ranked) | +14.4 (+9.4..+19.1) | 0.0002 |
| S2m Head A (M3AV train) - R4 dense option-aware top-4 (96 ranked) | +1.9 (-2.9..+6.5) | 0.45 |
| S2m Head A (M3AV train) - S2c Head A (CG-Bench) | -0.2 (-2.5..+2.2) | 0.9313 |

## Systems

| System | hit@4 (%) | frames shown (mean) | questions with < 4 frames |
|---|---|---|---|
| S0 starting design | 64.2 | 3.94 | 8 |
| S1 best framework (also S1t's frames) | 70.2 | 4.0 | 1 |
| S2c Head A (CG-Bench) | 77.2 | 4.0 | 0 |
| S2m Head A (M3AV train) | 77.0 | 4.0 | 0 |
| S3 Head B | 70.8 | 4.0 | 1 |
| R2 4 evenly spaced | 26.5 | 4.0 | 0 |
| R3 dense MobileCLIP top-4 (96 ranked) | 62.6 | 3.91 | 30 |
| R4 dense option-aware top-4 (96 ranked) | 75.1 | 3.91 | 31 |
| R5 VSI-inspired (96 ranked) | 76.1 | 3.96 | 13 |
| S2c, picker A_rank | 74.5 | 4.0 | 0 |
| S2m, picker A_rank | 76.7 | 4.0 | 0 |
| R3 dense MobileCLIP top-4, h0 | 60.5 | 3.43 | 154 |
| R3 dense MobileCLIP top-4, h3 | 58.9 | 3.02 | 240 |
| R4 dense option-aware top-4, h0 | 73.2 | 3.42 | 150 |
| R4 dense option-aware top-4, h3 | 71.0 | 3.02 | 239 |
| R5 VSI-inspired, h0 | 76.1 | 3.82 | 49 |
| R5 VSI-inspired, h3 | 72.8 | 3.52 | 125 |
| R3 (4-candidate pool, merged) | 46.7 | 1.4 | 510 |
