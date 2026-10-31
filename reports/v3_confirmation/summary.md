# Confirmation on fresh LongVideoBench questions

Primary analysis on the 894 eligible questions; failed answers count as wrong. Holm-adjusted p-values are final only when all four hypotheses are final. p = (resamples on the other side of zero + 1) / (10,000 + 1), two-sided, so the smallest reportable p is about 0.0002.

| Hypothesis | Status | Difference (95% interval) | p | p (Holm) | Questions |
|---|---|---|---|---|---|
| H1: S4 - S1, 4B | final | -0.2 (-3.0..+2.7) | 0.9185 | 0.9185 | n 894, videos 500, failed [0, 0] |
| H2: S4 - S1, Nemotron-120B | final | +2.1 (-0.8..+5.1) | 0.1712 | 0.3424 | n 894, videos 500, failed [0, 0] |
| H3: S1t - S0, 4B | final | +8.9 (+5.7..+12.2) | 0.0002 | 0.0008 | n 894, videos 500, failed [1, 0]; without failures +9.0 (+5.7..+12.2), n 893 |
| H4: S1t - S0, Nemotron-120B | final | +8.3 (+4.8..+11.8) | 0.0002 | 0.0008 | n 894, videos 500, failed [0, 0] |

## Secondary (no correction)

| Comparison | Status | Difference (95% interval) | p | | Questions |
|---|---|---|---|---|---|
| S2 - S1 2B | final | +1.8 (-0.6..+4.1) | 0.1376 |  | n 894, videos 500, failed [0, 0] |
| S3 - S1 2B | final | +0.1 (-2.0..+2.2) | 0.9729 |  | n 894, videos 500, failed [0, 0] |
| S4t - S1t 2B | final | -0.6 (-3.2..+2.1) | 0.7199 |  | n 894, videos 500, failed [0, 0] |
| S1 - R3 2B | final | -0.8 (-3.3..+1.8) | 0.5611 |  | n 894, videos 500, failed [0, 0] |
| S1 - R2 2B | final | +7.5 (+4.3..+10.8) | 0.0002 |  | n 894, videos 500, failed [0, 0] |
| S2 - S1 4B | final | +1.2 (-1.5..+3.9) | 0.3914 |  | n 894, videos 500, failed [0, 0] |
| S3 - S1 4B | final | -0.3 (-2.9..+2.2) | 0.8371 |  | n 894, videos 500, failed [0, 0] |
| S4t - S1t 4B | final | -1.6 (-4.5..+1.5) | 0.3258 |  | n 894, videos 500, failed [1, 1]; without failures -1.6 (-4.5..+1.5), n 893 |
| S1 - R3 4B | final | -0.1 (-2.9..+2.6) | 0.9553 |  | n 894, videos 500, failed [0, 0] |
| S1 - R2 4B | final | +8.6 (+5.1..+12.2) | 0.0002 |  | n 894, videos 500, failed [0, 0] |
| S2 - S1 Nemotron-120B | final | +2.8 (+0.0..+5.6) | 0.0518 |  | n 894, videos 500, failed [0, 0] |
| S3 - S1 Nemotron-120B | final | +0.8 (-1.8..+3.4) | 0.5715 |  | n 894, videos 500, failed [0, 0] |
| S4t - S1t Nemotron-120B | final | +2.5 (-0.7..+5.6) | 0.1358 |  | n 894, videos 500, failed [0, 0] |
| S1 - R3 Nemotron-120B | final | -1.1 (-4.1..+1.8) | 0.4824 |  | n 894, videos 500, failed [0, 0] |
| S1 - R2 Nemotron-120B | final | +5.4 (+2.0..+8.8) | 0.002 |  | n 894, videos 500, failed [0, 0] |

## Accuracy

| System | Answered | Correct | Accuracy | Failed |
|---|---|---|---|---|
| S0 2B | 894 | 397 | 44.4 | 0 |
| S1 2B | 894 | 454 | 50.8 | 0 |
| S1t 2B | 894 | 455 | 50.9 | 0 |
| S2 2B | 894 | 470 | 52.6 | 0 |
| S3 2B | 894 | 455 | 50.9 | 0 |
| S4 2B | 894 | 454 | 50.8 | 0 |
| S4t 2B | 894 | 450 | 50.3 | 0 |
| R2 2B | 894 | 387 | 43.3 | 0 |
| R3 2B | 894 | 461 | 51.6 | 0 |
| S0 4B | 894 | 432 | 48.3 | 0 |
| S1 4B | 894 | 500 | 55.9 | 0 |
| S1t 4B | 894 | 512 | 57.3 | 1 |
| S2 4B | 894 | 511 | 57.2 | 0 |
| S3 4B | 894 | 497 | 55.6 | 0 |
| S4 4B | 894 | 498 | 55.7 | 0 |
| S4t 4B | 894 | 498 | 55.7 | 1 |
| R2 4B | 894 | 423 | 47.3 | 0 |
| R3 4B | 894 | 501 | 56.0 | 0 |
| S0 Nemotron-120B | 894 | 441 | 49.3 | 0 |
| S1 Nemotron-120B | 894 | 505 | 56.5 | 0 |
| S1t Nemotron-120B | 894 | 515 | 57.6 | 0 |
| S2 Nemotron-120B | 894 | 530 | 59.3 | 0 |
| S3 Nemotron-120B | 894 | 512 | 57.3 | 0 |
| S4 Nemotron-120B | 894 | 524 | 58.6 | 0 |
| S4t Nemotron-120B | 894 | 537 | 60.1 | 0 |
| R2 Nemotron-120B | 894 | 457 | 51.1 | 0 |
| R3 Nemotron-120B | 894 | 515 | 57.6 | 0 |
