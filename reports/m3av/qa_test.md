# Answer accuracy (secondary), M3AV test: 853 questions, 112 lectures

Qwen3-VL-4B, K = 4, same prompt for every condition. secondary and descriptive: paired bootstrap over lectures (10,000, seed 20261101), raw p uncorrected; not part of the primary Holm family. The masks depend on Qwen3-VL-4B; they are not verified visually necessary subsets.

| Condition | accuracy, all (%) | without q_only-correct (%) | without t_only-correct (%) | abstained | forced | failed | frames | words (median) | prompt text tokens | visual tokens | answer s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 79.5 | 69.0 | 65.3 | 0 | 6 | 0 | 3.95 | 111.0 | 426.0 | 880.0 | 2.985 |
| S1 | 82.5 | 75.0 | 71.5 | 0 | 0 | 0 | 4.0 | 0.0 | 180.0 | 880.0 | 2.41 |
| S1t | 83.5 | 75.2 | 70.3 | 0 | 0 | 0 | 4.0 | 591.0 | 1507.0 | 880.0 | 4.264 |
| S2c | 88.2 | 82.2 | 80.8 | 0 | 1 | 0 | 4.0 | 0.0 | 180.0 | 880.0 | 2.365 |
| S2m | 87.2 | 81.2 | 80.8 | 0 | 0 | 0 | 4.0 | 0.0 | 180.0 | 880.0 | 2.363 |
| S3 | 82.8 | 74.3 | 70.6 | 0 | 0 | 0 | 4.0 | 0.0 | 180.0 | 880.0 | 2.464 |
| R2 | 63.5 | 45.7 | 45.8 | 0 | 3 | 0 | 4.0 | 0.0 | 178.0 | 880.0 | 2.646 |
| R3 | 78.3 | 67.5 | 68.1 | 0 | 2 | 0 | 3.87 | 0.0 | 179.0 | 880.0 | 2.476 |
| R4 | 84.8 | 77.2 | 76.2 | 0 | 3 | 0 | 3.88 | 0.0 | 179.0 | 880.0 | 2.414 |
| R5 | 86.5 | 79.6 | 74.9 | 0 | 1 | 0 | 3.94 | 0.0 | 180.0 | 880.0 | 2.353 |
| q_only | 51.2 | 0.0 | 18.3 | 0 | 0 | 0 | 0.0 | 0.0 | 128.0 | 0.0 | 1.469 |
| t_only | 62.1 | 36.5 | 0.0 | 0 | 1 | 0 | 0.0 | 590.0 | 1453.0 | 0.0 | 3.88 |

Masks: all 853; without q_only-correct 416; without t_only-correct 323. Controls: q_only {'correct': 437, 'failed (unknown)': 0, 'wrong': 416}; t_only {'correct': 530, 'failed (unknown)': 0, 'wrong': 323}

## Contrasts, all (853 questions)

| Contrast | points (95% interval) | raw p |
|---|---|---|
| S2m - R5 | +0.7 (-1.7..+3.0) | 0.5763 |
| S2m - S1 | +4.7 (+2.5..+6.9) | 0.0002 |
| S2m - S2c | -0.9 (-2.4..+0.5) | 0.2248 |
| S2m - R3 | +8.9 (+5.9..+11.9) | 0.0002 |
| S2m - R4 | +2.5 (+0.0..+5.0) | 0.05979 |
| S3 - S1 | +0.2 (-1.9..+2.4) | 0.8685 |
| S1t - S1 | +0.9 (-0.8..+2.8) | 0.3312 |

## Contrasts, without q_only-correct (416 questions)

| Contrast | points (95% interval) | raw p |
|---|---|---|
| S2m - R5 | +1.7 (-2.1..+5.4) | 0.3912 |
| S2m - S1 | +6.2 (+2.8..+9.7) | 0.0004 |
| S2m - S2c | -1.0 (-3.4..+1.4) | 0.4954 |
| S2m - R3 | +13.7 (+9.1..+18.1) | 0.0002 |
| S2m - R4 | +4.1 (-0.2..+8.2) | 0.06559 |
| S3 - S1 | -0.7 (-4.1..+2.7) | 0.7243 |
| S1t - S1 | +0.2 (-2.3..+2.8) | 0.9321 |

## Contrasts, without t_only-correct (323 questions)

| Contrast | points (95% interval) | raw p |
|---|---|---|
| S2m - R5 | +5.9 (+1.6..+10.2) | 0.0112 |
| S2m - S1 | +9.3 (+5.0..+13.6) | 0.0002 |
| S2m - S2c | +0.0 (-3.0..+2.8) | 1 |
| S2m - R3 | +12.7 (+7.6..+17.5) | 0.0002 |
| S2m - R4 | +4.6 (+0.3..+9.1) | 0.0484 |
| S3 - S1 | -0.9 (-5.6..+3.8) | 0.7621 |
| S1t - S1 | -1.2 (-4.1..+1.8) | 0.468 |
