# Answer accuracy (secondary), M3AV dev: 24 questions, 22 lectures

Qwen3-VL-4B, K = 4, same prompt for every condition. secondary and descriptive: paired bootstrap over lectures (10,000, seed 20261101), raw p uncorrected; not part of the primary Holm family. The masks depend on Qwen3-VL-4B; they are not verified visually necessary subsets.

| Condition | accuracy, all (%) | without q_only-correct (%) | without t_only-correct (%) | abstained | forced | failed | frames | words (median) | prompt text tokens | visual tokens | answer s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 87.5 | 81.8 | 77.8 | 0 | 0 | 0 | 4.0 | 107.0 | 427.5 | 880.0 | 2.5635 |
| S1 | 87.5 | 81.8 | 77.8 | 0 | 0 | 0 | 4.0 | 0.0 | 183.0 | 880.0 | 2.3015 |
| S1t | 87.5 | 81.8 | 77.8 | 0 | 0 | 0 | 4.0 | 590.0 | 1512.5 | 880.0 | 4.6 |
| S2c | 91.7 | 90.9 | 88.9 | 0 | 0 | 0 | 4.0 | 0.0 | 183.0 | 880.0 | 2.232 |
| S2m | 91.7 | 90.9 | 88.9 | 0 | 0 | 0 | 4.0 | 0.0 | 183.0 | 880.0 | 2.2195 |
| S3 | 79.2 | 63.6 | 55.6 | 0 | 0 | 0 | 4.0 | 0.0 | 184.0 | 880.0 | 2.2385 |
| R2 | 66.7 | 63.6 | 44.4 | 0 | 0 | 0 | 4.0 | 0.0 | 182.5 | 880.0 | 2.674 |
| R3 | 79.2 | 63.6 | 66.7 | 0 | 0 | 0 | 3.79 | 0.0 | 182.0 | 880.0 | 2.385 |
| R4 | 87.5 | 72.7 | 66.7 | 0 | 0 | 0 | 3.79 | 0.0 | 180.5 | 880.0 | 2.2455 |
| R5 | 95.8 | 90.9 | 88.9 | 0 | 0 | 0 | 3.88 | 0.0 | 183.5 | 880.0 | 2.2705 |
| q_only | 54.2 | 0.0 | 11.1 | 0 | 0 | 0 | 0.0 | 0.0 | 131.5 | 0.0 | 1.7375 |
| t_only | 62.5 | 27.3 | 0.0 | 0 | 0 | 0 | 0.0 | 592.0 | 1466.5 | 0.0 | 3.607 |

Masks: all 24; without q_only-correct 11; without t_only-correct 9. Controls: q_only {'correct': 13, 'failed (unknown)': 0, 'wrong': 11}; t_only {'correct': 15, 'failed (unknown)': 0, 'wrong': 9}

## Contrasts, all (24 questions)

| Contrast | points (95% interval) | raw p |
|---|---|---|
| S2m - R5 | -4.2 (-13.6..+0.0) | 0.6887 |
| S2m - S1 | +4.2 (-9.1..+18.2) | 0.7969 |
| S2m - S2c | +0.0 (-12.5..+12.5) | 1 |
| S2m - R3 | +12.5 (-4.5..+31.8) | 0.244 |
| S2m - R4 | +4.2 (-9.1..+18.2) | 0.7865 |
| S3 - S1 | -8.3 (-22.7..+0.0) | 0.2452 |
| S1t - S1 | +0.0 (+0.0..+0.0) | 1 |

## Contrasts, without q_only-correct (11 questions)

| Contrast | points (95% interval) | raw p |
|---|---|---|
| S2m - R5 | +0.0 (+0.0..+0.0) | 1 |
| S2m - S1 | +9.1 (+0.0..+27.3) | 0.6875 |
| S2m - S2c | +0.0 (+0.0..+0.0) | 1 |
| S2m - R3 | +27.3 (+0.0..+54.5) | 0.05979 |
| S2m - R4 | +18.2 (+0.0..+45.5) | 0.2102 |
| S3 - S1 | -18.2 (-45.5..+0.0) | 0.2196 |
| S1t - S1 | +0.0 (+0.0..+0.0) | 1 |

## Contrasts, without t_only-correct (9 questions)

| Contrast | points (95% interval) | raw p |
|---|---|---|
| S2m - R5 | +0.0 (+0.0..+0.0) | 1 |
| S2m - S1 | +11.1 (+0.0..+33.3) | 0.6857 |
| S2m - S2c | +0.0 (+0.0..+0.0) | 1 |
| S2m - R3 | +22.2 (+0.0..+55.6) | 0.2056 |
| S2m - R4 | +22.2 (+0.0..+55.6) | 0.2056 |
| S3 - S1 | -22.2 (-55.6..+0.0) | 0.193 |
| S1t - S1 | +0.0 (+0.0..+0.0) | 1 |
