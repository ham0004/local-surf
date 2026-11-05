# Where the source slide is lost, M3AV test (853 questions, 112 lectures)

Automatic source-slide intervals (a visibility proxy with known false matches), not human-verified. Percent of questions with 95% lecture-bootstrap intervals (10,000 resamples, seed 20261101). Descriptive; no p-values. Test is revealed: post hoc, never used to choose a method.

| System | Candidates (mean) | Slide in pool | Lost at candidates | Lost at selection | Picked (hit@4) | Picked if in pool |
|---|---|---|---|---|---|---|
| S0 | 8.6 | 75.3 (71.7..78.8) | 24.7 (21.2..28.3) | 7.9 (5.9..9.8) | 67.4 (63.7..71.0) | 89.6 |
| S1 | 9.0 | 81.2 (78.0..84.4) | 18.8 (15.6..22.0) | 7.2 (5.3..9.0) | 74.1 (70.6..77.5) | 91.2 |
| S2c | 9.1 | 88.9 (86.4..91.3) | 11.1 (8.7..13.6) | 9.1 (6.9..11.6) | 79.7 (76.5..82.9) | 89.7 |
| S2m | 9.0 | 89.0 (86.7..91.2) | 11.0 (8.8..13.3) | 9.6 (7.5..11.8) | 79.4 (76.6..82.3) | 89.2 |
| S3 | 9.0 | 81.2 (78.0..84.4) | 18.8 (15.6..22.0) | 7.6 (5.7..9.6) | 73.6 (70.3..77.0) | 90.6 |
| R2 | 4.0 (saved: the chosen frames only) | - | - | - | 22.9 (19.7..26.2) | - |
| R3 | 3.9 (saved: the chosen frames only) | - | - | - | 64.1 (60.2..68.1) | - |
| R4 | 3.9 (saved: the chosen frames only) | - | - | - | 76.9 (73.4..80.4) | - |
| R5 | 3.9 (saved: the chosen frames only) | - | - | - | 79.8 (76.3..83.3) | - |

R2-R5 save only the frames they show (R3-R5 rank 96 dense frames first), so their loss is not split by stage; their misses are all 'not picked'.

## Answer accuracy by stage (Qwen3-VL-4B, secondary)

Accuracy among the questions in each cell; in brackets, the same without questions the question-only control answers correctly.

| System | Picked: n, acc. | In pool, not picked: n, acc. | Not in pool: n, acc. |
|---|---|---|---|
| S0 | 575, 92.2% [269, 85.9%] | 67, 52.2% [40, 37.5%] | 211, 53.6% [107, 38.3%] |
| S1 | 632, 91.1% [306, 85.9%] | 61, 65.6% [34, 44.1%] | 160, 55.0% [76, 44.7%] |
| S2c | 680, 94.0% [334, 90.4%] | 78, 62.8% [33, 42.4%] | 95, 67.4% [49, 53.1%] |
| S2m | 677, 93.9% [328, 90.5%] | 82, 56.1% [39, 35.9%] | 94, 66.0% [49, 55.1%] |
| S3 | 628, 92.2% [303, 87.1%] | 65, 53.8% [37, 35.1%] | 160, 57.5% [76, 42.1%] |
| R2 | 195, 85.6% [93, 77.4%] | not picked (all stages): 658, 57.0% [323, 36.5%] | - |
| R3 | 547, 94.7% [256, 91.8%] | not picked (all stages): 306, 49.0% [160, 28.8%] | - |
| R4 | 656, 95.1% [313, 92.7%] | not picked (all stages): 197, 50.3% [103, 30.1%] | - |
| R5 | 681, 94.6% [328, 91.8%] | not picked (all stages): 172, 54.7% [88, 34.1%] | - |

S2m, all 853 questions: picked and correct 636, picked and wrong 41, not picked and correct 108, not picked and wrong 68.
