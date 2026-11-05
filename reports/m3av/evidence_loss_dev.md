# Where the source slide is lost, M3AV dev (514 questions, 96 lectures)

Automatic source-slide intervals (a visibility proxy with known false matches), not human-verified. Percent of questions with 95% lecture-bootstrap intervals (10,000 resamples, seed 20261101). Descriptive; no p-values. Dev: the basis for any later redesign.

| System | Candidates (mean) | Slide in pool | Lost at candidates | Lost at selection | Picked (hit@4) | Picked if in pool |
|---|---|---|---|---|---|---|
| S0 | 8.6 | 71.8 (67.6..75.9) | 28.2 (24.1..32.4) | 7.6 (5.4..9.9) | 64.2 (59.8..68.6) | 89.4 |
| S1 | 8.9 | 77.8 (73.6..81.9) | 22.2 (18.1..26.4) | 7.6 (5.4..10.0) | 70.2 (66.1..74.3) | 90.2 |
| S2c | 9.0 | 85.8 (82.5..89.0) | 14.2 (11.0..17.5) | 8.6 (6.0..11.4) | 77.2 (73.3..81.0) | 90.0 |
| S2m | 9.0 | 85.8 (82.5..89.0) | 14.2 (11.0..17.5) | 8.8 (6.1..11.6) | 77.0 (73.0..80.9) | 89.8 |
| S3 | 8.9 | 77.8 (73.6..81.9) | 22.2 (18.1..26.4) | 7.0 (4.9..9.3) | 70.8 (66.5..75.1) | 91.0 |
| R2 | 4.0 (saved: the chosen frames only) | - | - | - | 26.5 (22.4..30.7) | - |
| R3 | 3.9 (saved: the chosen frames only) | - | - | - | 62.6 (58.1..67.5) | - |
| R4 | 3.9 (saved: the chosen frames only) | - | - | - | 75.1 (70.7..79.6) | - |
| R5 | 4.0 (saved: the chosen frames only) | - | - | - | 76.1 (72.1..79.8) | - |

R2-R5 save only the frames they show (R3-R5 rank 96 dense frames first), so their loss is not split by stage; their misses are all 'not picked'.
