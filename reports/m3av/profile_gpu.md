# Lecture study: dev resource profile (frozen code, measured again)

NVIDIA GeForce RTX 5060 Ti, 15.93 GiB, 14.79 GiB free at start (the rest is the desktop and other processes). First 100 dev questions in hash order; R5 scoring over all 514 dev questions. Dedicated = process memory on the card; shared = process GPU memory held in system RAM (Windows counters, 1 s samples). Answer stages: torch peaks are the maximum over their calls.

| Stage | seconds | torch peak allocated (GiB) | torch peak reserved (GiB) | dedicated peak (GiB) | shared peak (GiB) |
|---|---|---|---|---|---|
| features (68 lectures) | 676.83 | 1.68 | 2.041 | 2.189 | 0.076 |
| R5 scoring (514 questions) | 93.07 | 3.479 | 4.932 | 5.08 | 0.076 |
| pools S1 (hybrid) | 77.16 | 1.038 | 2.18 | 2.328 | 0.076 |
| pools S2m (hybrid_ahead_m3av) | 113.66 | 1.849 | 3.602 | 3.75 | 0.076 |
| pools R5 (vsi4x) | 160.84 | 1.767 | 1.865 | 2.015 | 0.076 |
| answer model load | 9.04 | 8.364 | 8.381 | 8.511 | 0.076 |
| answers, frozen (600 calls) | 1619.53 | 11.459 | 17.256 | 15.231 | 1.557 |
| answers, capped (600 calls) | 1517.62 | 11.459 | 14.189 | 14.129 | 0.076 |

Answer weights resident after load: 8.364 GiB. Allocator cap in the capped pass: 14.28 GiB.

## Per answer call

| Pass | Condition | seconds (mean) | p95 | peak allocated (max, GiB) | peak reserved (max, GiB) | text tokens |
|---|---|---|---|---|---|---|
| frozen | S1 | 2.20 | 3.12 | 9.02 | 16.43 | 185 |
| frozen | S2m | 2.14 | 3.10 | 9.02 | 16.43 | 185 |
| frozen | R5 | 2.10 | 3.14 | 9.00 | 16.43 | 185 |
| frozen | S1t | 4.71 | 7.17 | 11.46 | 17.26 | 1533 |
| frozen | q_only | 1.63 | 2.41 | 8.45 | 16.43 | 134 |
| frozen | t_only | 3.42 | 4.25 | 9.76 | 16.43 | 1480 |
| capped | S1 | 2.10 | 2.94 | 9.02 | 13.98 | 185 |
| capped | S2m | 2.07 | 2.86 | 9.02 | 13.98 | 185 |
| capped | R5 | 2.03 | 3.00 | 9.00 | 13.98 | 185 |
| capped | S1t | 4.09 | 5.53 | 11.46 | 14.19 | 1533 |
| capped | q_only | 1.59 | 2.30 | 8.45 | 13.98 | 134 |
| capped | t_only | 3.29 | 4.04 | 9.76 | 13.98 | 1480 |

## Checks

- features_max_abs_diff: 0.0
- vsi_times_identical: 1.0
- pool_times_identical_S1: 1.0
- pool_times_identical_S2m: 1.0
- pool_times_identical_R5: 1.0
- capped_replies_identical: 1.0
- capped_out_of_memory: 0
- frozen_out_of_memory: 0
- replies_identical_to_dev_check_records: {'n': 144, 'identical': 144}
