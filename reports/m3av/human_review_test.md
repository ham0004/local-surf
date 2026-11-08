# Lecture study: blind human review of the 100 test packets

Analyses declared in the protocol (Question review, amendment 3) and fixed in research log steps 98-99 before any human verdict was seen. The cohort's labels are otherwise automatic. Sensitivity results are descriptive and never replace the primary result. Intervals cover sampling only, not annotator error or non-response.

Reviewers 3: H1 100 valid, 0 incomplete, 0 unresolved, 0 late; H2 34 valid, 0 incomplete, 0 unresolved, 0 late; H3 33 valid, 0 incomplete, 0 unresolved, 0 late.
Items reviewed 100 of 100 (votes per item: {1: 33, 2: 67}); no strict majority: 3.

| Combined human verdict | accept | reject | uncertain |
|---|---|---|---|
| items | 97 | 0 | 3 |

Rates: planned estimate (all 100 sampled items reviewed).

| Rate, percent (95% lecture-bootstrap interval) | lecture-balanced | cohort question-weighted |
|---|---|---|
| rejected | 0.0 (0.0..0.0) | 0.0 (0.0..0.0) |
| rejected or uncertain | 3.0 (0.0..7.0) | 2.7 (0.0..6.4) |

## Agreement

| Pair | items | agree | percent | Cohen's kappa |
|---|---|---|---|---|
| human vs builder | 100 | 82 | 82.0 | 0.019 |
| human vs auditor | 100 | 77 | 77.0 | 0.074 |
| builder vs auditor | 100 | 84 | 84.0 | 0.525 |
| H1 vs H2 | 34 | 31 | 91.2 | 0.0 |
| H1 vs H3 | 33 | 33 | 100.0 | None |

## Counts over the 167 valid human answers (reviewers can overlap)

- key: n 3, y 164
- one: n 2, y 165
- slide: n 1, y 166
- legible: y 167
- shortcut: knowledge 1, none 85, transcript 81
- reason: KEY 3, OK 164

## Hit@4 with the human-rejected sample questions removed (descriptive)

| Cohort | questions | S1 | S2m | R5 | L1 S2m − R5 (95% interval) | L2 S2m − S1 (95% interval) |
|---|---|---|---|---|---|---|
| full cohort (primary) | 853 | 74.1 | 79.4 | 79.8 | -0.5 (-3.8..+2.9) | +5.3 (+2.7..+7.8) |
| secondary 1: without human-rejected | 853 | 74.1 | 79.4 | 79.8 | -0.5 (-3.8..+2.9) | +5.3 (+2.7..+7.8) |
| conservative: without human-rejected or uncertain | 850 | 74.0 | 79.4 | 79.8 | -0.4 (-3.7..+3.0) | +5.4 (+2.8..+8.0) |

S-src hit@4:

| Cohort | S1 | S2m | R5 | L1 | L2 |
|---|---|---|---|---|---|
| full cohort (primary) | 68.2 | 73.3 | 74.8 | -1.5 (-4.9..+1.8) | +5.0 (+2.4..+7.6) |
| secondary 1: without human-rejected | 68.2 | 73.3 | 74.8 | -1.5 (-4.9..+1.8) | +5.0 (+2.4..+7.6) |
| conservative: without human-rejected or uncertain | 68.1 | 73.3 | 74.7 | -1.4 (-4.8..+1.9) | +5.2 (+2.5..+7.8) |

Answer-model accuracy (descriptive):

| Cohort | S1 | S2m | R5 | q_only | t_only |
|---|---|---|---|---|---|
| full cohort (primary) | 82.5 | 87.2 | 86.5 | 51.2 | 62.1 |
| secondary 1: without human-rejected | 82.5 | 87.2 | 86.5 | 51.2 | 62.1 |
| conservative: without human-rejected or uncertain | 82.6 | 87.3 | 86.5 | 51.4 | 62.4 |

Accepted sample questions alone (97 questions, 97 lectures; descriptive; the accepted sampled cases, not a cohort estimate): hit@4 S0 70.1, S1 77.3, S2c 80.4, S2m 80.4, S3 77.3, R2 24.7, R3 71.1, R4 78.4, R5 78.4.

All systems and conditions, dropped ids and reasons, and per-lecture attrition: human_review_test.json.
