| policy | condition | n | quality | frames | visual tok | decoded | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| transcript_only | clean | 18 | 0.444 | 0.00 | 0 | 0 | 1419 | 2450 |
| transcript_only | targeted_damage | 18 | 0.333 | 0.00 | 0 | 0 | 693 | 991 |
| transcript_only | control_damage | 18 | 0.444 | 0.00 | 0 | 0 | 937 | 1357 |
| transcript_only | asr_noise | 18 | 0.389 | 0.00 | 0 | 0 | 940 | 1422 |
| uniform | clean | 18 | 1.000 | 6.00 | 1320 | 672 | 1759 | 2664 |
| uniform | targeted_damage | 18 | 1.000 | 6.00 | 1320 | 500 | 1305 | 1886 |
| uniform | control_damage | 18 | 1.000 | 6.00 | 1320 | 679 | 1426 | 1736 |
| uniform | asr_noise | 18 | 1.000 | 6.00 | 1320 | 652 | 1416 | 1738 |
| retrieval | clean | 18 | 1.000 | 3.00 | 660 | 672 | 1483 | 2457 |
| retrieval | targeted_damage | 18 | 0.333 | 0.00 | 0 | 500 | 846 | 1149 |
| retrieval | control_damage | 18 | 0.778 | 3.00 | 660 | 679 | 1178 | 1517 |
| retrieval | asr_noise | 18 | 1.000 | 3.00 | 660 | 652 | 1157 | 1471 |
| scout_similarity | clean | 18 | 1.000 | 6.00 | 1320 | 672 | 1782 | 2832 |
| scout_similarity | targeted_damage | 18 | 1.000 | 6.00 | 1320 | 500 | 1286 | 1854 |
| scout_similarity | control_damage | 18 | 1.000 | 6.00 | 1320 | 679 | 1450 | 1945 |
| scout_similarity | asr_noise | 18 | 1.000 | 6.00 | 1320 | 652 | 1411 | 1727 |
| heuristic | clean | 18 | 1.000 | 4.67 | 1027 | 672 | 1660 | 2445 |
| heuristic | targeted_damage | 18 | 1.000 | 5.00 | 1100 | 500 | 1178 | 1610 |
| heuristic | control_damage | 18 | 1.000 | 4.44 | 978 | 679 | 1277 | 1445 |
| heuristic | asr_noise | 18 | 1.000 | 4.67 | 1027 | 652 | 1296 | 1506 |
| learned[head_paired] | clean | 18 | 0.944 | 0.67 | 147 | 672 | 1391 | 2452 |
| learned[head_paired] | targeted_damage | 18 | 0.722 | 0.72 | 159 | 500 | 792 | 1224 |
| learned[head_paired] | control_damage | 18 | 0.778 | 0.67 | 147 | 679 | 1038 | 1563 |
| learned[head_paired] | asr_noise | 18 | 0.778 | 0.56 | 122 | 652 | 1073 | 1464 |
| learned[head_unpaired] | clean | 18 | 1.000 | 4.00 | 880 | 672 | 1592 | 2807 |
| learned[head_unpaired] | targeted_damage | 18 | 1.000 | 5.00 | 1100 | 500 | 1168 | 1631 |
| learned[head_unpaired] | control_damage | 18 | 0.944 | 3.56 | 782 | 679 | 1214 | 1628 |
| learned[head_unpaired] | asr_noise | 18 | 1.000 | 4.33 | 953 | 652 | 1292 | 1719 |

| policy | targeted_response | control_overspend | selectivity |
|---|---|---|---|
| transcript_only | +0.00 | +0.00 | +0.00 |
| uniform | +0.00 | +0.00 | +0.00 |
| retrieval | -3.00 | +0.00 | -3.00 |
| scout_similarity | +0.00 | +0.00 | +0.00 |
| heuristic | +0.33 | -0.22 | +0.56 |
| learned[head_paired] | +0.06 | +0.00 | +0.06 |
| learned[head_unpaired] | +1.00 | -0.44 | +1.44 |
