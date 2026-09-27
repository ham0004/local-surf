| policy | condition | n | quality | frames | visual tok | decoded | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| transcript_only | clean | 43 | 0.349 | 0.00 | 0 | 0 | 1302 | 2317 |
| transcript_only | targeted_damage | 43 | 0.302 | 0.00 | 0 | 0 | 937 | 2289 |
| transcript_only | control_damage | 43 | 0.326 | 0.00 | 0 | 0 | 1031 | 2283 |
| transcript_only | asr_noise | 43 | 0.302 | 0.00 | 0 | 0 | 1055 | 2303 |
| uniform | clean | 43 | 0.349 | 5.98 | 1315 | 1198 | 2027 | 3573 |
| uniform | targeted_damage | 43 | 0.442 | 6.00 | 1320 | 1222 | 1688 | 2969 |
| uniform | control_damage | 43 | 0.349 | 5.98 | 1315 | 1224 | 1849 | 3236 |
| uniform | asr_noise | 43 | 0.395 | 5.98 | 1315 | 1258 | 1787 | 3127 |
| retrieval | clean | 43 | 0.419 | 4.58 | 1008 | 1198 | 1841 | 3481 |
| retrieval | targeted_damage | 43 | 0.488 | 4.30 | 947 | 1222 | 1530 | 2697 |
| retrieval | control_damage | 43 | 0.465 | 4.53 | 998 | 1224 | 1723 | 2920 |
| retrieval | asr_noise | 43 | 0.442 | 4.60 | 1013 | 1258 | 1625 | 2711 |
| scout_similarity | clean | 43 | 0.488 | 5.98 | 1315 | 1198 | 1928 | 2997 |
| scout_similarity | targeted_damage | 43 | 0.465 | 6.00 | 1320 | 1222 | 1645 | 2694 |
| scout_similarity | control_damage | 43 | 0.442 | 5.98 | 1315 | 1224 | 1819 | 2982 |
| scout_similarity | asr_noise | 43 | 0.512 | 5.98 | 1315 | 1258 | 1716 | 2711 |
| heuristic | clean | 43 | 0.442 | 5.63 | 1238 | 1198 | 2005 | 3775 |
| heuristic | targeted_damage | 43 | 0.488 | 5.63 | 1238 | 1222 | 1635 | 2059 |
| heuristic | control_damage | 43 | 0.465 | 5.63 | 1238 | 1224 | 1801 | 3554 |
| heuristic | asr_noise | 43 | 0.442 | 5.60 | 1233 | 1258 | 1717 | 2670 |
| learned[head_paired] | clean | 43 | 0.395 | 0.26 | 56 | 1198 | 1866 | 3193 |
| learned[head_paired] | targeted_damage | 43 | 0.372 | 0.26 | 56 | 1222 | 1629 | 3128 |
| learned[head_paired] | control_damage | 43 | 0.372 | 0.23 | 51 | 1224 | 1617 | 3121 |
| learned[head_paired] | asr_noise | 43 | 0.326 | 0.23 | 51 | 1258 | 1667 | 3187 |
| learned[head_unpaired] | clean | 43 | 0.372 | 0.81 | 179 | 1198 | 1931 | 3173 |
| learned[head_unpaired] | targeted_damage | 43 | 0.419 | 0.95 | 210 | 1222 | 1551 | 2947 |
| learned[head_unpaired] | control_damage | 43 | 0.372 | 0.84 | 184 | 1224 | 1733 | 3171 |
| learned[head_unpaired] | asr_noise | 43 | 0.349 | 0.81 | 179 | 1258 | 1644 | 3176 |
| learned[head_paired]@0 | clean | 43 | 0.395 | 2.88 | 634 | 1198 | 1854 | 3109 |
| learned[head_paired]@0 | targeted_damage | 43 | 0.395 | 2.79 | 614 | 1222 | 1691 | 3296 |
| learned[head_paired]@0 | control_damage | 43 | 0.419 | 2.74 | 604 | 1224 | 1742 | 3119 |
| learned[head_paired]@0 | asr_noise | 43 | 0.442 | 2.51 | 553 | 1258 | 1718 | 3106 |
| learned[head_unpaired]@0 | clean | 43 | 0.419 | 1.79 | 394 | 1198 | 1855 | 3148 |
| learned[head_unpaired]@0 | targeted_damage | 43 | 0.419 | 1.74 | 384 | 1222 | 1585 | 2544 |
| learned[head_unpaired]@0 | control_damage | 43 | 0.419 | 1.77 | 389 | 1224 | 1680 | 3162 |
| learned[head_unpaired]@0 | asr_noise | 43 | 0.419 | 2.02 | 445 | 1258 | 1694 | 3057 |
| learned[head_paired]@0.1 | clean | 43 | 0.442 | 1.44 | 317 | 1198 | 1908 | 3161 |
| learned[head_paired]@0.1 | targeted_damage | 43 | 0.372 | 1.49 | 327 | 1222 | 1622 | 3041 |
| learned[head_paired]@0.1 | control_damage | 43 | 0.419 | 1.44 | 317 | 1224 | 1709 | 3126 |
| learned[head_paired]@0.1 | asr_noise | 43 | 0.419 | 1.28 | 281 | 1258 | 1711 | 3198 |
| learned[head_unpaired]@0.1 | clean | 43 | 0.395 | 0.77 | 169 | 1198 | 1945 | 3161 |
| learned[head_unpaired]@0.1 | targeted_damage | 43 | 0.419 | 0.77 | 169 | 1222 | 1525 | 2902 |
| learned[head_unpaired]@0.1 | control_damage | 43 | 0.349 | 0.79 | 174 | 1224 | 1726 | 3174 |
| learned[head_unpaired]@0.1 | asr_noise | 43 | 0.349 | 0.79 | 174 | 1258 | 1646 | 3179 |
| learned[head_paired]@0.3 | clean | 43 | 0.349 | 0.09 | 20 | 1198 | 1853 | 3188 |
| learned[head_paired]@0.3 | targeted_damage | 43 | 0.326 | 0.07 | 15 | 1222 | 1563 | 3052 |
| learned[head_paired]@0.3 | control_damage | 43 | 0.326 | 0.09 | 20 | 1224 | 1617 | 3105 |
| learned[head_paired]@0.3 | asr_noise | 43 | 0.302 | 0.09 | 20 | 1258 | 1692 | 3219 |
| learned[head_unpaired]@0.3 | clean | 43 | 0.349 | 0.07 | 15 | 1198 | 1850 | 3164 |
| learned[head_unpaired]@0.3 | targeted_damage | 43 | 0.326 | 0.07 | 15 | 1222 | 1563 | 3044 |
| learned[head_unpaired]@0.3 | control_damage | 43 | 0.326 | 0.07 | 15 | 1224 | 1616 | 3119 |
| learned[head_unpaired]@0.3 | asr_noise | 43 | 0.302 | 0.09 | 20 | 1258 | 1668 | 3217 |
| learned[head_paired]@0.6 | clean | 43 | 0.349 | 0.00 | 0 | 1198 | 1876 | 3142 |
| learned[head_paired]@0.6 | targeted_damage | 43 | 0.302 | 0.00 | 0 | 1222 | 1555 | 3027 |
| learned[head_paired]@0.6 | control_damage | 43 | 0.326 | 0.00 | 0 | 1224 | 1645 | 3112 |
| learned[head_paired]@0.6 | asr_noise | 43 | 0.302 | 0.00 | 0 | 1258 | 1680 | 3211 |
| learned[head_unpaired]@0.6 | clean | 43 | 0.349 | 0.00 | 0 | 1198 | 1881 | 3163 |
| learned[head_unpaired]@0.6 | targeted_damage | 43 | 0.302 | 0.00 | 0 | 1222 | 1556 | 3057 |
| learned[head_unpaired]@0.6 | control_damage | 43 | 0.326 | 0.00 | 0 | 1224 | 1650 | 3096 |
| learned[head_unpaired]@0.6 | asr_noise | 43 | 0.302 | 0.00 | 0 | 1258 | 1680 | 3222 |

| policy | targeted_response | control_overspend | selectivity |
|---|---|---|---|
| transcript_only | +0.00 | +0.00 | +0.00 |
| uniform | +0.02 | +0.00 | +0.02 |
| retrieval | -0.28 | -0.05 | -0.23 |
| scout_similarity | +0.02 | +0.00 | +0.02 |
| heuristic | +0.00 | +0.00 | +0.00 |
| learned[head_paired] | +0.00 | -0.02 | +0.02 |
| learned[head_unpaired] | +0.14 | +0.02 | +0.12 |
| learned[head_paired]@0 | -0.09 | -0.14 | +0.05 |
| learned[head_unpaired]@0 | -0.05 | -0.02 | -0.02 |
| learned[head_paired]@0.1 | +0.05 | +0.00 | +0.05 |
| learned[head_unpaired]@0.1 | +0.00 | +0.02 | -0.02 |
| learned[head_paired]@0.3 | -0.02 | +0.00 | -0.02 |
| learned[head_unpaired]@0.3 | +0.00 | +0.00 | +0.00 |
| learned[head_paired]@0.6 | +0.00 | +0.00 | +0.00 |
| learned[head_unpaired]@0.6 | +0.00 | +0.00 | +0.00 |
