| policy | condition | n | quality | frames | visual tok | decoded | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| transcript_only | clean | 7 | 0.286 | 0.00 | 0 | 0 | 1651 | 6319 |
| transcript_only | targeted_damage | 7 | 0.286 | 0.00 | 0 | 0 | 567 | 1035 |
| transcript_only | control_damage | 7 | 0.286 | 0.00 | 0 | 0 | 567 | 1036 |
| transcript_only | asr_noise | 7 | 0.286 | 0.00 | 0 | 0 | 569 | 1039 |
| uniform | clean | 7 | 0.571 | 6.00 | 1320 | 423 | 2107 | 5989 |
| uniform | targeted_damage | 7 | 0.571 | 6.00 | 1320 | 423 | 1234 | 1727 |
| uniform | control_damage | 7 | 0.571 | 6.00 | 1320 | 423 | 1234 | 1723 |
| uniform | asr_noise | 7 | 0.571 | 6.00 | 1320 | 423 | 1234 | 1728 |
| retrieval | clean | 7 | 0.286 | 0.00 | 0 | 423 | 1693 | 5444 |
| retrieval | targeted_damage | 7 | 0.286 | 0.00 | 0 | 423 | 838 | 1257 |
| retrieval | control_damage | 7 | 0.286 | 0.00 | 0 | 423 | 837 | 1254 |
| retrieval | asr_noise | 7 | 0.286 | 0.00 | 0 | 423 | 837 | 1258 |
| scout_similarity | clean | 7 | 0.714 | 6.00 | 1320 | 423 | 2069 | 5447 |
| scout_similarity | targeted_damage | 7 | 0.714 | 6.00 | 1320 | 423 | 1213 | 1644 |
| scout_similarity | control_damage | 7 | 0.714 | 6.00 | 1320 | 423 | 1211 | 1637 |
| scout_similarity | asr_noise | 7 | 0.714 | 6.00 | 1320 | 423 | 1213 | 1648 |
| heuristic | clean | 7 | 0.857 | 4.86 | 1069 | 423 | 2061 | 5796 |
| heuristic | targeted_damage | 7 | 0.857 | 5.00 | 1100 | 423 | 1218 | 1652 |
| heuristic | control_damage | 7 | 0.857 | 5.00 | 1100 | 423 | 1216 | 1645 |
| heuristic | asr_noise | 7 | 0.857 | 4.86 | 1069 | 423 | 1203 | 1649 |
| learned[head_paired] | clean | 7 | 0.429 | 0.43 | 94 | 423 | 1723 | 5471 |
| learned[head_paired] | targeted_damage | 7 | 0.429 | 0.57 | 126 | 423 | 878 | 1284 |
| learned[head_paired] | control_damage | 7 | 0.429 | 0.43 | 94 | 423 | 866 | 1274 |
| learned[head_paired] | asr_noise | 7 | 0.429 | 0.43 | 94 | 423 | 867 | 1283 |
| learned[head_unpaired] | clean | 7 | 0.429 | 0.43 | 94 | 423 | 1723 | 5469 |
| learned[head_unpaired] | targeted_damage | 7 | 0.429 | 0.43 | 94 | 423 | 873 | 1285 |
| learned[head_unpaired] | control_damage | 7 | 0.429 | 0.43 | 94 | 423 | 867 | 1277 |
| learned[head_unpaired] | asr_noise | 7 | 0.429 | 0.43 | 94 | 423 | 868 | 1285 |

| policy | targeted_response | control_overspend | selectivity |
|---|---|---|---|
| transcript_only | +0.00 | +0.00 | +0.00 |
| uniform | +0.00 | +0.00 | +0.00 |
| retrieval | +0.00 | +0.00 | +0.00 |
| scout_similarity | +0.00 | +0.00 | +0.00 |
| heuristic | +0.14 | +0.14 | +0.00 |
| learned[head_paired] | +0.14 | +0.00 | +0.14 |
| learned[head_unpaired] | +0.00 | +0.00 | +0.00 |
