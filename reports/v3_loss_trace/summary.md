# Loss trace on CG-Bench dev (82 questions)

Each question is put in one class: right answers by whether a chosen frame is inside the marked seconds, wrong answers by the first pipeline step whose output misses them. The classes are descriptive, not causal: they do not say which change would recover the answer, "model limit" means the 5 tested placements failed, and text systems are compared with an oracle whose text differs. Pre-filter and window losses are not separated.

| System | n | pool miss | choice miss | reading miss | model limit | right, hit | right, miss |
|---|---|---|---|---|---|---|---|
| S0 2B | 82 | 41 (50%) | 6 (7%) | 6 (7%) | 6 (7%) | 8 (10%) | 15 (18%) |
| S1 2B | 82 | 24 (29%) | 3 (4%) | 10 (12%) | 12 (15%) | 19 (23%) | 14 (17%) |
| S1t 2B | 82 | 25 (30%) | 4 (5%) | 13 (16%) | 9 (11%) | 19 (23%) | 12 (15%) |
| S3 2B | 82 | 25 (30%) | 2 (2%) | 11 (13%) | 12 (15%) | 17 (21%) | 15 (18%) |
| S4 2B | 82 | 22 (27%) | 7 (9%) | 9 (11%) | 11 (13%) | 19 (23%) | 14 (17%) |
| S4t 2B | 82 | 24 (29%) | 7 (9%) | 13 (16%) | 8 (10%) | 18 (22%) | 12 (15%) |
| S0 4B | 82 | 39 (48%) | 4 (5%) | 4 (5%) | 5 (6%) | 11 (13%) | 19 (23%) |
| S1 4B | 82 | 23 (28%) | 8 (10%) | 13 (16%) | 8 (10%) | 20 (24%) | 10 (12%) |
| S1t 4B | 82 | 21 (26%) | 8 (10%) | 12 (15%) | 8 (10%) | 21 (26%) | 12 (15%) |
| S3 4B | 82 | 25 (30%) | 3 (4%) | 12 (15%) | 9 (11%) | 19 (23%) | 14 (17%) |
| S4 4B | 82 | 20 (24%) | 5 (6%) | 11 (13%) | 7 (9%) | 21 (26%) | 18 (22%) |
| S4t 4B | 82 | 19 (23%) | 5 (6%) | 9 (11%) | 8 (10%) | 22 (27%) | 19 (23%) |

## Unconditional stage coverage (all questions)

| System | pool inside | Path A inside | Path B inside | chosen inside | accuracy | accuracy, chosen inside | accuracy, chosen outside |
|---|---|---|---|---|---|---|---|
| S0 2B | 33% | 26% | 18% | 24% | 28% | 40% | 24% |
| S1 2B | 60% | 26% | 49% | 50% | 40% | 46% | 34% |
| S1t 2B | 60% | 26% | 49% | 50% | 38% | 46% | 29% |
| S3 2B | 60% | 26% | 49% | 49% | 39% | 42% | 36% |
| S4 2B | 65% | 33% | 49% | 48% | 40% | 49% | 33% |
| S4t 2B | 65% | 33% | 49% | 48% | 37% | 46% | 28% |
| S0 4B | 33% | 26% | 18% | 24% | 37% | 55% | 31% |
| S1 4B | 60% | 26% | 49% | 50% | 37% | 49% | 24% |
| S1t 4B | 60% | 26% | 49% | 50% | 40% | 51% | 29% |
| S3 4B | 60% | 26% | 49% | 49% | 40% | 48% | 33% |
| S4 4B | 65% | 33% | 49% | 48% | 48% | 54% | 42% |
| S4t 4B | 65% | 33% | 49% | 48% | 50% | 56% | 44% |
