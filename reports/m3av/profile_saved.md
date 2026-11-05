# Lecture study: time per stage from the frozen runs' own records

Saved files only; no model was run. Seconds. Pools: per question, the sum of the four instrumented components (retrieval, the speech/Head A scorer, frame decoding, MobileCLIP); it leaves out OCR and text embeddings, serialisation, the per-lecture feature index, R5's separate scoring pass, model loading, selection and transcript building. The 'Head A' column holds the ordinary speech scorer in systems without the learned head. Answers: per new answer call; reused replies are excluded, so unequal fresh-call subsets do not show that one selector's answers are faster.

## Pool building, dev

| System | pool | retrieval | Head A | decode | MobileCLIP | total (mean) | total (median) | total (p95) |
|---|---|---|---|---|---|---|---|---|
| S0 | v2 | 0.020 | 0.042 | 0.315 | 0.078 | 0.46 | 0.43 | 0.59 |
| S1 | hybrid | 0.021 | 0.044 | 0.507 | 0.092 | 0.66 | 0.64 | 0.84 |
| S1t | hybrid | 0.021 | 0.044 | 0.507 | 0.092 | 0.66 | 0.64 | 0.84 |
| S2c | hybrid_ahead | 0.022 | 0.340 | 0.512 | 0.088 | 0.96 | 0.74 | 1.76 |
| S2m | hybrid_ahead_m3av | 0.024 | 0.342 | 0.517 | 0.088 | 0.97 | 0.75 | 1.78 |
| S3 | hybrid | 0.021 | 0.044 | 0.507 | 0.092 | 0.66 | 0.64 | 0.84 |
| R2 | uniform4d | 0.021 | 0.044 | 0.058 | 0.043 | 0.17 | 0.12 | 0.26 |
| R3 | dense4x | 0.021 | 0.055 | 0.803 | 0.500 | 1.38 | 1.27 | 1.80 |
| R4 | dense4ox | 0.022 | 0.043 | 0.803 | 0.499 | 1.37 | 1.26 | 1.80 |
| R5 | vsi4x | 0.023 | 0.043 | 0.824 | 0.512 | 1.40 | 1.30 | 1.81 |

## Pool building, test

| System | pool | retrieval | Head A | decode | MobileCLIP | total (mean) | total (median) | total (p95) |
|---|---|---|---|---|---|---|---|---|
| S0 | v2 | 0.019 | 0.038 | 0.312 | 0.077 | 0.45 | 0.44 | 0.56 |
| S1 | hybrid | 0.024 | 0.039 | 0.511 | 0.095 | 0.67 | 0.66 | 0.84 |
| S1t | hybrid | 0.024 | 0.039 | 0.511 | 0.095 | 0.67 | 0.66 | 0.84 |
| S2c | hybrid_ahead | 0.026 | 0.343 | 0.539 | 0.091 | 1.00 | 0.78 | 1.83 |
| S2m | hybrid_ahead_m3av | 0.023 | 0.343 | 0.514 | 0.090 | 0.97 | 0.74 | 1.77 |
| S3 | hybrid | 0.024 | 0.039 | 0.511 | 0.095 | 0.67 | 0.66 | 0.84 |
| R2 | uniform4d | 0.025 | 0.040 | 0.058 | 0.043 | 0.17 | 0.12 | 0.28 |
| R3 | dense4x | 0.025 | 0.038 | 0.821 | 0.507 | 1.39 | 1.28 | 1.83 |
| R4 | dense4ox | 0.025 | 0.038 | 0.818 | 0.507 | 1.39 | 1.28 | 1.82 |
| R5 | vsi4x | 0.025 | 0.038 | 0.838 | 0.516 | 1.42 | 1.31 | 1.83 |

## Answer model (Qwen3-VL-4B), test

| Condition | new calls | reused | seconds (mean) | median | p95 | visual tokens | text tokens |
|---|---|---|---|---|---|---|---|
| S0 | 853 | 0 | 3.09 | 2.98 | 4.70 | 865 | 429 |
| S1 | 853 | 0 | 2.47 | 2.41 | 3.63 | 876 | 182 |
| S1t | 853 | 0 | 4.55 | 4.26 | 5.85 | 876 | 1522 |
| S2c | 851 | 2 | 2.41 | 2.37 | 3.54 | 877 | 182 |
| S2m | 636 | 217 | 2.41 | 2.33 | 3.58 | 877 | 182 |
| S3 | 838 | 15 | 2.55 | 2.46 | 3.68 | 877 | 183 |
| R2 | 853 | 0 | 2.72 | 2.65 | 4.08 | 877 | 181 |
| R3 | 853 | 0 | 2.52 | 2.48 | 3.77 | 848 | 181 |
| R4 | 744 | 109 | 2.46 | 2.40 | 3.69 | 857 | 182 |
| R5 | 853 | 0 | 2.44 | 2.35 | 3.57 | 864 | 182 |
| q_only | 853 | 0 | 1.48 | 1.47 | 2.18 | 0 | 131 |
| t_only | 853 | 0 | 3.97 | 3.88 | 5.37 | 0 | 1469 |

## Feature pass (MobileCLIP every 2 s, whole lecture)

- dev: 96 lectures, 626 frames and 20.9 min each; 9.4 s per lecture after the first (first includes loading), 900 s in all.
- test: 112 lectures, 579 frames and 19.3 min each; 8.4 s per lecture after the first (first includes loading), 955 s in all.

## Test answer session

- model load 8.5 s; torch peak allocated 11.49 GiB, peak reserved 16.8 GiB, card 15.93 GiB (NVIDIA GeForce RTX 5060 Ti).
