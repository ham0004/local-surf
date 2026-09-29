# Benchmark results: LongVideoBench (v1 framework)

440 multiple-choice questions over 252 videos from the LongVideoBench validation set, each answered on its original subtitles by a **2B-parameter** vision-language model (Qwen3-VL-2B-Instruct) running locally on one RTX 5060 Ti (16 GB), with **at most 8 frames** per question. Chance level: 21.4%.

## How answers are scored

Each question has 4–5 options and one correct option marked by the dataset's annotators. The model answers in free text with the option letter first; the letter is extracted and compared with the correct one (1 = match, 0 = otherwise), the leaderboard's own exact-match protocol. An "insufficient evidence" reply counts as wrong. Parsing check: 2200 of 2200 answers begin with a clean option letter.

## Frame-selection strategies (same model, same questions)

| strategy | accuracy | 95% CI (by video) | frames shown to the model | frames decoded | time per question (warm, median) |
|---|---|---|---|---|---|
| Transcript only (no frames) | **32.5%** | 28.3–36.7% | 0.0 | 0 | 1.14 s |
| Uniform frames + transcript | **39.3%** | 34.7–44.0% | 6.0 | 428 | 1.26 s |
| Transcript-retrieved moments | **36.4%** | 31.7–40.8% | 4.2 | 289 | 1.11 s |
| MobileCLIP top-k frames | **41.6%** | 36.7–46.4% | 6.0 | 1267 | 2.25 s |
| Cost-aware heuristic controller | **37.5%** | 32.9–42.0% | 5.7 | 521 | 1.47 s |

## Accuracy by video length

| strategy | 8–15 s | 15–60 s | 3–10 min | 15–60 min |
|---|---|---|---|---|
| Transcript only (no frames) | 38.9% (n=72) | 45.5% (n=55) | 29.8% (n=124) | 26.9% (n=186) |
| Uniform frames + transcript | 54.2% (n=72) | 40.0% (n=55) | 37.9% (n=124) | 34.4% (n=186) |
| Transcript-retrieved moments | 50.0% (n=72) | 41.8% (n=55) | 33.9% (n=124) | 31.2% (n=186) |
| MobileCLIP top-k frames | 51.4% (n=72) | 43.6% (n=55) | 40.3% (n=124) | 37.6% (n=186) |
| Cost-aware heuristic controller | 51.4% (n=72) | 43.6% (n=55) | 37.1% (n=124) | 30.6% (n=186) |

## Paired comparisons (same 440 questions)

Difference in accuracy between two strategies on the *same* questions, with a 95% bootstrap interval resampling videos. ✔ = the interval excludes zero.

| comparison | difference | 95% CI | |
|---|---|---|---|
| MobileCLIP top-k frames vs Transcript only (no frames) | +9.1 pts | +4.6 to +13.6 | ✔ |
| MobileCLIP top-k frames vs Uniform frames + transcript | +2.3 pts | -1.2 to +5.9 | ✗ not significant |
| MobileCLIP top-k frames vs Transcript-retrieved moments | +5.2 pts | +1.6 to +8.9 | ✔ |
| MobileCLIP top-k frames vs Cost-aware heuristic controller | +4.1 pts | +0.7 to +7.4 | ✔ |
| Uniform frames + transcript vs Transcript only (no frames) | +6.8 pts | +2.3 to +11.3 | ✔ |
| Transcript-retrieved moments vs Transcript only (no frames) | +3.9 pts | -0.5 to +8.2 | ✗ not significant |
| Uniform frames + transcript vs Transcript-retrieved moments | +3.0 pts | -0.9 to +7.0 | ✗ not significant |

## Accuracy vs time

Measured per-action costs on this GPU (`configs/cost_model_rtx5060ti.yaml`): decoding ≈ 43 ms per frame read, MobileCLIP scoring ≈ 47 ms per frame, the answer model reading images ≈ 0.36 ms per image token (≈ 80 ms per 640-px frame). Video can only be decoded forward from a keyframe, so reaching one frame often means reading many (the *frames decoded* column).

| strategy | accuracy | time per question | vs transcript only |
|---|---|---|---|
| Transcript-retrieved moments | 36.4% | 1.11 s | +3.9 pts for -0.03 s |
| Transcript only (no frames) | 32.5% | 1.14 s | baseline |
| Uniform frames + transcript | 39.3% | 1.26 s | +6.8 pts for +0.12 s |
| Cost-aware heuristic controller | 37.5% | 1.47 s | +5.0 pts for +0.33 s |
| MobileCLIP top-k frames | 41.6% | 2.25 s | +9.1 pts for +1.11 s |

Uniform frames give most of the gain over text for almost no time (+6.8 pts for +0.12 s). MobileCLIP top-k is the most accurate, but it decodes and scores every candidate (~1267 frames read), which costs about 1.0 s more than uniform for a +2.3-point difference that is not significant. Uniform is the speed choice and MobileCLIP the accuracy choice; the framework measures both so the choice is explicit.

## Findings

1. **Looking at frames helps.** Uniform frames and MobileCLIP top-k both beat transcript-only (see the paired table), MobileCLIP by about 9 points. Transcript-retrieved frames did not clearly beat transcript-only.
2. **Choosing frames by image content beat choosing them by subtitle match.** BM25 usually finds the right ~60 s transcript window, but the retrieval strategy samples only its start, middle and end, so its frames can be tens of seconds from the moment the question refers to. MobileCLIP compares frame images with the question and can pick the exact frame. (Likely explanation, consistent with the numbers; not separately tested.)
3. **But image-based selection was not clearly better than simply spreading frames evenly**, and it costs about a second more per question.
4. **The hand-tuned cost-aware heuristic was worse than MobileCLIP top-k** and no better than uniform: a controller has to be learned or validated, not hand-set.
5. **The longest videos (15–60 min) are the hardest for every strategy**, most likely because 8 frames cover an hour thinly.

**Why the absolute accuracy is modest.** The answer model has 2B parameters and sees at most 8 frames at 640 px, on one 16 GB consumer GPU. LongVideoBench rewards many frames of long videos; the leaderboard leaders use 7B–72B models or GPT-4o with 128–256 frames. The ceiling most likely comes from the model size and frame budget, which are configuration settings; a larger model and more frames have not been benchmarked yet.

**Not measured here:** the answer model given the raw video with its own default frame sampling (no pipeline), open-ended answers, and the correctness of the model's timestamp citations.

## Published reference points (official leaderboard)

| model | size | frames | LongVideoBench val accuracy |
|---|---|---|---|
| GPT-4o (0513) | proprietary | 256 | 66.7% |
| Gemini-1.5-Pro (0514) | proprietary | 256 | 64.0% |
| LLaVA-Video-7B-Qwen2 | 7B | 128 | 61.1% |
| GPT-4o-mini | proprietary | 250 | 56.5% |
| Idefics2 | 8B | 16 | 49.7% |
| Phi-3-Vision-Instruct | 4.2B | 16 | 49.6% |
| LLaVA-Next-Mistral-7B | 7B | 8 | 49.1% |
| **This framework — MobileCLIP top-k frames** | 2B | ≤ 8 | 41.6% |
| LLaVA-1.5-7B | 7B | 8 | 40.3% |

**How to read this comparison.** The leaderboard numbers are the models' own published results on the full 1,337-question validation set, each with its own protocol and frame count (8–256 frames). This framework was run on the subset of validation questions whose videos are held locally, with a much smaller model and far fewer frames, so the rows are reference points, not a controlled head-to-head. The controlled comparison is the strategy table above: same model, same questions, same frame budget.

Source: https://longvideobench.github.io/ (leaderboard read 2026-09-29).

Reproduce: `bash scripts/run_lvb_benchmark.sh` then `python scripts/benchmark_report.py`. Every prediction is in `records.jsonl`.
