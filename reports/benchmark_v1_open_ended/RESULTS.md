# Open-ended benchmark: LongVideoBench without options (v1 framework)

The same 440 LongVideoBench questions as the multiple-choice benchmark, but the **options are hidden**: the model writes its own answer, which is graded against the dataset's correct answer text. Same model (Qwen3-VL-2B), same ≤8-frame budget, same GPU.

## How answers are graded

A local judge model (microsoft/Phi-4-mini-instruct, a different model family from the answerer, greedy decoding) reads the question, the reference answer and the candidate answer, and returns correct/incorrect plus a 0–5 score. The exact judge prompt is in `results.json`. Token F1 (word overlap with the reference) is reported alongside as a judge-free check. `human_audit.csv` holds 50 random answers for a person to grade; until that is done, the judge's agreement with humans is **not yet measured**.

## Results

| strategy | judge accuracy | 95% CI (by video) | mean score (0–5) | token F1 | frames | time / question |
|---|---|---|---|---|---|---|
| Transcript only (no frames) | **44.1%** | 39.1–49.1% | 2.24 | 0.068 | 0.0 | 1.42 s |
| Uniform frames + transcript | **33.4%** | 28.5–38.2% | 1.80 | 0.084 | 6.0 | 1.64 s |
| Transcript-retrieved moments | **32.0%** | 27.5–36.6% | 1.74 | 0.083 | 4.3 | 1.50 s |
| MobileCLIP top-k frames | **32.3%** | 27.9–36.9% | 1.75 | 0.086 | 6.0 | 2.85 s |
| Cost-aware heuristic controller | **29.8%** | 25.2–34.3% | 1.65 | 0.082 | 5.7 | 1.92 s |

Judge replies that could not be parsed (counted as incorrect): 0.

**Caveats.** LongVideoBench is a multiple-choice benchmark; its reference answers are short option texts (e.g. "black short hair"), so this measures short free answers, not long explanations. There is no published open-ended leaderboard for it, so these numbers compare the strategies with each other, not with other models.

Reproduce: `evaluate --clean-only --open-ended` (see `scripts/run_lvb_benchmark.sh`), then `python scripts/judge_open_ended.py`.
