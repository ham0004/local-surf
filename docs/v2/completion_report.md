# Fourth-frame completion: results

**Short answer.** The declared hypothesis is **refuted**. A scorer trained on
deployment-matched fourth-frame labels did not beat the baseline's own fourth
frame. It scored −1.0 points (95% CI −2.5 to 0.0) and fell back to the
baseline in 18 of 20 folds. The exhaustive labels still produce three
measured findings about where the fourth slot's value lies.

Design, budget and closest prior work were committed before any label was
collected (`novelty_check.md`, commit `ab311aa`).

## Setup
- **Data:** the same 197 synthetic MIT questions, 20 lectures, frozen pools and
  retained transcripts as the main run (development data, not an external test).
- **Anchor:** top-3 frames by zero-shot transcript relevance (the strongest
  simple baseline). Every other pool candidate was labelled as the fourth frame:
  R(T, top-3 + {c}). 1,500 completions plus 197 three-frame references.
- **Cost:** 1,697 evaluations, of which **1,368 were fresh answerer calls**
  (329 served from the main run's exact-evidence cache). 2,563 s of GPU time
  (1.87 s per call). The single-frame labels used 1,770 evaluations.
- **Learner:** pairwise ridge residual anchored on the baseline's rank-4 pick.
  Features cover the candidate plus its redundancy and novelty relative to the
  chosen frames and retained speech (26 features). Nested leave-one-lecture-out
  cross-validation with a baseline fallback. Labels never enter features.

## Results (exact four-frame QA accuracy, 197 questions)
| Fourth-frame policy (same three anchor frames) | Accuracy |
|---|---|
| Three frames only (reference, not a K=4 arm) | 33.5% |
| MobileCLIP-best remaining frame | 33.5% |
| MMR (similarity − redundancy with anchor) | 34.0% |
| Random remaining frame (exact expectation) | 35.1% |
| Residual scorer trained on single-frame labels | 36.0% |
| **Completion head (deployment-matched labels)** | **37.6%** |
| **Relevance rank-4 (baseline)** | **38.6%** |
| Oracle best fourth frame | 45.2% |

Paired differences (percentage points, lecture bootstrap):

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| Completion head − relevance rank-4 (**primary**) | −1.0 | −2.5 to 0.0 | **refuted** |
| Completion head − single-frame residual (secondary) | +1.5 | −2.7 to +7.0 | not significant |
| Oracle − relevance rank-4 | **+6.6** | +3.6 to +9.9 | real headroom exists |
| MMR − relevance rank-4 | **−4.6** | −8.1 to −1.5 | diversity hurts here |
| MobileCLIP − relevance rank-4 | **−5.1** | −9.5 to −1.0 | image similarity hurts here |

## What the exhaustive labels show
1. **The fourth slot has measurable headroom: +6.6 points**, from 38.6% to
   45.2%. It sits in a small set of questions: the choice of fourth frame
   changes the answer in only **36 of 197** questions.
2. **A fourth frame can break a correct answer.** In 13 of the 66 questions
   the three anchor frames answer correctly, at least one fourth frame makes the
   answer wrong. No question was broken by every candidate. In 23 of 131
   anchor-wrong questions, some fourth frame rescues the answer.
3. **On these lectures, transcript relevance is the best completion signal.**
   Its own rank-4 frame is significantly better than the MobileCLIP-best
   (−5.1) and MMR (−4.6) choices, and better than random. Visual diversity,
   which MarKey/FORTE-style selectors optimise, is harmful in this setting.

## Why the learner failed (most likely)
- **Too few informative questions.** Only 36 questions have any variation in
  the completion label, about 1.8 per lecture. Inner cross-validation saw no
  regularisation that beat the fallback in 18 of 20 folds.
- **The available features do not identify the rescuing frame.** Lexical OCR
  novelty, image redundancy and time gaps are weak proxies on chalkboard
  frames (RapidOCR is weak on chalk; MobileCLIP sees boards as similar).

## What this adds to the project's claims
- A deployment-matched label target did **not** rescue learned selection at
  this scale. Neither single-frame nor completion labels beat transcript
  relevance.
- **Measured headroom** for the last frame (+6.6 points) with an exact oracle,
  plus the finding that diversity-based completion is worse than relevance
  here. Both are reproducible from the committed per-question table.
- These are development results on synthetic, generator-screened questions
  from one course. Before generalising, they need an untouched,
  human-verified test set (see `benchmark_readiness.md`).

## Reproduce
```
python scripts/v2_completion.py label   --run runs/v2_completion
python scripts/v2_completion.py analyze --run runs/v2_completion --out reports/v2_completion
```
Outputs: `reports/v2_completion/summary.json`, `folds.json`, and
`per_question.jsonl`, the full completion table that any new fourth-frame
policy can be scored against without answerer calls.
