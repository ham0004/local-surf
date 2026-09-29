# LongVideoBench full run — results (2026-09-27)

Everything here is produced by `scripts/run_lvb_full.sh` and `scripts/analyze_premise.py`;
raw numbers are in the JSON files next to this one. Frozen answerer: Qwen3-VL-2B-Instruct
@89644892 (bf16, RTX 5060 Ti). Scout: MobileCLIP-S2. Relevance: `annotation` (the dataset's own
quoted-subtitle evidence only). Damage: DELETE.

## Data

| | |
|---|---|
| Videos / questions | 252 / 440 (every LongVideoBench val video with a subtitle-quoting T* question) |
| Fair damage triples | 330 — train 208, dev 48, calibration 31, test 43 (video-level splits) |
| Manipulation check | answer-relevant line reaches the answerer in 317/330 clean and 317/330 control transcripts vs 5/330 targeted |
| Labelling cost | 16,981 real answerer calls, 4.8 GPU-hours (train 13,498 calls / 3.7 h; dev 3,483 / 1.1 h) |

## 1. Main hypothesis: paired training → selective looking — **null**

Selectivity = frames under targeted damage − frames under matched control damage, per test
question (clean cancels). 95% intervals resample whole videos (43 questions / 43 videos).

| Paired − unpaired selectivity at the same STOP threshold | diff | 95% CI |
|---|---|---|
| each head's own dev-tuned threshold | −0.093 | [−0.415, +0.146] |
| threshold 0.0 | +0.070 | [−0.649, +1.000] |
| threshold 0.1 | +0.070 | [−0.179, +0.425] |
| threshold 0.3 | −0.023 | [−0.077, +0.000] |
| threshold 0.6 | 0.000 | [0, 0] (neither head looks) |

Dev-set model quality over 5 training seeds (mean ± sd):

| | paired | unpaired |
|---|---|---|
| dev pair-difference MAE | 0.1558 ± 0.0045 | 0.1588 ± 0.0055 |
| dev realised net gain of the tuned policy | 0.0105 ± 0.0095 | 0.0177 ± 0.0132 |

No difference between paired and unpaired training, at any operating point.

## 2. Why: the premise does not hold on this dataset

The hypothesis needs looking to be *more valuable* when answer-relevant speech is lost than when
equal unrelated speech is lost. Measured directly on matched label pairs (same question, same
candidate frame, only the transcript differs), `premise_train.json` / `premise_dev.json`:

| | train (208 q, 3,185 pairs) | dev (48 q, 741 pairs) |
|---|---|---|
| gain_targeted − gain_control | **−0.022** [−0.050, +0.004] | **−0.016** [−0.106, +0.083] |
| pairs with identical gain | 92.5% | 87.6% |

Base accuracy with no frames is not lowered by the targeted damage either (train: clean 0.312,
targeted 0.356, control 0.312). No question category shows a positive effect (largest: T2A
+0.023).

Interpretation: in LongVideoBench's subtitle-anchored questions the **answer is visual** ("eggs",
"Wearing a helmet") and the quoted subtitle only says *when* to look — and that quote is repeated
in the question text itself. Removing the line from the transcript therefore removes neither the
answer nor the anchor from the answerer's view. There is no signal for any controller to learn,
paired or not. **This dataset cannot test the hypothesis**; it behaves as a negative control.
A valid test needs questions whose answer is *spoken* and *also visible* (e.g. a lecturer reading
out a number that is on the slide) — the design of the synthetic `q_lr` question, and plausibly
lecture datasets such as EduVidQA.

## 3. What the run does show

Test accuracy (43 questions; chance ≈ 0.22–0.25 for 4–5 options), paired differences with
video-clustered 95% CIs:

| comparison (clean transcripts) | Δ accuracy | 95% CI | frames used |
|---|---|---|---|
| scout similarity (6 frames) − uniform (6 frames) | **+0.140** | [+0.048, +0.250] | 6 vs 6 |
| scout similarity − transcript only | **+0.140** | [+0.020, +0.286] | 6 vs 0 |
| uniform − transcript only | 0.000 | [−0.105, +0.116] | 6 vs 0 |
| learned (paired, t=0) − transcript only | +0.047 | [−0.073, +0.179] | 2.9 vs 0 |
| scout similarity − learned (paired, t=0) | +0.093 | [+0.000, +0.200] | 6 vs 2.9 |

* **Which frames matter more than how many:** six uniformly spaced frames add nothing over the
  transcript alone, while six question-similar frames add +0.14.
* **The learned controllers mostly learn to stop.** Only 8–9% of individual frame actions change
  the answer for the better (and 6–8% for the worse), so the measured utility signal from this 2B
  answerer is sparse and noisy, and the dev-tuned policies look at < 1 frame on average. They are
  cheap but not significantly more accurate than transcript-only.
* **Transcript-gated retrieval degrades exactly when speech is lost**: the retrieval baseline
  looks at 0.28 fewer frames under targeted damage (selectivity −0.23 [−0.53, −0.02]).

## Caveats

43 test questions from 43 videos; intervals are wide. One damage type (DELETE). One frozen
answerer. The learned head is a 32-unit MLP over hand-designed features, not the LLM controller
described in the research plan.
