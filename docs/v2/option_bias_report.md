# Option-position bias in the MIT question set, and partially corrected results

## The problem
- The generator (Qwen3-VL-4B) was never asked to shuffle options, and its
  JSON example used `"answer_index": 0`. Gold is option A or B in 282 of 349
  verified questions.
- The "drop if answerable without evidence" filter then removed C/D-gold
  questions more often (61%) than A/B-gold ones (39%). The 2B answerer prefers
  C/D, so it "answers" those without evidence by luck.
- The 197 kept questions are therefore **anti-aligned with the answerer's
  letter preference**: gold is A/B in 171 of 197; with transcript only the
  model picks C/D in 142 of 197. "Always B" would score 46.7%, and
  transcript-only scored 15.2%, below chance.

Details: `reports/v2_review/option_position_bias.json`.

## The fix: circular evaluation
Every question is asked under all 4 cyclic rotations of its options (MMBench's
CircularEval). The score is the mean over rotations; "strict" needs all four
correct. Frames, transcript and selections are unchanged.
`scripts/v2_circular_eval.py`, 3,495 fresh answerer calls.

| Arm (K = 4) | Original order | Circular mean | Strict (all 4) |
|---|---|---|---|
| Transcript only | 15.2% | 34.3% | 2.0% |
| MobileCLIP top-4 (legacy pools) | 34.5% | 49.7% | 23.4% |
| MMR (legacy pools) | 36.0% | 51.5% | 24.4% |
| Zero-shot relevance (legacy pools) | 38.6% | 52.2% | 25.9% |
| MobileCLIP top-4 (balanced pools) | 39.1% | **53.3%** | 25.9% |
| Zero-shot relevance (balanced pools) | 39.1% | 52.4% | 25.9% |

Accuracy by rotation varies from 34.5% to 67.0% for the SAME frames
(MobileCLIP, legacy), which shows how strong the position effect is.

Paired differences on circular means (lecture bootstrap):

| Comparison | Difference | 95% CI |
|---|---|---|
| Frames (MobileCLIP) − transcript only | +15.5 | +11.4 to +19.6 |
| Relevance − MobileCLIP (legacy) | +2.4 | −0.5 to +5.5 |
| MMR − MobileCLIP (legacy) | +1.8 | −0.8 to +4.6 |
| **Balanced − legacy scan, MobileCLIP** | **+3.6** | **+0.4 to +6.9** |
| Balanced − legacy scan, relevance | +0.3 | −0.3 to +1.2 |
| Relevance − MobileCLIP (balanced) | −0.9 | −4.4 to +2.6 |

## What changes in the earlier conclusions
- **Absolute accuracies** on MIT were understated by about 15 points; the
  "below chance" transcript-only result was an artefact.
- **Selector rankings** are unchanged in direction; relevance's lead over
  MobileCLIP is not significant after debiasing.
- **Balanced scanning** becomes a significant improvement for MobileCLIP on
  this development set. With it, label-free MobileCLIP matches relevance.
- **Label-based analyses** (single-frame utility, the speech-conditioning
  decomposition, the fourth-frame completion table) used the original option
  order only. Their counts include position-bias effects; in particular the
  "transcript interferes with a correct frame" cases may partly reflect
  letter preference. They are kept as recorded, with this caveat.

## What rotation does not fix
Rotation corrects how the 197 questions are scored. It does not restore the 152
questions removed by the filter, verify the synthetic answers, rebuild the
single-order training labels, retrain the heads, or rebuild the fourth-frame
table. Four cyclic orders also cover 4 of the 24 possible permutations.
Close prior work: PriDe (arXiv 2309.03882) on positional bias and Ovcharov
(arXiv 2608.15428) on model-dependent filtering; this is a new instance of a
known effect, not a new mechanism.

## Prevention
New question generators must shuffle options with a recorded seed, and any
answerer-conditioned filter must be checked for gold-position balance before
and after filtering.
