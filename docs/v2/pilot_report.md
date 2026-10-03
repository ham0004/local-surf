# v2 pilot: does a context-dependent utility selector choose better frames?

**Hypothesis.** A small selector trained on context-dependent evidence utility
chooses better frames than similarity ranking or independently scored frames,
under the same teacher-labelling budget and measured inference cost.

**Verdict (pilot scale).** *Not supported.* Every learned arm is within noise of
MobileCLIP top-k, and history-conditioned labels did not beat single-frame
labels. The pilot did show that frames matter, that useful frames are usually
in the pool, and that speech credit and frame credit differ per segment. It is
too small (132 questions, 5 lectures, 86% zero labels) to detect a gain of a
few points. Novelty and performance remain unproven; see `novelty_check.md`.

## Setup
- **Data:** 132 synthetic multiple-choice questions mined from 5 MIT OCW
  lectures (Gilbert Strang; CC BY-NC-SA). Leave-one-lecture-out: the heads that
  evaluate a lecture never saw its labels.
- **Frozen models:** Qwen3-VL-2B-Instruct @89644892 (answerer and teacher),
  MobileCLIP-S2 (image/text embeddings), RapidOCR (CPU), MiniLM-L6
  cross-encoder (Head A backbone, zero-shot relevance in the pools).
- **Pool per question** (fixed for every arm): BM25 windows → Path A (6
  hot-moment frames) + Path B (up to 6 visual-scan frames, one per stable
  segment) → merged and de-duplicated. Mean 10.9 candidates; frames found by
  both paths are flagged.
- **Fixed for every arm:** candidate pixels, retained transcript (BM25
  excerpt, ≤120 words), answerer, prompt, and **K = 4 final frames**.
- **Budget:** a strict 2 GPU-hours, charged as wall time including CPU work.
  6,700 s were used: pools 936, audit 887, labels 2,328, eval 1,834, Head A
  labels 716. Nothing was extended.

## Audit (50 training questions, every single frame tested)
| | |
|---|---|
| Correct with transcript only | 23 / 50 |
| At least one frame fixes the answer | 13 / 50 (about half of the 27 wrong ones) |
| At least one frame makes it worse | 5 / 50 |
| Single-frame labels +1 / 0 / −1 | 50 / 465 / 24 (86% zero) |
| Determinism (20 re-asked evaluations) | 20 / 20 identical |
| Teacher speed | 1.46 s per call |

## Labels at the same nominal budget (9 calls per question per scheme)
| Scheme | Rows | Non-zero rows | Distinct (question, frame) pairs observed | New real calls | Cache hits |
|---|---|---|---|---|---|
| Independent (empty history) | 1,053 | 186 (17.7%) | 1,053 | 738 | 447 |
| Prefix chains (2 chains × 4 steps) | 1,056 | 103 (9.8%) | 854 | 839 | 349 |

Most independent-scheme hits come from the audit's earlier single-frame
evaluations. The prefix scheme observes fewer distinct frames and has
fewer non-zero labels: deeper contexts are labelled at the cost of coverage.
Real training rows (from `rows_*.jsonl`):

```json
{"qa_id": "mit:18.065_03:300", "question": "What is the lecturer referring to as 'a' in the context of the matrix multiplication...",
 "scheme": "independent", "history": [], "candidate": {"t": 366.3, "paths": ["B"], "clip_sim": 0.255},
 "before": 0.0, "after": 1.0, "gain": 1.0}
{"qa_id": "mit:18.065_03:1710", "question": "The lecturer refers to a matrix that is described as having how many rows and columns?",
 "scheme": "prefix", "history": ["...:f00", "...:f06"], "candidate": {"t": 327.2, "paths": ["A"], "clip_sim": 0.226},
 "before": 0.0, "after": 1.0, "gain": 1.0}
{"qa_id": "mit:18.065_03:1710", "scheme": "prefix", "history": [], "candidate": {"t": 1817.8, "paths": ["A"]},
 "before": 1.0, "after": 0.0, "gain": -1.0}
```

## Results (132 questions, 5 held-out lectures, K = 4)
| Arm | Accuracy | Composed latency (median s) |
|---|---|---|
| Transcript only | 44.7% | — |
| OCR text + transcript, no image (control) | 43.2% | — |
| A. MobileCLIP top-k | 56.1% | 5.53 |
| B. MobileCLIP + diversity (MMR) | 54.5% | 5.53 |
| C. Independent-label utility (3 seeds) | **57.6%** | 8.49 |
| D. History-conditioned utility, greedy (3 seeds) | 56.8% | 8.49 |
| D head scored once, no re-scoring (3 seeds) | 56.6% | 8.49 |

Paired differences (95% CI: question bootstrap / lecture bootstrap):

| Comparison | Difference | CI (questions) | CI (lectures) | Reading |
|---|---|---|---|---|
| A − transcript only | +11.4 | +5.3 to +18.2 | +4.4 to +19.6 | **frames matter** |
| A − OCR text, no image | +12.9 | +6.1 to +19.7 | +7.1 to +19.8 | **pixels matter; OCR text alone does not** |
| C − A (learned vs similarity) | +1.5 | −3.8 to +6.8 | −1.3 to +5.1 | not significant |
| D − A | +0.8 | −5.1 to +6.6 | −1.1 to +4.5 | not significant |
| D − B | +2.3 | −3.5 to +8.1 | −2.5 to +7.7 | not significant |
| D − C (label scheme) | −0.8 | −5.8 to +4.0 | −5.6 to +3.9 | **prefix labels did not help** |
| D greedy − D unary (re-scoring) | +0.3 | −3.3 to +3.8 | −3.3 to +2.6 | **re-scoring did not help** |

**Latency** is composed from per-question measured stage medians: decode 3.74 s,
OCR 2.94 s, MobileCLIP 0.33 s, Head A 0.07 s, retrieval 0.03 s, answer with 4
frames 1.36 s, selection < 0.01 s. Path B's scan decoding dominates every arm.
The learned arms also pay for OCR, about +3 s, for a gain that is not
significant. *These are composed estimates, not one end-to-end timed run.*

## Head A: separate text vs frame interventions (50 questions, 300 hot moments)
| | Text added (T0 = none) | Frame added (S0 = none) |
|---|---|---|
| Helped / neutral / hurt | 13 / 283 / 4 | 37 / 251 / 12 |

For the same moment the **frame helped but the line did not** in 29 cases, the
**line helped but the frame did not** in 5, and **both helped** in 8. So text
and frame credit genuinely differ per segment, which is the signal H1 needs.
Training the two-output head on these labels **failed** leave-one-lecture-out
(ROC-AUC trained vs zero-shot relevance: text 0.47 vs 0.63, frame 0.38 vs
0.55). 300 rows with 13/37 positives and 385 features overfit.

## What caused what (ablation reading)
- The only robust effect is **showing frames at all** (+11 to +13 points), and
  it comes from the pixels, not from OCR text.
- **Learned selection** (C, D) adds +0.8 to +1.5 points over similarity. That is
  inside noise, and costs about 3 s more per question for OCR.
- **History** (prefix labels, greedy re-scoring) adds nothing measurable at this
  scale. Prefix labels yield fewer non-zero examples than single-frame labels
  at equal budget, which plausibly explains it.

## Honest novelty assessment
The history-conditioned selector is covered in principle by Ross et al. (ICML
2013), and for frames by ReFoCUS (RL) and MarKey / FORTE (training-free). Answer-
signal supervision is covered by Frame-Voyager, TSPO and SeViLA. What remains
narrow and unproven: supervised signed utility conditioned on transcript and
history, a matched-budget label comparison, and per-segment speech-vs-frame
credit. This pilot gives **no evidence** that the selector beats similarity.

## Justified next experiment
1. **More labels, not a bigger model.** The bottleneck is 86% zero labels on
   132 questions. Mine the remaining ~100 MIT lectures (≈1,200 speech-dependent
   questions projected) and add a human-written test set before training
   anything larger.
2. **H1 with a smaller Head A.** Use scalar features only (relevance,
   overlap, OCR/frame similarity, position) to stop overfitting, then re-test
   per-segment text vs frame credit, the one signal the labels show.
3. **Make learned arms cost-competitive.** OCR only the top few candidates, and
   cut Path B's scan decoding (keyframes instead of a 3 s grid). Then compare at
   *equal measured latency*, not just equal frame count.
4. Only if 1–3 give a significant gain: a second domain (LongVideoBench) and a
   second answerer (Qwen3-VL-4B) for transfer.

## Reproduce
```
python -m videoqa.v2.pilot pools  --data data/mit_lectures --run runs/v2_pilot
python -m videoqa.v2.pilot audit  --run runs/v2_pilot
python -m videoqa.v2.pilot label  --run runs/v2_pilot
python -m videoqa.v2.pilot eval   --run runs/v2_pilot
python -m videoqa.v2.pilot head-a --run runs/v2_pilot
python scripts/v2_head_a_cv.py    --run runs/v2_pilot
```
Lectures: `python scripts/research/fetch_lectures.py --limit 5`; questions:
`scripts/research/mine_moments.py` and `verify_moments.py`. Result files are in
`reports/v2_pilot/`.
