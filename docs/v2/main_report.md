# v2 main experiment: does speech change which frames help, and can small heads learn it?

**Short answer.** *Speech does change which frames help, but the small learned
heads did not exploit it.* The retained transcript makes many frames redundant:
frame usefulness with and without speech agrees only moderately (Cohen's
κ = 0.56). This is a measured, reproducible effect. Every learned selector was
at best equal to simple baselines, though, and the trained frame-credit head
(H1) was significantly **worse** than zero-shot transcript relevance. The
pre-declared hypotheses are **not supported**. No novelty is claimed; see
`novelty_check.md`.

## Setup (declared before results)
- **Data:** 197 synthetic multiple-choice questions mined from **20 MIT OCW
  lectures** (Gilbert Strang, CC BY-NC-SA). 349 passed verification, and 152 were
  dropped by the declared rule (the answerer is right with no evidence at all).
  Leave-one-lecture-out: each lecture is evaluated by heads that never saw it.
- **Frozen models:** Qwen3-VL-2B-Instruct (answerer and teacher), MobileCLIP-S2,
  RapidOCR, MiniLM-L6 cross-encoder (Head A backbone).
- **Fixed for every arm:** the candidate pool (both paths, about 11 frames), the
  retained transcript (BM25 excerpt), the answerer, the prompt, and **K = 4 frames**.
- **Budget:** 6 GPU-hours declared, **3.4 h used** (pools 0.31 h, labels 1.97 h,
  eval 1.16 h). Data creation (download, mining, verification) was separate,
  about 2.5 h.
- **Labels:** Head B gets 9 calls per question (baseline + 8 single frames, with
  the retained transcript). Head A uses separate interventions on an empty
  context (line alone vs frame alone) for each Path A moment: 1,182 moments.

## Results (197 questions, 20 held-out lectures)
| Arm | Accuracy | Composed latency (median s) |
|---|---|---|
| Transcript only | 15.2% | — |
| A. MobileCLIP top-k | 34.5% | 4.48 |
| B. MobileCLIP + diversity | 36.0% | 4.48 |
| C. Head B with OCR (3 seeds) | 34.7% | 7.66 |
| C. Head B, no OCR (3 seeds) | 32.8% | 4.48 |
| C. Head B, no OCR, no speech features (3 seeds) | 33.2% | 4.48 |
| **E. Trained Head A frame credit** (H1) | **30.5%** | 4.49 |
| F. Zero-shot transcript relevance | 38.6% | 4.48 |
| G. Trained Head A text credit (control) | 37.1% | 4.49 |

*Transcript-only is below the 25% chance level because of the declared filter:
it keeps exactly the questions the answerer misses without evidence. That is a
selection effect, not a pipeline bug.*

Paired differences (percentage points; 95% CI from a lecture bootstrap, 20 lectures):

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| A − transcript only | +19.3 | +11.9 to +27.0 | **frames matter** |
| E − F (H1: frame credit vs relevance) | **−8.1** | −13.2 to −3.6 | **H1 refuted: trained frame credit is worse** |
| E − G (frame credit vs text credit) | −6.6 | −13.0 to −0.6 | refuted |
| E − A | −4.1 | −10.3 to +1.5 | not significant |
| C (no OCR) − A (equal latency) | −1.7 | −7.5 to +3.7 | no gain at equal cost |
| C (OCR) − A | +0.2 | −5.7 to +5.7 | no gain, and +3.2 s |
| C (OCR) − C (no OCR) | +1.9 | −1.5 to +5.3 | OCR does not help significantly |
| C (no OCR) − C (no speech features) | −0.3 | −3.2 to +2.2 | **the head does not use speech features** |
| F − A *(exploratory, not declared)* | +4.1 | −1.4 to +9.4 | not significant |

## The mechanism result: speech changes which frames help
For 894 frames measured both ways (same pixels, with vs without the retained
transcript; no extra teacher calls; `scripts/v2_transcript_conditioning.py`):

| | Useful with speech | Not useful with speech |
|---|---|---|
| **Useful without speech** | 140 | **132** |
| **Not useful without speech** | **15** | 607 |

- **Cohen's κ = 0.56** (95% CI 0.45–0.67). Usefulness agrees only moderately,
  so the transcript changes which frames help.
- The change is lopsided. **132 frames are useful only *without* speech** (the
  transcript already carried their information). Only **15 are useful only
  *with* speech** (true complementarity).
- Adding speech lowers the share of useful frames by **13.1 points** (CI 7.8–18.8).

**Interpretation:** on these lectures, retained speech mostly **substitutes**
for frames rather than **complementing** them. A selector should therefore
spend its frames where the transcript is *not* already informative. The idea
is sound in the labels, but the heads here failed to learn it:
- Head B's speech features changed nothing (−0.3 points).
- Head A's frame-credit output could not predict frame gain from text
  (out-of-fold AUC 0.50; zero-shot relevance 0.45).

## Why the learned parts failed (most likely causes)
- **Frame usefulness is not predictable from text.** Head A only sees the
  transcript line, but whether a frame helps depends on what is on the board.
  Its frame-credit labels were 303 frame-only vs 52 text-only positives.
- **Too little data for Head B.** About 1,600 single-frame labels across 20
  lectures, mostly zero, with ~1.2k-dimensional inputs.
- **The best simple signal was transcript relevance (F).** On chalkboard
  lectures, *where speech matches the question* points at the right moment
  better than CLIP's image similarity. This is consistent with v1, where
  MobileCLIP won on LongVideoBench's visual questions, and is exploratory here.

## What this project can honestly claim
1. A working, documented, fully local two-path QA pipeline with measured costs.
2. A measured effect: **retained speech substantially changes which frames are
   useful, mostly by making them redundant** (κ = 0.56, with a 132 vs 15
   asymmetry), on 197 questions from 20 lectures.
3. Negative results, with controls: compact learned selectors trained on
   answer-utility labels did not beat similarity, relevance or diversity
   baselines at equal cost, and per-segment frame credit learned from text was
   worse than zero-shot relevance.

## Next experiment, justified by these results
Give the head the information it lacked. Predict frame utility from **image +
OCR features conditioned on whether the retained transcript already answers**
(a cheap transcript-only draft, as in 2607.05438), trained on more lectures. The
label-level asymmetry suggests the decision "is speech already enough?" carries
most of the value. Also include **F (transcript relevance) as a baseline** from
now on.

## Reproduce
```
python scripts/research/fetch_lectures.py --limit 20 --out data/mit_lectures
python scripts/research/mine_moments.py --lectures data/mit_lectures --out runs/mined/v2 --limit-windows 32
python scripts/research/verify_moments.py --lectures data/mit_lectures --mined runs/mined/v2 \
       --qa-out data/mit_lectures/qa_v2.jsonl --drop-prior
python -m videoqa.v2.experiment pools --data data/mit_lectures --qa-file qa_v2.jsonl --run runs/v2_main
python -m videoqa.v2.experiment label --run runs/v2_main
python -m videoqa.v2.experiment eval  --run runs/v2_main
python scripts/v2_transcript_conditioning.py --run runs/v2_main
```
Result files: `reports/v2_main/`.
