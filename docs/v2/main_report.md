# v2 main experiment: does speech change which frames help, and can small heads learn it?

**Short answer.** *Retained speech changes measured frame utility, but the small
learned heads showed no reliable selection gain.* Positive frame-utility labels
with and without speech agree moderately (Cohen's κ = 0.56). The differences
include both redundant successes and negative interference. Every learned
selector was at best statistically indistinguishable from simple baselines,
and the trained frame-credit head
(H1) was significantly **worse** than zero-shot transcript relevance. The
pre-declared hypotheses are **not supported**. No novelty is claimed; see
`novelty_check.md`.

**Audit correction (2026-10-04).** The historical accuracy, confidence intervals
and timing estimates below are preserved. Reanalysis of existing labels corrects
the interpretation of the 132 lost positive gains. The historical speech
ablation also retained speech-derived Head A scores and proposal-path flags;
the code now masks them, and the corrected ablation was rerun on the same pools,
labels and seeds (32.5%; summary in
[`reports/v2_review/eval_summary_corrected_ablation.json`](../../reports/v2_review/eval_summary_corrected_ablation.json)).
Derived audit outcomes are in
[`reports/v2_review/transcript_conditioning_review.json`](../../reports/v2_review/transcript_conditioning_review.json).

## Setup (declared before results)
- **Data:** 197 synthetic multiple-choice questions mined from **20 MIT OCW
  lectures** (Gilbert Strang, CC BY-NC-SA). 349 passed verification, and 152 were
  dropped by the declared rule (the answerer is right with no evidence at all).
  Leave-one-lecture-out: each lecture is evaluated by heads that never saw it.
  These are synthetic, generator-screened questions with answerer-conditioned
  filtering, not an independent human-verified benchmark of ordinary questions.
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
- **Scope:** both paths use transcript-retrieved windows. Pools are proposed by
  zero-shot relevance; the trained Head A arms only rerank this fixed pool.
  Main-run Head B ranks single-frame gains once, with empty frame history;
  prefix labels and greedy history selection were tested in the pilot.

## Results (197 questions, 20 held-out lectures)
| Arm | Accuracy | Composed latency (median s) |
|---|---|---|
| Transcript only | 15.2% | — |
| A. MobileCLIP top-k | 34.5% | 4.48 |
| B. MobileCLIP + diversity | 36.0% | 4.48 |
| C. Head B with OCR (3 seeds) | 34.7% | 7.66 |
| C. Head B, no OCR (3 seeds) | 32.8% | 4.48 |
| C. Head B, no OCR, partial speech ablation (historical, 3 seeds) | 33.2% | 4.48 |
| C. Head B, no OCR, corrected speech ablation (3 seeds) | 32.5% | 4.48 |
| **E. Trained Head A frame credit** (H1) | **30.5%** | 4.49 |
| F. Zero-shot transcript relevance | 38.6% | 4.48 |
| G. Trained Head A text credit (control) | 37.1% | 4.49 |

*The filter selects no-evidence errors, not transcript-only errors. It can
depress correlated transcript-only performance; the 25% random-guess reference
does not imply expected accuracy on this conditional challenge set. The +19.3
points is the paired effect of adding selected frames on this set, not a learned
selector improvement or an estimate for ordinary lecture questions.*

*Latencies are sums of measured stage medians and a shared answer-call median,
not end-to-end timed medians. Historical no-OCR costs omit the nearby-speech
embedding time; equal inference cost has not been established.*

Paired differences (percentage points; 95% CI from a lecture bootstrap, 20 lectures):

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| A − transcript only | +19.3 | +11.9 to +27.0 | **frames matter** |
| E − F (H1: frame credit vs relevance) | **−8.1** | −13.2 to −3.6 | **H1 refuted: trained frame credit is worse** |
| E − G (frame credit vs text credit) | −6.6 | −13.0 to −0.6 | refuted |
| E − A | −4.1 | −10.3 to +1.5 | not significant |
| C (no OCR) − A | −1.7 | −7.5 to +3.7 | no reliable gain; cost estimate incomplete |
| C (OCR) − A | +0.2 | −5.7 to +5.7 | no gain, and +3.2 s |
| C (OCR) − C (no OCR) | +1.9 | −1.5 to +5.3 | OCR does not help significantly |
| C (no OCR) − C (partial speech ablation) | −0.3 | −3.2 to +2.2 | historical incomplete ablation; no conclusion about all speech features |
| C (no OCR) − C (corrected speech ablation) | +0.3 | −2.9 to +3.5 | **speech features give Head B no measurable gain** |
| F − A *(exploratory, not declared)* | +4.1 | −1.4 to +9.4 | not significant |

## The mechanism result: speech changes which frames help
For 894 matched question/frame pairs measured both ways (same pixels, with vs without the retained
transcript; no extra teacher calls; `scripts/v2_transcript_conditioning.py`):

| | Useful with speech | Not useful with speech |
|---|---|---|
| **Useful without speech** | 140 | **132** |
| **Not useful without speech** | **15** | 607 |

- **Cohen's κ = 0.56** (95% CI 0.45–0.67) measures chance-adjusted agreement
  of positive-gain labels. It is not a statistical independence test.
- **132 pairs lose positive gain with speech; 15 gain it.** Positive gain is
  measured against different correctness baselines, so a zero gain can mean
  either that the transcript already succeeded or that the combined input failed.
- Adding speech lowers the share with positive frame gain by **13.1 points**
  (CI 7.8–18.8). These pairs cluster within 197 questions and 20 lectures;
  intervals resample lectures. Only matched Path A/Head B labels are included.

The audit reconstructs correctness from each label's baseline and gain:

| Among the 132 lost positive gains | Frame alone | Speech alone | Combined | Count |
|---|---|---|---|---|
| Redundant success | correct | correct | correct | **67** |
| Destructive combination | correct | correct | wrong | **3** |
| Speech interferes with a successful frame | correct | wrong | wrong | **62** |

Thus **65 of 132 combined inputs fail despite a correct frame-only answer**;
the earlier claim that all 132 reflect transcript redundancy was incorrect.
The 15 gained positives are joint-only successes for this answerer: neither
source alone succeeds, but the combination does. These are correctness outcomes
against synthetic gold, not proof of where semantic information resides.

Head A's frame-credit output had out-of-fold AUC 0.50 (zero-shot relevance 0.45).
The corrected Head B ablation (+0.3 points, CI −2.9 to +3.5) does not establish
an accuracy benefit from this head's speech features. The corrected
feature mask removes nearby speech, transcript distance, Head A scores, proposal
flags and history path overlap. It still evaluates **the same transcript-retrieved
pool**; upstream retrieval and proposals are not speech-free.

## Why the learned parts failed (most likely causes)
- **This Head A failed to predict frame usefulness from text.** Head A sees
  transcript context but no board pixels, and its empty-context label target
  differs from deployment with a retained transcript and four frames.
  Its frame-credit labels were 303 frame-only vs 52 text-only positives.
- **Too little data for Head B.** About 1,600 single-frame labels across 20
  lectures, mostly zero, with 1,556-dimensional inputs before projection.
- **The best simple signal was transcript relevance (F).** On chalkboard
  lectures, *where speech matches the question* is a useful baseline; its
  observed advantage over CLIP is exploratory and its interval includes zero.

## What this project can honestly claim
1. A working, documented, fully local two-path QA pipeline with stage costs
   and composed latency estimates.
2. A measured effect: **retained speech changes positive frame-utility labels**
   (κ = 0.56, with a 132 vs 15 asymmetry), including redundant successes and
   negative interference, on 197 filtered synthetic questions from 20 lectures.
3. Negative results, with controls: compact learned selectors trained on
   answer-utility labels did not beat similarity, relevance or diversity
   baselines in this experiment, and per-segment frame credit learned from text was
   worse than zero-shot relevance.

## Next experiment, justified by these results
Head B already receives **image, OCR and nearby-speech features**. Add a direct
representation of the **retained transcript's content and answerability**, which
its current features do not provide; a transcript-only draft or uncertainty
signal is one candidate, with its inference cost measured. Separate already
correct baselines from negative interference, and train on additional lectures.
Keep **F (transcript relevance)**,
MobileCLIP and diversity baselines; measure full end-to-end latency and use a
human-verified, untouched lecture/course test set. Whether this improves accuracy
remains an experimental question.

## Reproduce
```
python scripts/research/fetch_lectures.py --limit 20 --out data/mit_lectures
python scripts/research/mine_moments.py --lectures data/mit_lectures --out runs/mined/v2 --limit-windows 32
python scripts/research/verify_moments.py --lectures data/mit_lectures --mined runs/mined/v2 \
       --qa-out data/mit_lectures/qa_v2.jsonl --drop-prior
python -m videoqa.v2.experiment pools --data data/mit_lectures --qa-file qa_v2.jsonl --run runs/v2_main
python -m videoqa.v2.experiment label --run runs/v2_main
python -m videoqa.v2.experiment eval  --run runs/v2_main
python scripts/v2_transcript_conditioning.py --run runs/v2_main \
       --out reports/v2_review/transcript_conditioning_review.json
```
Historical result files: `reports/v2_main/`. Audit-derived outcomes:
`reports/v2_review/transcript_conditioning_review.json`. Re-running evaluation with the
corrected feature mask is a new experiment and will not reproduce the historical
partial-ablation score; use a new run directory rather than overwrite the archive.
