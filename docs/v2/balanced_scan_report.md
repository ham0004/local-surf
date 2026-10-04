# Balanced visual scan: candidate-generation experiment

**Short answer.** Spreading the same 24 scan slots across all retrieved
windows (instead of filling the earliest windows first) raised MobileCLIP
top-k from 34.5% to **39.1%** (+4.6 points, 95% CI −0.5 to +10.4), at the same
cost. The interval includes zero, so this is a promising, **not established**,
improvement. The relevance selector barely changed (+0.5). Balanced scanning
stays opt-in.

## Setup
- **Only the pool changes.** Same 197 questions, retained transcripts, frozen
  Qwen3-VL-2B, K = 4, Path A budget (6), Path B budget (6), scan cap (24) and
  step (5 s). New pools in `runs/v2_balanced`; the legacy pools are the
  published `runs/v2_main`.
- **No retraining.** Only the three label-free selectors are compared, so the
  new pool is not mixed into the learned-reranker comparisons.
- **Cost:** 579 fresh answerer calls (1,123 s) for the balanced arms; the
  legacy arms were fully cached.

## Results
| | Legacy scan | Balanced scan | Balanced − legacy (95% CI) |
|---|---|---|---|
| MobileCLIP top-k | 34.5% | **39.1%** | +4.6 (−0.5 to +10.4) |
| MobileCLIP + MMR | 36.0% | 37.6% | +1.5 (−4.3 to +8.0) |
| Zero-shot relevance | 38.6% | 39.1% | +0.5 (0.0 to +1.7) |
| Pool evidence recall | 74.6% | 77.7% | +3.0 (0.0 to +6.3) |
| Pool size (median) | 11 | 11 | — |
| Label-free selector cost (median stages) | 2.42 s | 2.45 s | ≈ equal |

Selection evidence recall (selected frames inside the question's evidence
interval): MobileCLIP 43.1% → 39.1%, MMR 53.3% → 44.2%, relevance 64.0% →
63.5%.

## Reading
- With balanced pools, **MobileCLIP ties relevance** (39.1% each). Under the
  legacy scan it trailed by 4.1 points. Part of relevance's apparent
  advantage was a candidate-coverage artefact: the legacy scan offered
  MobileCLIP frames only from the earliest windows.
- Relevance mostly chooses Path A frames, which do not depend on the scan, so
  it barely moves.
- MobileCLIP accuracy rose while its selection evidence recall fell. The
  evidence intervals are the generator's coarse 30 s mining windows; useful
  board frames often lie outside them (a board written earlier stays visible).
  Evidence recall is therefore a weak proxy here and accuracy is the outcome.
- Costs are composed stage medians, not end-to-end timings. Both policies
  decode the same number of frames; label-free selectors need no OCR.

## Status
Not adopted as the default: the main gain's interval includes zero, and the
MIT set has already been used for development. The next confirmation should
use an untouched set (see `benchmark_readiness.md`) with the policy fixed in
advance.

## Reproduce
```
python -m videoqa.v2.experiment pools --data data/mit_lectures --qa-file qa_v2.jsonl \
    --scan-policy balanced --run runs/v2_balanced
python scripts/v2_pool_compare.py --legacy runs/v2_main --balanced runs/v2_balanced --out reports/v2_balanced
```
