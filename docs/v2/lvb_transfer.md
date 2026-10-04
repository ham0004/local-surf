# LongVideoBench transfer test (declared 2026-10-04, before any v2 run on it)

**Purpose.** Check whether two MIT development findings hold on a real, public
benchmark with naturally balanced answer positions:

1. **Balanced scan** improved MobileCLIP top-4 on MIT (+4.6, CI −0.5 to +10.4).
2. **Zero-shot transcript relevance** was the strongest MIT selector.

**Data.** The 440 LongVideoBench validation questions on 252 videos already
used to develop and benchmark v1 (v1 pipeline, 41.6% with MobileCLIP top-k).
v2's scan policies, pool builder and selectors were never run or tuned on
them. This is **not** an untouched final test, and it is not the full
benchmark (validation has 1,337 questions). Gold positions are balanced
(105/103/92/83/57 over 4–5 options), so the MIT option-order problem does not
apply.

**Protocol (fixed now).**
- Pools: `python -m videoqa.v2.experiment pools --data data/longvideobench_full
  --qa-file qa.jsonl --no-ocr --scan-policy {legacy,balanced}`. OCR is off
  because none of the compared selectors use it.
- Same pool config as MIT (scan step 5 s, cap 24, Path A 6, Path B 6), K = 4,
  the same BM25 retained subtitles (≤ 120 words), frozen Qwen3-VL-2B.
- Arms on both pools: MobileCLIP top-4 (A), MMR (B), zero-shot relevance (F).
  `scripts/v2_pool_compare.py` checks that only the scan differs.
- **Primary (H-T1):** balanced − legacy for A, video-bootstrap 95% CI.
- **Secondary (H-T2):** F − A on legacy pools.
- Reported: per-arm accuracy, pool sizes, stage costs, fresh calls.

**Decision rule.** Balanced scanning becomes the default only if H-T1's
interval excludes zero on this set. Otherwise it stays opt-in and the MIT
result is reported as not transferring.
