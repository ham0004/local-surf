# Benchmark readiness (checked 2026-10-04)

Nothing below has been labelled or evaluated with Framework 2 yet. This page
records what each benchmark needs before teacher calls are spent on it.
Gate before labelling any benchmark: accessible videos, stable IDs,
video-disjoint splits, five real examples inspected, an explicit schema, a
working scorer, and a small manually audited scoring-error estimate.

| Check | EduVidQA (EMNLP 2025) | LongVideoBench | NExT-GQA |
|---|---|---|---|
| Role | lecture-domain training + untouched real test | fixed-selector transfer (MC) | MC QA training + grounding diagnostics |
| Questions available locally | 5,234 (train 3,909 / synthetic test 1,056 / real test 269) | 440-question subset (prior development) | not downloaded |
| Video-disjoint splits | **yes**: 157 / 40 / 99 videos, zero overlap | no official train split; validation 1,337, test 5,341 | official splits |
| Videos accessible | **no**: none downloaded (YouTube/NPTEL availability unverified) | 252 videos for the 440 subset (8.6 GB) | links only; CLIP features cannot feed Qwen |
| Transcripts | 87 / 296 videos (18 auto-generated); the rest failed to fetch | subtitles included | none; ASR would be a declared extension |
| Scorer | **missing**: free-text answers. `CachedTeacher` rejects them by design | official MC accuracy, already works | official MC accuracy; grounding metrics need val/test spans |
| Main constraint | keep real test (269) and synthetic test untouched; separate timestamp-given vs timestamp-hidden evaluation; do not convert to MC and call it official | disclose prior development on the 440 subset; keep an untouched final split | do not use released val/test spans to make training labels |

Next actions, in order:
1. **EduVidQA scorer.** Implement a free-text answer-quality scorer and audit
   it by hand on about 50 answers (the earlier Phi-4-mini judge was too
   lenient and its results were never committed). Only then label.
2. **EduVidQA videos.** Check how many of the 296 YouTube videos and their
   transcripts can still be fetched; report the accessible fraction per split.
3. **LongVideoBench.** Run the fixed selectors and any frozen learned scorer on
   a validation subset disjoint from the 440 already studied.
4. **NExT-GQA.** Download videos for a small train sample and confirm the MC
   format with five real examples before any labelling.
