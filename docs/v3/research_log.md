# v3 research log

A dated record of every step of the v3 study: what was done, why, what ran, what it cost, what came
out, and what was decided. Newest entries at the bottom. Commit hashes refer to branch `framework-2`.
Detailed results live in the linked reports; this log is the chronological index.

---

## 2026-10-05 — Plan and test-set lock

**Step 1. Research plan before implementation** (`c23d86e`)
- Why: v2's trained selectors never beat simple rules; the next stage (learned temporal Head A,
  set-selecting Head B, open-ended answers) needed a written plan before any spending.
- Done: mapped the v2 implementation against the v3 specification, chose EduVidQA (EMNLP 2025) as the
  candidate benchmark, defined two gates (G2: validate the answer scorer; G1: check that frames help at
  all), shortlisted three hypotheses (speech-anchored offset Head A, context-conditioned set Head B,
  operating choices), and proposed per-stage call/time budgets.
- Literature read for the plan: TAN (CVPR 2022: only ~30% of narration is visually alignable, ~15% well
  aligned), AKS (training-free relevance + coverage keyframe baseline), the EduVidQA paper (metrics,
  protocol, its own warning on visual dependence).
- Document: `docs/v3/research_plan.md`.

**Step 2. Lock the final test** (`0be99f1`)
- Decision: EduVidQA's 269 expert-verified real-world questions are the untouched final test; all
  development uses the official synthetic training split, split by video.

## 2026-10-05 to 10-08 — Open-ended evaluation tooling

**Step 3. Open-ended answering and scoring** (`f928a4e`, `307a540`, `9a8f9fd`, `5cfca42`, `e553dd9`)
- Open-ended prompt (256 tokens) for the frozen Qwen3-VL-2B with a request-identity answer cache and a
  call/time budget.
- EduVidQA video access: 20 of 20 sampled training videos downloadable (video-only, ≤ 480p, 0.79 GB).
- Scoring: the paper's FactQA precision/recall prompt (verbatim) plus BLEU-1, ROUGE-L, METEOR and an
  NLI entailment score. Judge choice: Gemini's free tier allowed only 20 requests/day and its paid tier
  was declined; after testing four free NVIDIA NIM models on the same item, Nemotron-3 Ultra 550B was
  chosen (most careful verdict; one fixed judge for all answers).
- A human labelling page for 50 answers was built for judge validation (optional).

## 2026-10-08 to 10-09 — Gates on EduVidQA

**Step 4. Gate G1: do evenly placed frames help?** (`fda4e59`)
- 92 training questions × {transcript only, transcript + 4 evenly spaced frames}: 184 answers.
- Result (74 judged): FactQA precision −0.094, recall −0.098 (both CIs below 0).

**Step 5. Gate G1b: do selected frames help?** (`4559e9b`, `66cb5ec`)
- Added one frame at the question timestamp (paper protocol) and MobileCLIP top-4 frames.
- Result (72 questions): both lower FactQA (reference frame precision −0.096, recall −0.123; MobileCLIP
  precision −0.087, recall −0.128).

**Step 6. Data fault found and fixed** (`c38e625`)
- A 4,895-token prompt revealed Tamil caption tracks labelled "en": 6 of 20 gate videos, 10 of 87
  transcripts. Script-based language check added (`src/videoqa/v3/language.py`).
- Re-analysis on English transcripts: frames leave precision unchanged but lower recall by ~0.10.

**Step 7. Larger answerer check** (`805b90b`, `aa02de2`)
- Qwen3-VL-4B, transcript vs MobileCLIP top-4, 35 English-transcript questions: precision −0.003,
  recall −0.046 (no gain). Conclusion: the limitation is the data, not model size.

**Step 8. Verify the explanation before reporting it** (`710652e`)
- Measured with the FactQA judge: a question's own ±120 s transcript supports 49% of its reference
  answer's claims (another video's transcript: 0.6%). "The transcript contains the answer" is therefore
  only partly true; reports were corrected.
- Report: `docs/v3/gate_g1_report.md`.

## 2026-10-09 — Choosing a benchmark where frames matter

**Step 9. Benchmark survey from primary sources** (`710652e`, `e793f04`)
- Compared CG-Bench, Video-MMMU, LongVideoBench, Video-MME, Neptune and EduVidQA (size, format, evidence
  that frames matter, splits, access). Document: `docs/v3/benchmark_survey.md`.
- Downloaded the small files after accepting both dataset agreements:
  - CG-Bench: 117 of 1,219 videos have English subtitles; the release archives hold 770 videos, 51 of
    them English-subtitled.
  - Video-MMMU: all 300 videos released; Perception = 277 OCR + 23 ASR multiple-choice questions.
- Decision: Video-MMMU Perception is the next gate (its questions need the screen); CG-Bench (long
  videos, human evidence intervals) is the long-video development set.

**Step 10. Video-MMMU and CG-Bench data preparation** (`2104677`, `343ac15`)
- Video-MMMU: 301 videos extracted, 600 multiple-choice questions (Perception + Comprehension). Declared
  split by video: 1/3 dev (98 + 98 questions), 2/3 test (202 + 202), test untouched.
- ASR: Whisper large-v3-turbo with word timestamps grouped into caption-like lines (~23 s per
  3.7-minute video), because Video-MMMU ships no transcripts.
- CG-Bench: byte-range fetcher downloads only the 51 English-subtitled videos from the remote archives.
- Gate declared (`scripts/v3_gate_videommmu.py`): question only / transcript / + 4 uniform frames /
  + MobileCLIP top-4, exact letter scoring, dev only, 400 calls / 1 GPU-hour; passes if frames beat the
  transcript with a CI above zero.
- Status: dev transcription running; the gate starts automatically when it finishes.

**Step 11. CG-Bench long-video sample downloaded** (2026-10-09)
- `scripts/research/fetch_cgbench_videos.py` read the 32 remote archive directories and downloaded only
  the 51 videos that have English subtitles: 15.21 GB instead of ~128 GB of whole archives. Videos are in
  `data/cgbench/videos/`, subtitles in `data/cgbench/transcripts/` (not in git; log in
  `data/cgbench/fetch_log.json`).
- These 51 videos (~500 questions with human evidence intervals) are the long-video development pool;
  their train/dev/test split by video will be declared before any result.

**Step 12. CG-Bench question file and declared split** (2026-10-09)
- `scripts/research/prepare_cgbench.py`: 510 questions on the 51 English-subtitled videos, each with
  choices, gold index, human clue intervals, domain and sub-category.
- Split declared before any CG-Bench result, by video (sha256 rule): train 37 videos / 373 questions,
  dev 8 / 82, test 6 / 55. Custom protocol (CG-Bench has no official split). The test part is small;
  results on it will carry wide intervals and are reported as such.
- Gold answers spread over A–H (largest E = 85/510); 6–8 options per question.
- Manifest: `reports/cgbench_manifest.json`.

**Step 13. CG-Bench gate declared** (2026-10-09)
- `scripts/v3_gate_cgbench.py`, run on the 82 dev questions (8 videos): question only; BM25-retrieved
  transcript (≤ 300 words); + 4 uniform frames; + MobileCLIP top-4 (4 s scan, ≥ 8 s apart); and E4 =
  4 frames inside the human clue intervals. E4 uses gold evidence locations, so it is an upper
  reference for frame selection (selection headroom = E4 − C4), not a method. Budget 420 calls / 1 h.
- Queued to run after the Video-MMMU gate.

**Step 14. Three pipeline faults fixed while running the Video-MMMU gate** (2026-10-09)
- MobileCLIP encoded a whole-video scan in one batch and ran out of GPU memory; images are now encoded
  in batches of 64 (`src/videoqa/v2/encoders.py`).
- Whisper word timestamps occasionally ran backwards (e.g. [68.76, 61.94]); ASR segments are now forced
  to be time-ordered; 67 of 98 dev transcripts were repaired in place (`--repair`), none re-run.
- The answer prompt and parser only knew options A–H; Video-MMMU has up to 10 (one question 14). Letters
  now run A–P, unchanged for ≤ 8 options, and the parser prefers the reply's leading letter and ignores
  the pronoun "I". Regression tests added for all three.

**Step 15. Video-MMMU gate: frames help — gate passes** (2026-10-09)
- 97 dev Perception questions, frozen Qwen3-VL-2B, exact letter scoring, 388 calls / 1,552 s.
- Accuracy: question only 28.9%, transcript 40.2%, + 4 uniform frames 52.6% (+12.4, CI +2.1 to +22.7),
  + 4 MobileCLIP frames 56.7% (+16.5, CI +6.2 to +26.8). MobileCLIP vs uniform +4.1 (CI −3.1 to +12.4).
- Decision: Video-MMMU is a benchmark where frame selection can show an effect; it becomes the main
  development and test benchmark for the selection methods (dev third for development, test two thirds
  untouched). Report: `docs/v3/gate_videommmu_report.md`.

**Step 16. CG-Bench gate: selection headroom measured** (2026-10-09)
- 82 dev questions (8 long videos, median 39 min, evidence windows median 14 s), 410 calls / 447 s.
- Accuracy: question only 15.9%, retrieved transcript 15.9%, + 4 uniform frames 17.1%, + 4 MobileCLIP
  frames 40.2% (+24.4 over transcript, CI +14.9 to +32.5), + 4 frames inside the human evidence 63.4%.
  Headroom of a perfect selector over MobileCLIP: +23.2 (CI +11.3 to +36.5).
- Decision: the main research target is Head A ("where to look") on long videos, where uniform frames fail
  and better localisation is worth up to ~23 points. CG-Bench's human evidence intervals provide direct
  supervision for it. Report: `docs/v3/gate_cgbench_report.md`.

**Step 17. Phase 2 plan: learned "where to look" (Head A) for long videos** (2026-10-09)
- Decision from the gates: the largest measurable headroom (+23 points over MobileCLIP) is localisation in
  long videos, and CG-Bench's human evidence intervals can supervise it. Plan: question-conditioned
  temporal scorer over frozen MobileCLIP features, offline hit@4 and online QA evaluation, pilot on the 51
  videos before scaling. Document: `docs/v3/phase2_plan.md`.

**Step 18. Features and Head A implementation** (2026-10-09)
- `scripts/v3_features.py`: frozen MobileCLIP-S2 embedding every 2 s of each whole video (~1 minute per
  40-minute video); running on the 51 CG-Bench videos.
- `src/videoqa/v3/temporal_head.py`: Head A as a residual over MobileCLIP's question–frame similarity,
  with question conditioning (FiLM), time features and a 2-layer transformer over time (~1 M parameters).
  An untrained head ranks frames exactly like MobileCLIP (unit-tested), so any change is learned.
- `scripts/v3_head_a.py`: pilot declared before any result (protocol in the script docstring).

**Step 19. Head A pilot: does not pass (overfitting on 306 questions)** (2026-10-09)
- Dev hit@4 (82 questions; a chosen frame inside the human evidence): random 8.6%, uniform 7.3%,
  MobileCLIP 43.9%, smoothed MobileCLIP 45.1%, AKS 43.9% (threshold 0.05 tuned on train), Head A 46.3%
  (3-seed score average). Head A − MobileCLIP = +2.4 (CI −6.1 to +11.0): the declared rule (CI above 0)
  is not met. Training data: 306 fit / 67 val questions from 37 train videos.
- Diagnosis from the curves: training loss falls throughout while held-out train-video hit@4 drops from
  ~0.39 (epochs 0–7, i.e. close to the MobileCLIP starting point) to ~0.12–0.16 by epoch 29; the kept
  checkpoints are essentially untrained. The model (~1 M parameters) overfits a few hundred questions.
- Decision: the pilot is reported as failed. Because the failure has a specific, testable cause (too
  little supervision), the next experiment scales the training data rather than changing the method:
  Head A needs no subtitles, so all released CG-Bench videos except the 14 dev/test videos can supply
  training questions with human evidence. This is declared as a new experiment with its own stop rule
  (if scaled Head A still does not beat MobileCLIP on dev, Head A is stopped). Report:
  `reports/v3_head_a/offline_dev.json`.

**Step 20. Option-aware query (label-free baseline improvement, train-only finding)** (2026-10-09)
- On the 37 pilot train videos only (dev untouched), MobileCLIP retrieval with "question + options" text
  instead of the question alone raises hit@4 from 30.8% to 36.2% (+5.4, CI +1.6 to +9.0, video bootstrap).
  Mean of question and option embeddings: 31.6%; question + max over options: 33.5%.
  (`reports/v3_head_a/query_variants_train.json`). Adding options to the query is a common practice, not a
  contribution; it is adopted as the stronger baseline and as Head A's input.

**Step 21. Scaled Head A declared; training features streaming** (2026-10-09)
- `scripts/research/cgbench_train_features.py` streams the 719 other released CG-Bench videos (dev/test
  excluded by id): byte-range download, 2 s MobileCLIP features, video deleted (no 160 GB stored).
  ~30–80 s per video, ~9 hours in total, running.
- `scripts/v3_head_a_scaled.py` (declared before results): training on all those videos' questions,
  config chosen on an inner 10% of training videos, 3 seeds, one dev comparison against option-aware
  MobileCLIP with a stop rule (CI must exclude 0).

**Step 22. Head A method research and design choice** (2026-10-09)
- Read and compared the main families for "where to look": training-free relevance + coverage (AKS,
  MarKey, FORTE, GIFT, Ground-Cover-Refine), answer-aware queries (2609.31668), temporal search with object
  detection (T*, LV-Haystack), learned temporal agents (TSPO), grounding, and answer-utility selectors.
- Diagnostic on train: option-aware MobileCLIP puts the human evidence among its top-32 candidates for 71%
  of questions but in its top-4 for only 36%. The problem is ranking, not recall.
- Decision: build Head A as retrieve-then-verify (cheap top-32 shortlist, precise frozen verification
  signals targeted at measured failure types, low-capacity learned ranker trained on human evidence).
  Test order and stop rules in `docs/v3/head_a_methods.md`.

**Step 23. Roles corrected; the two paths are complementary** (2026-10-09)
- Correction: in the project architecture Head A is the transcript path (speech close to the query → where
  the picture is). The whole-video MobileCLIP scorer of steps 18–21 is Path B work, and the
  retrieve-then-verify ranker of step 22 is Head B (choosing final frames from the pooled candidates).
- Path diagnostic on CG-Bench train: speech-line proposals reach the evidence for 38% of questions (top 6),
  MobileCLIP for 42% (top 6), the pool of both for 64% (6 + 6) and 80% (16 + 16). The paths find different
  evidence, which supports the two-path design. Head A's measurable target: raise the transcript path's
  evidence recall at a fixed proposal budget. Details: `docs/v3/head_a_methods.md`.

**Step 24. Back to the handoff: original designs for Head A and Head B** (2026-10-09)
- Re-read `FRAMEWORK_HANDOFF.md`. Steps 18–22 had drifted from it by building a whole-video visual scorer
  under the name Head A; the plan of step 1 (speech-anchored Head A, set-selecting Head B) is restored.
- Head A (proposed): speech-to-sight offset localiser — per transcript line, question-conditioned relevance
  plus an offset distribution (−40 to +40 s) for where the picture is; trained on CG-Bench clue intervals and
  EduVidQA question timestamps (weak). Ablation: offset fixed at 0.
- Head B (proposed): answer-discriminative evidence set selector — set model over pooled A + B candidates
  using per-option evidence profiles (decisive and complementary evidence); trained on human evidence, then
  answerer swap labels. Ablations: no option profile; no set context.
- Document: `docs/v3/heads_design.md`.

**Step 25. Process correction: return to the handoff's sequence** (2026-10-09)
- The handoff's order is: benchmark/scorer readiness → **baseline cycle** (framework settings tuned on the
  common benchmark's dev split and frozen) → research and a 2–3 method shortlist with an original design →
  one full cycle per method (Head A, then Head B) against the frozen baseline → combined + ablations → final
  frozen comparison on the same test questions → external comparison.
- Steps 18–22 jumped from the gates straight to training a learned component, skipping the baseline cycle,
  and the component was not the handoff's Head A. Those runs are kept as Path B diagnostics; they do not
  replace the baseline cycle.
- Also restored from the handoff: open-ended answers are the primary objective. CG-Bench provides open-ended
  reference answers, so dev/test comparisons will report open-ended FactQA (validated judge) next to
  multiple-choice accuracy.
- Next, in order: (1) baseline cycle on CG-Bench dev and Video-MMMU dev; (2) Head A cycle; (3) Head B cycle;
  (4) combined and ablations; (5) final comparison.

**Step 26. Starting framework written down** (2026-10-10)
- `docs/v3/baseline_framework.md`: the v2 two-path pipeline with rules only (BM25 windows → Path A speech lines
  + Path B visual scan → de-duplicated pool → rule selector → frozen Qwen3-VL-2B), where Head A and Head B
  plug in later, and the settings the baseline cycle tunes on dev before freezing the baseline.

**Step 27. Baseline cycle started (declared trials)** (2026-10-10)
- Runner `scripts/v3_baseline_cycle.py`; framework option added: Path B scope "video" (whole-video scan,
  cap 48) next to v2's "windows" (`PoolConfig.path_b_scope`). OCR off (no rule selector uses it).
- Trials on CG-Bench dev (82) and Video-MMMU Perception dev (98), one setting at a time from v2 defaults:
  1. Pools {v2, balanced, video} × selectors {clip, clipopt, mmr, mmropt, relevance}, K = 4, 120 words.
  2. Path ablation on the v2 pool: Path A only, Path B only (option-aware ranking).
  3. On the best pool + selector: K ∈ {2, 8}; transcript ∈ {0, 300} words.
- Budget: 4,000 answer calls / 4 GPU-hours (one ledger). Results: `reports/v3_baseline/table.json`.
- The Video-MMMU test transcription (147/301 done) is paused to free GPU memory; it resumes before the
  final test (resumable).

**Step 28. Head A data: text and evidence times only** (2026-10-10; preparation, no training)
- Head A reads the question and the timed speech, so its data needs subtitles and evidence times, not
  videos. `scripts/v3_head_a_data.py` builds `data/head_a/{train,dev}.jsonl`:
  - CG-Bench: the release's subtitles.zip has English subtitles for 83 train-split videos (877 questions
    with human clue intervals) and 17 dev-split videos (164 questions), also for videos whose video file we
    do not hold. Test-split videos are skipped.
  - EduVidQA synthetic_train: questions that mention a time ("At 3:54, ...", "Referring to the slide at
    1:46, ..."); the time is removed from the question (otherwise the model reads the answer position from
    the text) and used as weak evidence (time ± 20 s). 546 questions have an English caption track; 2,870
    more lack one, so their audio is being downloaded for Whisper transcription
    (`fetch_eduvidqa_videos.py --audio-only`; some videos return HTTP 403 and are skipped).
- Train 1,423 questions / 117 videos; dev 164 / 17 (CG-Bench only, human intervals).

**Step 29. Head A harness and the zero-shot Path A rules on dev** (2026-10-10)
- `scripts/v3_head_a_eval.py`: every Path A method returns ranked moments; metric = share of questions with
  at least one of the first k moments inside a human clue interval (± 1 s). Dev, 164 questions:

  | Method (zero-shot) | R@1 | R@2 | R@4 | R@6 |
  |---|---|---|---|---|
  | uniform 6 moments (chance) | 3.0 | 6.1 | 9.8 | 12.8 |
  | BM25 window centres | 7.9 | 9.8 | 14.0 | 14.0 |
  | BM25 single lines | 13.4 | 18.9 | 23.2 | 26.2 |
  | v2 Path A (MiniLM inside BM25 windows, line end) | 10.4 | 15.9 | 25.0 | 25.0 |
  | MiniLM over all lines | 11.0 | 15.9 | 26.2 | 28.0 |

  (8 s non-maximum suppression changes these by at most ±2 points.) The v2 rule's 6 speech moments reach
  the evidence for 1 question in 4: large room for a learned Head A.
- Where is the evidence relative to the best-matching line? (train, 877 human-interval questions;
  `reports/v3_head_a/offsets_train.json`): evidence centre − line end is within ±10 s for 166 (19%),
  10–40 s away for 109 (12%), 40–120 s for 104 (12%), more than 120 s for 498 (57%).
- Reading: the main failure is choosing the wrong line (57% far away), not the offset; when the line is
  near, the evidence is often 10–40 s before or after it. Head A must therefore learn line relevance first
  and the offset second. Methods to compare in the Head A cycle (after the baseline is frozen):
  (a) learned relevance on frozen MiniLM features; (b) fine-tuned cross-encoder; (c) (b) + offset
  distribution (the proposed design); (d) (c) + multi-line evidence aggregation over a time grid; with the
  zero-shot rules above as baselines.

**Step 30. Baseline cycle: one more Path B setting (dense whole-video features)** (2026-10-10)
- Interim stage-1 numbers on CG-Bench dev (K = 4, 120 words): pools built inside the BM25 windows hold an
  evidence frame for only 32–33% of questions and reach 23–32% accuracy; the whole-video even scan (cap 48)
  raises pool evidence recall to 50% and accuracy to 34% (MobileCLIP top-K). Both are below the gate's
  dense rule (MobileCLIP top-4 over 2 s features of the whole video, 40.2%, gate_cgbench_report.md). A
  baseline that ignores that rule would be too weak a bar for the heads, so the rule enters the cycle as a
  pool setting:
  - `PoolConfig.path_b_scope = "features"`: Path B ranks the precomputed 2 s MobileCLIP features of the
    whole video by question similarity, decodes the top 24 (≥ 4 s apart), keeps one per stable run and the
    best 6, as in the other settings. Path A is unchanged.
  - Same selectors, K = 4, 120 words, plus Path A-only / Path B-only on this pool
    (`scripts/run_baseline_dense.sh`). Video-MMMU dev features computed for this (98 videos,
    `v3_features.py --split dev`).

**Step 31. Baseline cycle stages 1–2: results** (2026-10-10; `reports/v3_baseline/table.json`)
- Multiple-choice accuracy, K = 4 frames, 120 transcript words, frozen Qwen3-VL-2B. CG-Bench dev 82
  questions (8 videos; "evidence" = an evidence frame is in the pool); Video-MMMU Perception dev 98.

  | Pool (Path B scope) | Selector | CG-Bench | Video-MMMU |
  |---|---|---|---|
  | v2 (BM25 windows; pool evidence 33%) | clip / clipopt / mmr / mmropt / relevance | 28.0 / 26.8 / 26.8 / 28.0 / 23.2 | 56.1 / 54.1 / 56.1 / 58.2 / 51.0 |
  | balanced windows (pool evidence 32%) | same order | 29.3 / 30.5 / 31.7 / 30.5 / 23.2 | 55.1 / 57.1 / 57.1 / 59.2 / 51.0 |
  | whole video, even (pool evidence 50%) | same order | **34.1** / 30.5 / 29.3 / 31.7 / 22.0 | 57.1 / **59.2** / 58.2 / 58.2 / 48.0 |
  | v2 pool, Path A only / Path B only (option-aware) | | 25.6 / 30.5 | 55.1 / 51.0 |

- References on the same CG-Bench dev questions (gate, 300 words): question only 15.9, evenly spaced 4
  frames 17.1, dense MobileCLIP top-4 40.2, frames inside the human evidence 63.4.
- Reading:
  - CG-Bench: where Path B looks matters most. Scanning the whole video raises pool evidence recall from
    33% to 50% and the best accuracy from 28–32% to 34%; still below the dense rule (40.2%), hence the
    dense setting (step 30, running).
  - The speech-relevance selector is the weakest rule on both benchmarks (22–23% CG-Bench, 48–51%
    Video-MMMU): ranking frames by the zero-shot speech score of the nearest line does not find the
    picture. This is the Path A weakness Head A targets (step 29).
  - Video-MMMU: all rules except relevance lie within 54–59%; with 98 questions these differences are
    within noise. On lectures the Path A frames alone (55.1) do better than Path B alone (51.0); on
    CG-Bench the reverse (25.6 vs 30.5): the two paths are useful on different kinds of video.

**Step 32. Baseline cycle: dense Path B results; hybrid setting declared** (2026-10-10)
- Dense pool (Path B from 2 s whole-video MobileCLIP features), K = 4, 120 words:

  | Selector | CG-Bench (pool evidence 62%) | Video-MMMU |
  |---|---|---|
  | clip / clipopt / mmr / mmropt / relevance | 32.9 / **37.8** / 32.9 / 34.1 / 25.6 | 49.0 / 52.0 / 55.1 / 54.1 / 50.0 |
  | Path A only / Path B only | 25.6 / 36.6 | 55.1 / 48.0 |

- Paired differences (95% CI, bootstrap over videos):
  - CG-Bench: dense+clipopt − v2+clip +9.8 (−1.2..+21.0); dense+clipopt − video+clip +3.7 (−5.3..+13.4);
    option-aware vs question-only on the dense pool +4.9 (**+1.0..+10.1**, the only clear difference).
  - Video-MMMU: dense+clipopt − v2+clip −4.1 (−11.2..+3.1); video+clipopt − v2+clip +3.1 (−4.1..+10.2).
- Reading: on long videos the evidence is usually not where the speech matches (pool evidence 33% with
  BM25-window scanning, 62% with the dense whole-video scan); on lectures the frames near the matching
  speech do at least as well. Accuracy differences between settings are mostly within the noise of 82/98
  dev questions; evidence recall (human labels, CG-Bench) is the steadier signal.
- Declared next trial: `hybrid` Path B, the best 3 frames from the BM25-window scan plus the best 3 of the
  rest (dense whole video), same selectors. Then the configuration with the best mean over both
  benchmarks (and the higher CG-Bench evidence recall on ties) goes to stage 3 (K, transcript budget,
  path budgets) and is frozen.

**Step 33. Baseline cycle: hybrid Path B results; setting chosen for stage 3** (2026-10-10)
- Hybrid pool (best 3 Path B frames from the BM25-window scan + best 3 of the rest), K = 4, 120 words:

  | Selector | CG-Bench (pool evidence 60%) | Video-MMMU |
  |---|---|---|
  | clip / clipopt / mmr / mmropt / relevance | 29.3 / 34.1 / 32.9 / **35.4** / 25.6 | 55.1 / 55.1 / 58.2 / **58.2** / 51.0 |
  | Path A only / Path B only | 25.6 / 34.1 | 55.1 / 52.0 |

- Mean of the two benchmarks (declared rule, step 32): hybrid + mmropt 46.8; even whole-video + clip 45.6;
  hybrid + mmr 45.6; dense + clipopt 44.9; v2 + clip (the v2 default) 42.0. Chosen for stage 3:
  **hybrid pool + MMR on question and options (mmropt)**. Its advantages over the next settings are within
  dev noise; it is chosen by the declared rule, and its CG-Bench pool evidence recall (60%) is close to
  the best (dense, 62%) while it keeps the lecture-friendly window frames.
- Stage 3 started on it (`scripts/run_baseline_stage3.sh`): K ∈ {2, 8}; transcript 0 or 300 words; Path A
  3 instead of 6 proposals; Path B 12 instead of 6.
- The EduVidQA audio download finished (122 of 157 videos; the rest returned HTTP 403); Whisper
  transcription of the 122 is running alongside (data/eduvidqa/transcripts_asr, Head A training data).
- The CG-Bench train-video feature extraction (246 of 719 done) is paused to keep the GPU for the
  baseline cycle; it is resumable and will serve Head B training data.

**Step 34. Baseline cycle stage 3 on hybrid + mmropt** (2026-10-10)
- One setting changed at a time from K = 4, 120 words, Path A 6 + Path B 6:

  | Change | CG-Bench | Video-MMMU |
  |---|---|---|
  | (setting of step 33) | 35.4 | 58.2 |
  | K = 2 | 34.1 | 50.0 |
  | K = 8 | 36.6 | (stopped at 49/98: GPU out of memory next to the Whisper job; rerun) |
  | no transcript (0 words) | **40.2** | **63.3** |
  | 300 words | 34.1 | 59.2 |
  | Path A 3 proposals | 36.6 | 57.1 |
  | Path B 12 proposals (pool evidence 68%) | 34.1 | 56.1 |

- Paired differences (95% CI, bootstrap over videos):
  - no transcript − 120 words: CG-Bench **+4.9 (+1.4..+7.7)**, Video-MMMU **+5.1 (+0.0..+11.2)**;
  - 300 − 120 words: −1.2 (−3.9..+0.0) and +1.0 (−3.1..+5.1);
  - K = 2 − K = 4: −1.2 (−6.9..+5.2) and **−8.2 (−15.3..−2.0)**;
  - hybrid + mmropt, no transcript − v2 default (BM25-window pool, clip, 120 words): CG-Bench
    **+12.2 (+4.6..+21.5)**, Video-MMMU +7.1 (−1.0..+15.3).
- Reading: with this 2B answerer the BM25 transcript excerpt lowers multiple-choice accuracy on both
  benchmarks; the frames carry the answer and the excerpt distracts. This holds for the multiple-choice
  format; whether it also holds for open-ended answers is measured before freezing (open-ended is the
  primary objective).
- Follow-up trials (no transcript): K = 2 and 8; selectors clip / clipopt / mmr; Path A 3; Path B 12. The
  Whisper job was paused for the GPU (resumable).

**Step 35. Baseline answer budget used up; second ledger** (2026-10-10)
- The declared ledger (4,000 answer calls, local GPU, no cost) was used up after stages 1–3 and part of the
  no-transcript follow-ups (3,919 calls had run; the runner stopped cleanly). Done before the stop:
  K = 8 at 120 words (CG-Bench 36.6, Video-MMMU 59.2), K = 2 without transcript on CG-Bench (37.8), and
  81/82 CG-Bench questions of K = 8 without transcript.
- A second ledger is declared for the remaining follow-ups of step 34 (about 1,100 calls):
  `runs/v3_baseline/budget_2.json`, 1,500 calls / 2 GPU-hours (`--ledger budget_2.json`). Open-ended answers
  keep their own ledger (1,500).

**Step 36. No-transcript follow-ups, open-ended check, and the frozen baseline** (2026-10-10)
- Without transcript (K = 4 unless stated), CG-Bench / Video-MMMU: mmropt 40.2 / 63.3; mmr 39.0 / 65.3;
  clipopt 39.0 / 63.3; clip 32.9 / 58.2; mmropt K = 2 37.8 / 52.0; K = 8 39.0 / 60.2; Path A 3 proposals
  37.8 / 61.2; Path B 12 proposals 36.6 / 59.2. No change beats mmropt at K = 4 (mmr's mean is 0.4 points
  higher, about one question; a change is kept only when it helps).
- Open-ended answers on CG-Bench dev (82; answers generated without options; short human references):
  - Judge: Nemotron-3-Ultra (NVIDIA NIM free tier), our one-line correct/incorrect prompt
    (`ShortAnswerJudge`), 246 verdicts. Hand check of 40 random verdicts: 40/40 agree (2 borderline:
    "motherboard" vs "computer host", father-son vs father-daughter). Three references are broken in the
    dataset (bare "3", "4" or "Can" for non-count questions); they count as incorrect for every method.
  - Correct: frozen setting without transcript 19.5%; with 120 words 17.1%; v2 default 13.4%.
    Paired: no transcript − v2 +6.1 (−1.3..+13.3); no transcript − 120 words +2.4 (−2.5..+6.5). Same
    direction as multiple choice; not significant on 82 questions.
- **Baseline frozen** (`docs/v3/baseline_framework.md`, "Frozen baseline"): hybrid Path B, MMR on question
  + options, K = 4, no transcript to the answerer. Dev: CG-Bench 40.2% MC / 19.5% open-ended / 60% pool
  evidence; Video-MMMU 63.3%. This is the bar for Head A and Head B. Next: Head A cycle (step 29 plan).

**Step 37. Head A cycle, method family 1: fine-tuning the MiniLM cross-encoder — negative result** (2026-10-10)
- `scripts/v3_head_a_train.py`: the MiniLM cross-encoder fine-tuned on train (CG-Bench human intervals +
  EduVidQA weak times) to rank lines within 10 s of the evidence above hard (high zero-shot, far) and random
  negatives; with or without an offset head (`ft_rel`, `ft_rel_off`).
- The evaluation path was checked first: the untrained model reproduces the zero-shot numbers exactly (dev
  R@6 28.0; with uniform offsets 21.3; with the density proposals 28.0).
- One epoch, lr 2e-5 (single random near line as positive) or 1e-5 (multiple-instance loss over the near
  lines; human labels only): dev R@6 **7.9** and **9.1**; inner validation 19.2 vs 44.6 for the zero-shot
  model. Training destroys the pretrained relevance ranking: for most CG-Bench questions the speech near the
  evidence is unrelated to the question (step 29: the best zero-shot line is > 120 s from the evidence for 57%),
  so "near the evidence" is a very noisy relevance label for an encoder with 22M free parameters.
- Decision: do not tune the encoder. Methods must keep the zero-shot ranking as their starting point:
  (2) a light residual head over frozen line features, (3) stronger frozen rerankers.

**Step 38. Head A cycle, method family 3: stronger frozen rerankers** (2026-10-10; `reports/v3_head_a/rerankers_dev.json`)
- Every line scored by a public reranker (line + one neighbour each side), zero-shot, question only (q) or
  question + options (qo); dev, 164 questions. Revisions: MiniLM-L12 7b02352, bge-reranker-base 2cfc18c,
  bge-reranker-v2-m3 953dc6f.

  | Reranker | R@1 | R@2 | R@4 | R@6 |
  |---|---|---|---|---|
  | MiniLM-L6 (v2's, q) — reference | 11.0 | 15.9 | 26.2 | 28.0 |
  | MiniLM-L12 q / qo | 10.4 / 12.2 | 17.1 / 17.7 | 23.2 / 20.1 | 27.4 / 22.0 |
  | bge-reranker-base q / qo | 11.6 / 15.2 | 16.5 / 20.7 | 26.8 / 26.2 | 29.3 / 31.1 |
  | bge-reranker-v2-m3 (568M) q / qo | 12.2 / 14.0 | 18.3 / 18.3 | 23.2 / 25.6 | 25.6 / 31.1 |

- Reading: a 25× larger reranker adds at most 3 points at R@6. The limit is not text-matching quality: on
  long videos the speech rarely describes what the question asks about. Gains must come from how the speech
  evidence is turned into moments (context, question type, offsets), which is what the light head learns.

**Step 39. Head A cycle, method family 2: light residual head over frozen line features** (2026-10-10)
- `scripts/v3_head_a_light.py`, shared code in `src/videoqa/v3/head_a.py`. score_i = z_i (frozen MiniLM) +
  MLP(21 label-free line features: relevance context over ±1/3/7 lines, BM25, option overlap, length,
  position, speech density, question cues); last layer zero-initialised, so the untrained head is the
  zero-shot rule. Multiple-instance listwise loss over all lines (near = within 10 s of the evidence). Inner
  validation (15% of train videos) picks the epoch; 3 seeds; dev scored once per model.
- Dev recall (mean ± sd over 3 seeds; zero-shot MiniLM R@1 11.0, R@2 15.9, R@4 26.2, R@6 28.0):

  | Variant | Train data | R@1 | R@2 | R@4 | R@6 |
  |---|---|---|---|---|---|
  | line ranking (`light`) | + EduVidQA weak | 15.0 ± 0.6 | 20.9 ± 0.6 | 29.3 ± 0.5 | 32.7 ± 0.3 |
  | line ranking | human only | 15.0 ± 0.3 | 21.3 ± 1.0 | 29.5 ± 1.3 | 32.1 ± 1.3 |
  | density moments (`light_dens`) | + EduVidQA weak | 15.2 ± 0.9 | 21.1 ± 1.3 | 29.5 ± 0.8 | 32.5 ± 1.9 |
  | density moments | human only | 15.2 ± 0.5 | 23.4 ± 0.3 | 31.5 ± 1.0 | 35.2 ± 1.3 |
  | learned offsets (`light_off`, the proposed design) | + weak | 14.2 ± 0.3 | 22.6 ± 0.5 | 26.4 ± 0.3 | 30.7 ± 1.3 |
  | learned offsets | human only | 15.4 ± 0.6 | 21.1 ± 1.5 | 27.8 ± 2.0 | 31.7 ± 1.8 |
  | line ranking + bge features (`ens`) | human only | 17.9 ± 0.8 | 23.6 ± 1.6 | 30.5 ± 1.0 | 33.5 ± 1.0 |
  | **density moments + bge features** | human only | 16.9 ± 0.3 | 24.8 ± 1.3 | 32.3 ± 1.5 | **36.2 ± 1.6** |

- Findings:
  - Keeping the frozen ranking and learning a residual works; tuning the encoder did not (step 37).
  - Density moments (several relevant lines vote for a moment) beat single-line ranking by about 3 points
    at R@6.
  - The proposed learned offsets do not help (−3.5 vs density moments without offsets): with 877 human
    questions, an offset distribution per line is not learnable beyond the Gaussian vote. Reported as a
    negative result for the original Head A design.
  - The EduVidQA weak times (question time ± 20 s) do not help; their noise (~35 s average error, per the
    paper) is larger than the 10 s "near" window.
- Chosen Head A: **density moments + bge features, human labels, 3 seeds averaged** (no seed picked on dev).
  Re-trained with checkpoints (identical numbers). Seed average on dev: R@1 17.7, R@2 26.2, R@4 32.9,
  R@6 36.6; paired vs zero-shot R@6 **+8.5 (+4.1..+13.1)**, bootstrap over the 17 dev videos.
- Next: the frozen baseline with only Path A replaced, by (a) zero-shot MiniLM over the whole transcript and
  (b) the chosen Head A; multiple choice on both benchmarks plus Path A-only frames
  (ledger `budget_heada.json`, 1,500 calls).

**Step 40. Head A in the frozen pipeline (only Path A changed)** (2026-10-10; ledger `budget_heada.json`)
- Frozen baseline (hybrid Path B, mmropt, K = 4, no transcript) with Path A from: the v2 rule (MiniLM inside
  the BM25 windows), zero-shot MiniLM over the whole transcript (`hybrid_azs`), or the chosen Head A
  (`hybrid_ahead`). Path A-only = the 4 best Path A frames (option-aware ranking).

  | Path A | CG-Bench pool evidence | CG-Bench A-only | CG-Bench mmropt | Video-MMMU A-only | Video-MMMU mmropt |
  |---|---|---|---|---|---|
  | v2 rule (baseline) | 59.8 | 30.5 | **40.2** | **61.2** | **63.3** |
  | zero-shot, whole transcript | 64.6 | 29.3 | 39.0 | 60.2 | 62.2 |
  | Head A (learned) | **64.6** | **34.1** | 39.0 | 57.1 | 62.2 |

- Paired (95% CI): CG-Bench pool evidence Head A − baseline **+4.9 (+1.0..+10.8)**; CG-Bench A-only +3.7
  (−2.5..+9.5); CG-Bench mmropt −1.2 (−4.7..+2.9); Video-MMMU A-only −4.1 (−11.2..+2.0); Video-MMMU mmropt
  −1.0 (−7.1..+5.1).
- Reading:
  - On long videos Head A puts the evidence into the pool more often (+4.9, significant) and its own frames
    answer better (+3.7, not significant), but the frozen selector does not use them: with mmropt the
    chosen frames hit the evidence for 45% of questions with Head A vs 50% with the baseline pool. The
    selector ranks by MobileCLIP similarity, so a better speech-derived candidate is not recognised as
    better. The gain is lost at box [5], which is Head B's job.
  - On lectures, Head A (trained on CG-Bench vlogs and films) transfers badly: v2's in-window line-end rule
    is better there (A-only 61.2 vs 57.1). The supervision covers one domain only.
- Head A outcome so far: a real localisation gain on long videos (moment recall +8.5, pool evidence +4.9),
  no end-to-end QA gain through the frozen selector, and a domain-transfer loss on lectures. Both Path A
  sources (baseline and Head A pools) go into the Head B cycle; the combined A + B step decides whether
  Head A is kept.

**Step 41. Head B cycle, stage 1: rules and a learned evidence scorer** (2026-10-10; ledger `budget_headb.json`)
- `scripts/v3_head_b.py`. Training pools: the frozen baseline's pools for the 373 CG-Bench train questions
  (37 videos; 1.0 evidence candidate per pool on average). 18 label-free features per candidate (question /
  question+options similarity, option profile over "question + option" texts, provenance, Path A speech
  score, nearby speech, time, redundancy). Methods:
  - `optset` (training-free, the proposed option-evidence ingredient): greedy set gain = relevance +
    λ·Δ(decisiveness of the set's mean option profile) − μ·redundancy;
  - `ev`: MLP evidence scorer (listwise loss towards candidates inside the human interval), 3 seeds averaged,
    then greedy with redundancy;
  - `ev_set`: `ev` + the option-evidence term; ablation without set context (μ = 0).
- λ, μ chosen on the inner validation videos of train only (20%): `ev_set` λ = 2, μ = 1 (chosen-evidence
  recall 38.8% vs 32.8% with λ = 0); `optset` flat over the grid (32.8% everywhere), declared defaults kept
  (λ = 1, μ = 0.3). Scorer inner-validation recall@4 (pools with evidence): 0.73 / 0.73 / 0.79.
- Dev (K = 4, no transcript; chosen-evidence recall on CG-Bench in brackets):

  | Pool | Selector | CG-Bench | Video-MMMU |
  |---|---|---|---|
  | baseline | frozen mmropt | 40.2 (50.0) | 63.3 |
  | baseline | optset / ev / ev_set / ev_set no set | 40.2 / 35.4 / 39.0 / 37.8 (47.6 / 48.8 / 48.8 / 48.8) | 63.3 / 62.2 / 62.2 / 60.2 |
  | Head A | frozen mmropt | 39.0 (45.1) | 62.2 |
  | Head A | optset / ev / ev_set / ev_set no set | **42.7** / 40.2 / 40.2 / 40.2 (48.8 / 47.6 / 47.6 / 47.6) | 60.2 / 62.2 / 62.2 / 62.2 |

- Reading: learning "frames inside the human interval" does not improve the answers; all differences are
  within dev noise (±8–10 points at n = 82 / 98). The best CG-Bench number (Head A pool + optset, 42.7) does
  not hold on Video-MMMU (60.2). The human interval is a proxy for what the answerer needs; Head B stage 2
  learns from the answerer itself.
- Stage 2 data started (`scripts/v3_head_b_labels.py`): every candidate of every train pool answered alone
  (one frame, no transcript), keeping the full option-letter softmax (`Answer.option_probs`, new optional
  field) → per-frame utility = probability of the gold option. Ledger 5,000 calls / 3 GPU-hours.

**Step 42. Video-MMMU Comprehension dev track added** (2026-10-10)
- With 82 / 98 dev questions only differences of about 8–10 points are detectable, which is why the heads'
  localisation gains (steps 39–40) cannot be confirmed in accuracy. The Video-MMMU Comprehension track (98
  dev, 202 test questions, same videos and split) was already planned for the final test; it is now also a
  dev set (`videommmu_comp`), doubling the lecture dev questions to 196.
- Pools built (v2, frozen baseline, Head A). The first answer run was invalid: the start-up waiter launched
  it while the CPU-bound Head B training (step 43) used every core, and the answerer slowed to ~5 minutes
  per call (ledger `budget_comp.json`: 38 calls, 12,352 s). It is discarded and rerun alone on a fresh
  ledger (`budget_comp2.json`); answer jobs are no longer started next to CPU-heavy training.

**Step 43. Head B stage 2: the answerer's own judgement of each frame** (2026-10-10)
- Labels (`scripts/v3_head_b_labels.py`, ledger `labels_budget.json`): all 4,203 candidates of the 373
  CG-Bench train pools answered alone (one frame, no transcript); utility = the answerer's probability of
  the gold option.
- Scorer (`v3_head_b.py train_util`): the same 18 features, listwise KL towards softmax(utility / T), 3 seeds;
  T ∈ {0.05, 0.1, 0.2} chosen on inner validation (67 questions, 20% of train videos). Inner validation,
  gold probability of the top-ranked frame: random frame 0.276, top MobileCLIP (question) 0.298, top
  MobileCLIP (question + options, the frozen rule's first pick) 0.333, **learned 0.357** (T = 0.05; seeds
  0.361 / 0.350 / 0.359), best frame in the pool (oracle) 0.506.
- Selection: `util` (learned utility + redundancy, μ = 1) and `util_set` (+ option-evidence term, λ = 2,
  μ = 1, reused from step 41). Dev answers on all three dev sets are running (ledger `budget_headb2.json`).

**Step 44. Head B stage 2 results and the combined A + B grid on three dev sets** (2026-10-10)
- Dev accuracy (K = 4, no transcript); Video-MMMU Comprehension added (step 42). "base" = frozen baseline
  pool, "HA" = Head A pool.

  | Pool + selector | CG-Bench | VMMMU Perception | VMMMU Comprehension | pooled − frozen baseline (95% CI, n = 278) |
  |---|---|---|---|---|
  | v2 default (BM25-window pool, clip, 120 words) | 28.0 | 56.1 | 32.7 | −5.8 (−10.2..−1.0) |
  | **frozen baseline (base + mmropt)** | **40.2** | **63.3** | 31.6 | — |
  | base + optset / ev / ev_set | 40.2 / 35.4 / 39.0 | 63.3 / 62.2 / 62.2 | 27.6 / 32.7 / 32.7 | −1.4 / −1.4 / −0.4 |
  | base + util / util_set (answerer labels) | 39.0 / 39.0 | 53.1 / 52.0 | 32.7 / 32.7 | −3.6 (−7.6..+0.4) / −4.0 (−8.2..+0.0) |
  | HA + mmropt (Head A only) | 39.0 | 62.2 | 32.7 | −0.4 (−3.3..+2.9) |
  | HA + optset / ev / ev_set | 42.7 / 40.2 / 40.2 | 60.2 / 62.2 / 62.2 | 29.6 / 30.6 / 30.6 | −1.1 / −0.7 / −0.7 |
  | HA + util / util_set (A + B) | 41.5 / 40.2 | 55.1 / 55.1 | 34.7 / 34.7 | −1.4 / −1.8 |

  Comprehension with 120 transcript words: frozen pool 33.7 (vs 31.6 without), v2 32.7. Path A only on
  Comprehension: base 29.6, HA 32.7.
- Findings:
  - The tuned rule baseline is the strongest system on dev: +5.8 points over the v2 default pooled over 278
    questions (CI excludes 0).
  - No Head A / Head B variant or combination beats it; every pooled difference is within −4.0..−0.4 and its
    CI includes 0 (the answerer-utility selector is the worst, mainly on lectures: 52–55 vs 63 on Perception).
  - Both heads are trained on CG-Bench only (the only labelled training source with frames); what they learn
    helps on CG-Bench in places (HA + optset 42.7, HA + util 41.5 vs 40.2) and hurts on Video-MMMU lectures.
    The supervision does not cover the lecture domain, and the dev sets are too small to confirm the
    CG-Bench gains (+1 to +2.5 points).
  - Comprehension questions depend less on frames (all settings 28–35%) and slightly more on the transcript.
- Selection headroom remains: in the pool, the best single frame gives the answerer 0.51 gold probability
  vs 0.33 for the frozen rule's first pick (train inner validation), and frames inside the human evidence
  lift CG-Bench to 63.4% (gate) vs 40.2%.

**Step 45. Head B, training-free variant: answer-confidence set selection** (2026-10-10; ledger `budget_verify.json`)
- Motivation (train labels, step 43): the frozen answerer's confidence tracks correctness (single frames:
  accuracy 15.5% at confidence < 0.3, 45.5% at ≥ 0.9; the most confident frame per question is right 36.5%
  of the time vs 27.6% for an average frame). Uses no training labels, so it cannot overfit the CG-Bench
  domain.
- Method (`v3_baseline_cycle.py verify`): 4 candidate sets per question (frozen rule and `optset`, each on the
  frozen pool and the Head A pool); the answerer answers with each; "max" keeps the most confident answer,
  "sum" adds the option probabilities. Cost: 4 answer calls per question instead of 1. Re-answered requests
  reproduce the stored answers exactly (1,025 / 1,025).
- Dev (95% CI vs frozen baseline):

  | Rule | CG-Bench | VMMMU Perception | VMMMU Comprehension | pooled (n = 278) |
  |---|---|---|---|---|
  | max | 42.7 (+2.4, +0.0..+6.1) | **66.3** (+3.1, −3.1..+9.2) | 29.6 (−2.0) | +1.1 (−1.7..+3.8) |
  | sum | 41.5 (+1.2) | 61.2 (−2.0) | 31.6 (0.0) | −0.4 (−2.7..+1.9) |

- The first variant above the frozen baseline on the pooled mean, but not significant; the gain is on the two
  frame-dependent sets (CG-Bench + Perception: +2.8, −0.6..+6.5), not on Comprehension.

**Step 46. Six candidate sets instead of four** (2026-10-10)
- Adding the learned selections (base + ev_set, Head A + util) as sets 5 and 6: CG-Bench max 42.7 (same),
  Perception max 63.3 (−3.1 vs four sets); sum 40.2 / 61.2. More sets add more confidently wrong answers.
  Comprehension stopped at the ledger cap and was not extended (already worse on the other two sets).
  Four sets are kept.

**Step 47. Transcript coverage: how much relevant speech each policy keeps** (2026-10-10)
- Requirement: the answerer should not lose information the transcript holds. v2 gave a 120-word BM25
  excerpt; the frozen baseline gives none (it scored better in multiple choice, steps 34/36).
- New transcript policies in the runner (`transcript_for`, `--tpolicy`): `hybrid` (every line ranked by
  reciprocal-rank fusion of BM25 on question + options and zero-shot MiniLM relevance; best lines with one
  neighbour), `frames` (speech around the frames actually shown, nearest first, within 15 s),
  `frames+hybrid` (half each), `full` (whole transcript in time order up to the budget).
- Coverage on CG-Bench dev (82 questions; "included" = the text contains speech within 10 s of a human clue
  interval; median transcript 1,570 words; frames from the frozen baseline):

  | Policy | Budget | Included |
  |---|---|---|
  | BM25 (v2) | 120 / 300 | 28.0 / 50.0 |
  | hybrid | 120 / 300 / 600 | 50.0 / 62.2 / 62.2 |
  | frames | 120 / 300 | 46.3 / 48.8 |
  | frames + hybrid | 300 / 600 | 65.9 / 68.3 |
  | full | 1,500 / 4,000 / all | 65.9 / 79.3 / **80.5** (ceiling: 19.5% have no speech near the evidence) |

- Reading: v2's 120-word excerpt dropped the relevant speech for 72% of questions. Meaning-based fusion
  doubles coverage at the same budget, and the whole transcript (short enough for the answerer's context)
  loses nothing. Next, after the final test frees the GPU: answer accuracy (multiple choice, three dev sets)
  and open-ended correctness (CG-Bench) for these policies, to see which gives better answers.

**Step 48. Final frozen comparison on test** (2026-10-10; `docs/v3/final_test_results.md`)
- Run once as declared (9 systems × 459 test questions; 5,370 calls). Pooled accuracy: question only 22.2;
  evenly spaced 36.4; dense MobileCLIP 36.8; v2 default 38.1; **tuned baseline 41.4**; + Head A 39.7;
  + Head B 41.4; Head A + B 41.4; answer-confidence selection 42.3.
- Tuned baseline − v2: CG-Bench **+16.4 (+7.0..+25.9)**, pooled +3.3 (−0.7..+7.0). Tuned baseline vs dense
  MobileCLIP **+4.6 (+1.3..+7.8)**, vs evenly spaced **+5.0 (+1.1..+8.9)**. Heads vs tuned baseline: all CIs
  include 0.
- The declared open-ended check on CG-Bench test (S0, S1, R3) runs after the transcript-policy trials
  (step 49), which were requested next and are already queued on the GPU.

**Step 49. Transcript policies: multiple-choice accuracy on dev** (2026-10-10; ledger `budget_text.json`)
- Frozen baseline frames (hybrid pool, mmropt, K = 4); only the transcript given to the answerer changes.

  | Transcript | CG-Bench | VMMMU Perception | VMMMU Comprehension | mean |
  |---|---|---|---|---|
  | none (frozen baseline) | **40.2** | **63.3** | 31.6 | **45.0** |
  | BM25 120 (v2) | 35.4 | 58.2 | 33.7 | 42.4 |
  | BM25 300 | 34.1 | 59.2 | — | — |
  | hybrid 120 / 300 | 35.4 / 37.8 | 58.2 / 62.2 | 32.7 / 33.7 | 42.1 / 44.6 |
  | frames + hybrid 300 / 600 | 36.6 / 37.8 | **63.3** / 60.2 | 32.7 / **34.7** | 44.2 / 44.2 |

- The uncapped full transcript ran out of GPU memory: lines carry ids and times, so a 7,191-word transcript is
  ~20k tokens and attention with 4 images exceeds 16 GB. Re-run capped at 1,500 and 2,500 words (2,500 covers
  the whole transcript of most videos: medians 1,570 CG-Bench, 1,194 Video-MMMU).
- Reading so far: better retrieval of the speech recovers most of what the BM25 excerpt cost (+2.2 mean at 300
  words) and helps the comprehension questions (+3.1 over no text with frames + hybrid 600), but in multiple
  choice the 2B answerer still does best on frame-dependent sets without text. Open-ended answers (the primary
  objective) are measured next for none, hybrid 300, frames + hybrid 600 and full 2,500.

**Step 50. Transcript policy chosen by open-ended correctness (CG-Bench)** (2026-10-11)
- Capped full transcript, multiple choice: 1,500 words CG-Bench 40.2 / Perception 60.2 / Comprehension 33.7
  (mean 44.7, vs 45.0 without text); 2,500 words runs out of GPU memory on the longest CG-Bench transcripts
  (Perception 61.2, Comprehension 32.7).
- Open-ended correctness, CG-Bench dev (82; judge as in step 36; 4 of 663 verdicts failed after retries and
  are excluded pairwise):

  | Transcript | Open-ended | MC | speech near evidence kept |
  |---|---|---|---|
  | v2 default (BM25 pool, clip, BM25 120) | 13.4 | 28.0 | 28.0 |
  | frozen frames + BM25 120 | 17.1 | 35.4 | 28.0 |
  | frozen frames, no text | 19.5 | **40.2** | 0 |
  | frozen frames + full 1,500 | 19.2 | **40.2** | 65.9 |
  | frozen frames + hybrid 300 | 20.7 | 37.8 | 62.2 |
  | **frozen frames + frames + hybrid 600** | **22.0** | 37.8 | **68.3** |

  Paired (10,000 resamples over videos): frames + hybrid 600 − v2 **+8.5 (+3.5..+14.7)**; − BM25 120
  **+4.9 (+1.3..+8.3)**; − no text +2.4 (−2.2..+6.9). Hybrid 300 − no text +1.2 (−6.0..+8.1); full 1,500 − no
  text +0.0.
- Decision (the requirement: better answers and less lost text): the framework's transcript policy becomes
  **frames + hybrid, 600 words**: speech around the frames shown plus the best lines by BM25 + meaning fusion.
  It gives the best open-ended answers, keeps 68% of the evidence speech (BM25 120: 28%), and costs 0.8
  points of multiple-choice mean on dev (within noise). This changes the framework after the declared test;
  its test numbers will be reported separately as a follow-up, not as part of the declared comparison.
- Declared open-ended check on CG-Bench test (55): v2 18.2, tuned framework (no text) 23.6, dense MobileCLIP
  25.5; tuned − v2 +5.5 (−7.3..+20.8), tuned − dense −1.8 (−8.9..+3.3): 6 test videos are too few to separate
  these.

**Step 51. Declared: does the framework's gain hold with a stronger answerer?** (2026-10-11)
- The 2B answerer caps every number (frames inside the human evidence: 63.4% on CG-Bench dev). Same pools and
  selections, answerer swapped to Qwen3-VL-4B-Instruct (revision ebb281e, `configs/gpu_16gb_4b.yaml`):
  v2 default, tuned framework (no text), tuned framework + frames + hybrid 600 (step 50), and on test also
  dense MobileCLIP top-4 and evenly spaced frames. Dev and test, multiple choice. Ledger `budget_4b.json`
  (5,000 calls / 5 GPU-hours). Not used to choose any setting (all settings were fixed on 2B dev).
- Also the 2B test follow-up for the step-50 transcript policy (reported separately from the declared test).

**Step 52. Stronger answerer (4B): partial results; second ledger** (2026-10-11)
- The 4B answerer needs ~11 s per call (2B: ~1.3 s), so the 5-GPU-hour ledger (`budget_4b.json`) ended after
  1,653 calls. Done: all dev runs, CG-Bench test (5 systems), Video-MMMU Perception test (v2, tuned; tuned +
  frames + hybrid 600 for 144 / 202). Missing: the rest of Perception test and Comprehension test.

  | 4B answerer | v2 default | tuned, no text | tuned + frames + hybrid 600 |
  |---|---|---|---|
  | CG-Bench dev | 36.6 | 36.6 | **40.2** |
  | VMMMU Perception dev | 62.2 | 65.3 | **68.4** |
  | VMMMU Comprehension dev | 37.8 | **39.8** | 38.8 |
  | CG-Bench test | 38.2 | 41.8 | **49.1** (dense MobileCLIP 40.0, evenly spaced 27.3) |
  | VMMMU Perception test | 60.9 | 62.9 | 68.1 (n = 144) |

- 2B follow-up of the step-50 policy on test (not part of the declared comparison): CG-Bench 49.1,
  Perception 49.0, Comprehension 31.7 (tuned without text: 50.9 / 48.0 / 32.2).
- Reading: a stronger answerer uses the retrieved speech: with 4B, frames + hybrid 600 is the best setting on
  4 of 5 sets, while with 2B it is roughly equal to no text. The selected speech carries information the
  stronger model can read.
- Second 4B ledger declared (`budget_4b_2.json`, same limits) to finish Perception test (tuned + text) and
  Comprehension test (v2, tuned, tuned + text) and the external rules on both Video-MMMU test sets.

**Step 53. 4B results complete; summary of the follow-ups** (2026-10-11; `docs/v3/final_test_results.md`)
- Second 4B ledger (`budget_4b_2.json`): 852 calls; it ran 7,323 s past its time limit on the last block
  before stopping (the check is made before each call), and the external-rule runs on Video-MMMU test were
  left incomplete rather than extended.
- Test, pooled 459 questions: 4B v2 50.3, tuned 52.7, **tuned + speech 55.6**; tuned + speech − v2
  **+5.2 (+1.0..+9.5)**, − tuned **+2.8 (+0.2..+5.6)**. 2B: 38.1 / 41.4 / 41.4.
- Conclusion of the transcript work: keeping the important speech (frames + hybrid, 600 words) loses no
  accuracy with 2B and adds a significant gain with 4B; it is the framework's transcript policy from here on.

**Step 54. Declared: the full combination grid** (2026-10-17)
- Not yet run: the heads together with the speech text and with the 4B answerer. The grid now covers every
  combination of answer model {2B, 4B} × speech text {none, frames + hybrid 600 words} × {best framework,
  + Head A, + Head B (ev_set), + Head A + Head B}; the 7 missing combinations run on all dev and test sets
  (`scripts/run_full_grid.sh`, ledgers `budget_grid_2b.json`, `budget_grid_4b.json`, 8,000 calls / 24 h
  each). No setting changes; test numbers are follow-ups, reported apart from the declared comparison.

**Step 55. Full combination grid: results** (2026-10-19; ledgers `budget_grid_2b.json` 2,162 calls,
`budget_grid_4b.json` 4,322 calls / 21.8 h)
- Accuracy, 4 frames. dev = CG-Bench 82 / Perception 98 / Comprehension 98; test = 55 / 202 / 202.

  | Answer model, speech text | Version | dev CG / Perc / Comp | test CG / Perc / Comp | test all 459 |
  |---|---|---|---|---|
  | 2B, none | best / +A / +B / +A+B | 40.2 63.3 31.6 / 39.0 62.2 32.7 / 39.0 62.2 32.7 / 40.2 62.2 30.6 | 50.9 48.0 32.2 / 41.8 45.0 33.7 / 47.3 48.0 33.2 / 34.5 52.0 32.7 | 41.4 / 39.7 / 41.4 / 41.4 |
  | 2B, 600 words | best / +A / +B / +A+B | 37.8 60.2 34.7 / 37.8 58.2 29.6 / 37.8 62.2 29.6 / 36.6 60.2 30.6 | 49.1 49.0 31.7 / 40.0 48.0 33.2 / 40.0 47.5 34.7 / 34.5 51.0 32.7 | 41.4 / 40.5 / 41.0 / 41.0 |
  | 4B, none | best / +A / +B / +A+B | 36.6 65.3 39.8 / 45.1 65.3 36.7 / 40.2 64.3 43.9 / 47.6 66.3 41.8 | 41.8 62.9 45.5 / 41.8 65.8 45.5 / 38.2 64.4 48.5 / 34.5 66.3 49.0 | 52.7 / 54.0 / 54.2 / 54.9 |
  | 4B, 600 words | best / +A / +B / +A+B | 40.2 68.4 38.8 / 51.2 65.3 37.8 / 46.3 67.3 38.8 / 50.0 70.4 40.8 | 49.1 67.8 45.0 / 47.3 66.8 47.0 / 43.6 64.9 45.5 / 45.5 66.8 45.5 | 55.6 / 55.8 / 53.8 / 54.9 |

- Head versions − best framework (95% CI, bootstrap over videos):
  - 4B, no speech text, Head A + Head B: dev +4.3 (+0.0..+8.4), test +2.2 (−1.3..+5.9), lecture test +3.5
    (−0.2..+7.2); **dev and test together (737) +3.0 (+0.2..+5.7)**. Head A alone +1.4, Head B alone +1.8 (both
    positive on dev and test, not significant).
  - 4B, 600 words: Head A + Head B dev **+4.3 (+0.4..+8.2)**, test −0.7 (−4.1..+2.9); dev and test together +1.2
    (−1.5..+3.9). Head A test +0.2, Head B test −1.7.
  - 2B (both text settings): every head difference between −1.7 and +2.2, none significant.
- Reading: the heads' better localisation starts to pay off with the stronger answerer. With 4B and no speech
  text, all three head versions are above the best framework on dev and test, and Head A + Head B is significant
  over the 737 dev + test questions (a pooled, after-the-fact view; the test set alone does not confirm it). When
  the 600-word speech text is given, the speech already carries much of what the heads add, and the gain mostly
  disappears on test. The 2B answerer cannot use the heads' frames.
- Highest test accuracy overall: 4B + 600 words + Head A 55.8 vs 55.6 for the best framework (not
  significant). The recommended system stays the best framework + 600 words + 4B; Head A + Head B is the best
  choice when no speech text is given to the 4B answerer.

**Step 56. Declared: ceiling with human-marked pictures for both answer models** (2026-10-21)
- The 63.4% ceiling (step 3 gate) was measured only for the 2B model, only on dev, and with a 300-word BM25 excerpt.
  `scripts/v3_ceiling.py` measures, on CG-Bench dev (82) and test (55), with the 2B and the 4B answer model: 4
  pictures spread over the human clue intervals with no transcript words (H0), and with the framework's 600-word
  speech text (H600). Reference only (uses gold locations). Ledger runs/v3_ceiling/budget.json, 1,000 calls / 4 h.

**Step 57. Ceiling with human-marked pictures: results** (2026-10-21; `reports/v3_ceiling/summary.json`)
- CG-Bench, 4 pictures spread over the human clue intervals; accuracy (questions):

  | Answer model | dev, no words (82) | dev, 600-word speech text (82) | test, no words (55) | test, 600-word speech text (55) |
  |---|---|---|---|---|
  | Qwen3-VL-2B | 58.5 | 59.8 | 61.8 | 58.2 |
  | Qwen3-VL-4B | 57.3 | 53.7 | 50.9 | 56.4 |

- Median answer time per call: 2B 0.5 s without words, 1.8 s with 600 words; 4B 2.2 s without words, 27.8 s with
  600 words.
- Reading: on long everyday videos the larger model is not better even with perfect pictures; the earlier 63.4%
  (step 3) included a 300-word BM25 excerpt and is not the same setup. Gap between the frameworks and this ceiling
  on CG-Bench test: 2B best framework 50.9 vs 61.8; 4B best framework + speech text 49.1 vs 56.4.

**Step 58. Head B v2: diagnosis and declared changes (dev only)** (2026-10-24)
- Diagnosis from steps 41–55: (1) Head B was trained on pools built with the speech rule but deployed on
  Head A pools (train/deploy mismatch); (2) it concentrates the 4 frames on one moment (86–89% speech-path frames
  and 1.6–2.4 min time spread on lectures, step 54 analysis), which helps lecture screen-reading and hurts long
  videos; (3) it was trained on 373 CG-Bench questions only. Related work: VSI (arXiv 2508.06869, training-free
  subtitle + visual keyframe search), routing of frame-selection strategies by question type (arXiv 2609.12517),
  adaptive greedy coverage selection (arXiv 2603.20180).
- Changes, each judged on dev only:
  - cross-fitted Head A pools for the training videos (`hybrid_ahead_cf`): Head A retrained in 4 video folds × 3
    seeds (`v3_head_a_light.py --folds 4 --fold k`), each training video's moments come from copies that never saw
    it; fold models reach dev evidence hit 0.34–0.39 (full Head A 0.366). Head B retrained on these pools;
  - set constraints in the greedy selection: minimum time gap between chosen frames (`--gap`) and a cap on
    speech-path frames (`--a-quota`); relaxed only when no candidate satisfies them.

**Step 59. Large answer model through the NVIDIA API; fresh confirmation set** (2026-10-24)
- `src/videoqa/answerer_api.py`, `configs/api_nemotron_vl.yaml`: Nemotron-3.5-Super-VL (120B parameters, ~12B
  active), same prompt as the local answer models, 2,048-token budget (it reasons first; 512 tokens cut the
  reasoning), option probabilities from the returned log-probabilities of the answer letter. Smoke test on 3
  CG-Bench dev questions: 2 correct; reported confidence is near 1.0, so confidence-based selection has little to
  work with for this model.
- Fresh confirmation set (`scripts/research/fetch_lvb_fresh.py`): every LongVideoBench validation question on a
  video not used by v1/v2: 501 videos, 897 questions, 19.5 GB; never used for any decision.
- Repository cleaned: generated PDF/Excel reports untracked (rebuilt by `scripts/build_*.py`), history notes
  moved to `docs/history/`, README rewritten.

**Step 60. Head B v2 dev screen** (2026-10-24; ledgers `budget_headb_v2*.json`)
- Head B v2 = ev_set retrained on the cross-fitted Head A training pools (3 seeds; inner validation recall@4
  0.76 / 0.76 / 0.81), deployed on Head A pools. CG-Bench dev evidence hit: cap 4 / no gap 48.8, cap 2 52.4 (v1:
  47.6); gaps of 15–60 s lower it to 45–48.
- Dev accuracy, 278 questions, no speech text (vs best framework; vs Head A + Head B v1):

  | Setting | 2B | 4B |
  |---|---|---|
  | best framework | 45.3 | 47.8 |
  | Head A + Head B v1 | 44.6 | **52.2** (+4.3, −0.3..+8.4) |
  | v2, cap 4, no gap | 45.0 | 48.6 (vs v1 −3.6, −6.5..−0.8) |
  | v2, cap 2, no gap | 43.9 | 47.8 (vs v1 −4.3, −7.8..−0.7) |
  | v2, cap 2, 30 s gap | 42.8 | 49.3 |
  | v2, cap 2, 60 s gap | 42.4 | 51.1 (vs v1 −1.1, −4.3..+2.1) |

- Reading: training on deployment-matched pools did not help (v2 below v1); the time gap and speech-path cap
  help v2 with 4B (48.6 → 51.1). Next: the same constraints on Head B v1 (dev), then the choice of the Head B used
  in the confirmation run.

**Step 61. Constraints on Head B v1; Head B choice for confirmation** (2026-10-24)
- Head A + Head B v1 with the step-58 constraints, dev 278, no speech text: 4B 60 s gap + cap 2 50.0, 30 s gap
  50.4, 60 s gap 50.0 (unconstrained v1 52.2); 2B 41.7–43.5 (unconstrained 44.6). The constraints do not help v1.
- Decision (dev): the confirmation run keeps Head B v1 (ev_set, λ 2, μ 1, trained on the speech-rule training
  pools, no constraints). Head B v2 and the constraints are reported as tried and not adopted.

**Step 62. Confirmation protocol declared** (2026-10-24; `docs/v3/confirmation_protocol.md`)
- Fresh LongVideoBench set (501 videos, 897 questions) with 2B, 4B and Nemotron-120B; systems S0, S1, S1t, S2,
  S3, S4, S4t, R2, R3; primary hypotheses H1 (heads, 4B), H2 (heads, Nemotron), H3 and H4 (framework vs v2 design,
  4B and Nemotron) with Holm correction; settings frozen at step 61. Drivers `scripts/run_confirmation.sh`
  (local) and `scripts/run_confirmation_api.sh` (API, plus the follow-up on the existing test sets).

**Step 63. Large-model ceiling: first run, a reply problem, and the fix** (2026-10-24)
- First ceiling run with Nemotron-120B (human-marked pictures, CG-Bench): dev 58.5 without words, 59.8 with the
  600-word speech text (82 questions; the same totals as 2B by coincidence: the two models disagree on 37–42 of
  82 questions); test stopped at 24 of 55 questions when the ceiling ledger (1,000 calls) was used up.
- 28 of 212 replies (13%) carried no option letter (scored as wrong). Re-asking the same requests returned valid
  letters, so these are replies whose reasoning ended without a final answer, not wrong answers. Fix
  (`answerer_api.py`): when a multiple-choice reply has no letter, ask once more with twice the token budget. The
  first run's answers are kept in `runs/v3_ceiling/answers_nemotron_first_run.jsonl`; the ceiling is re-run with
  the fix on a separate ledger (`budget_api.json`). The confirmation run's API part had not started and uses the
  fixed answerer.

**Step 64. Large-model ceiling with the reply fix** (2026-10-25; ledger `runs/v3_ceiling/budget_api.json`)
- Nemotron-120B with 4 pictures from the human-marked answer seconds (CG-Bench):

  | | dev, no words (82) | dev, 600-word speech text (82) | test, no words (55) | test, 600-word speech text (55) |
  |---|---|---|---|---|
  | Nemotron-3.5-Super-VL 120B | 59.8 | 63.4 | 65.5 | 69.1 |
  | Qwen3-VL-4B (step 57) | 57.3 | 53.7 | 50.9 | 56.4 |
  | Qwen3-VL-2B (step 57) | 58.5 | 59.8 | 61.8 | 58.2 |

- Replies without an option letter: 11 of 274 (4%) after the fix (13% before). The large model is the only one
  whose ceiling clearly gains from the speech text and from perfect pictures on test (69.1).

**Step 65. Confirmation run: memory failure and restart** (2026-10-25)
- First pass: all 2B systems without speech text finished on the 894 fresh questions; the 2B speech-text runs
  stopped at 658; with 4B only the two reference rules finished; the API run reached 449 answers at ~33 s each.
  Cause: the answer stage held every pool with its images in memory, and with the API run in parallel the machine
  ran out of RAM ("paging file too small") and GPU memory.
- Fix: pools are loaded one question at a time; API answer models are called by 4 threads (paced to the free
  tier); query embeddings for API runs on the CPU. Re-running a finished system reproduces it exactly (0 new calls,
  same accuracy). Both drivers restarted; finished answers come from the cache. No setting changed.

**Step 66. Oracle ceiling declared; experiments index; audit loop** (2026-10-25)
- The ceiling so far places the 4 pictures evenly inside the human-marked seconds; it does not search for the
  placement that helps the answer model most. Oracle ceiling (`v3_ceiling.py --oracle`): 5 placements (shifted
  grids) inside the marked seconds, a question counts as answerable if any placement is answered correctly; 2B,
  4B and Nemotron-120B, with and without the speech text, CG-Bench dev and test (`scripts/run_oracle_ceiling.sh`),
  queued after the confirmation runs.
- `docs/EXPERIMENTS.md`: index of all experiments (question, script, result file, outcome) and how to run the
  best framework.
- An external auditor reviews claims, novelty and benchmarks through a local log; its notes and the replies are
  summarised here when they change a decision.

**Step 67. Audit corrections: bootstrap clusters, wording, memory skip** (2026-10-30)
- The pooled test interval treated the two Video-MMMU tracks of the same lecture as different videos (410
  clusters instead of the 208 real source videos), which makes intervals too narrow. Both tracks now share one
  cluster per source video, and every interval uses the declared 10,000 resamples. Recomputed from cached answers
  (no model calls), 459 test questions:

  | Comparison | Difference (points) | Interval, 410 clusters | Interval, 208 clusters | Verdict |
  |---|---|---|---|---|
  | 4B tuned + speech text minus 4B starting framework | +5.2 | +1.0 to +9.4 | +1.1 to +9.4 | confirmed |
  | 4B tuned + speech text minus 4B tuned without text | +2.8 | +0.2 to +5.6 | 0.0 to +5.6 | not confirmed (borderline) |
  | 2B tuned minus 2B starting framework | +3.3 | −0.7 to +7.0 | −0.4 to +6.8 | not confirmed |
  | 4B tuned minus 4B starting framework | +2.4 | −1.5 to +6.4 | −1.6 to +6.3 | not confirmed |

- Wording fixed in the README: the 4B and speech-text rows are follow-ups after the original frozen 2B
  comparison; Head A's 36.6% is the share of 164 CG-Bench dev questions where one of its 6 proposed moments falls on
  the marked moment; the speech figure is the share of questions with a chosen line within 10 s of the marked
  moment, at different word budgets.
- On the fresh set both 4B speech-text systems stopped at 839 of 894: one music video (TMe7oXMJoSM_0) has its 600
  words in 261 short segments, each with a timestamp, and the prompt runs out of GPU memory with 4B. The answer
  stage now records such a question in `runs/v3_baseline/skipped.jsonl` and continues; skipped questions are
  reported as exclusions, and paired comparisons use identical question sets.
- The results table (`docs/report/all_results.csv`, now 557 rows with the Nemotron runs on the test sets) was
  rebuilt with the corrected clusters and 10,000 resamples. 10 "confirmed?" cells changed, all near the edge of
  their interval (for example 2B speech text minus starting framework on CG-Bench dev, gain to not confirmed; on
  CG-Bench test, not confirmed to gain; Head A + Head B minus starting framework on both Video-MMMU tracks together,
  not confirmed to gain). The 4B headline (+5.2 over all 459) is unchanged.

**Step 68. Second audit: answer-model labels, change list, summary wording** (2026-10-30)
- The results table labelled every run without the 4B tag as Qwen3-VL-2B, so the 17 Nemotron-120B rows (11
  accuracy, 6 CG-Bench evidence hit) were called 2B and compared with the 2B starting framework. Example: CG-Bench
  test best framework with Nemotron, 33/55, was reported +25.5 (+13.6..+42.9, confirmed) because it subtracted 2B's
  19/55. Fix (`scripts/build_results_table.py`): the answer model comes from the run's tag (2B, 4B or Nemotron),
  every reference uses the same answer model, and Nemotron rows have their own stage 12 ("Large answer model,
  API follow-up") instead of "Final locked test". Rebuilt from cached answers (no model calls), Nemotron against
  Nemotron:

  | Nemotron-120B, test | Correct | vs starting framework (same model) |
  |---|---|---|
  | CG-Bench, best framework | 33 / 55 | +10.9 (−3.4..+27.1), not confirmed |
  | CG-Bench, best framework + speech text | 35 / 55 | +14.5 (+6.7..+24.5), confirmed |
  | CG-Bench, Head A + Head B + speech text | 32 / 55 | +9.1 (+3.3..+16.3), confirmed |
  | Video-MMMU Perception, best framework + speech text | 89 / 103 | +7.8 (−2.0..+17.6), not confirmed |

  The Nemotron runs on the old test sets are incomplete (50–110 of the questions per system; Video-MMMU
  Comprehension not yet run), and each comparison uses only the questions both systems answered; they are a
  diagnostic, not part of the local claim. No 2B or 4B row changed. The table now has 562 rows: 17 relabelled rows
  plus 5 rows from answers saved after the step 67 rebuild.
- Change list for step 67, by comparing the committed tables before and after commit 3de2078 (489 and 557 rows):
  **5** "confirmed?" cells changed, not 10 as written in step 67, and the CG-Bench test example given there was
  wrong. The five: 2B Video-MMMU Perception dev tuning variant (3 inside windows + 3 dense) vs best, loss to not
  confirmed; 2B CG-Bench dev best framework + 300-word speech text vs best, not confirmed to loss; 2B CG-Bench dev
  best framework + 600-word speech text vs starting framework, gain to not confirmed; 2B CG-Bench dev 300-word best
  lines vs starting framework, gain to not confirmed; 2B Head A + Head B on both Video-MMMU tracks vs starting
  framework, not confirmed to gain.
- One canonical source for published intervals: `scripts/build_results_table.py` (seed 20261019, 10,000
  resamples, 208 source-video clusters for the 459 test questions). With it, 4B speech text minus 4B without text
  is +2.8 (+0.0..+5.5), not confirmed; 2B +0.0 (−2.5..+2.6).
- Summaries brought in line: `docs/v3/final_test_results.md` (corrected intervals; +2.8 no longer called
  significant; "68% of evidence speech kept" replaced by the measured share at different word budgets; "beats
  standard frame-selection rules overall" withdrawn with a note), `docs/EXPERIMENTS.md` (Head A figure is a dev
  result on 164 questions; +16.4 on only 6 videos; "equal to dense MobileCLIP" replaced by "no demonstrated
  advantage").
- Two implementation facts found by the audit and checked: (1) pools keep a 300-word BM25 excerpt, so a BM25 policy
  above 300 words would be cut to 300; earlier BM25 results used 120 or 300 words and are not affected, but the
  planned matched-budget speech comparison must build every arm from the full transcript and record words, tokens
  and lines actually given. (2) Head B's `speech_z` feature is the frozen speech scorer's relevance of the line
  nearest the candidate, not Head A's learned relevance (`LearnedMoments.score` forwards the frozen scorer). It is
  now described that way; whether Head A's relevance would be a better feature is untested.

**Step 69. Target decided: CVPR 2027 main track; plan to the deadline; runs restarted** (2026-10-30)
- Owner's decision: submit to CVPR 2027 (paper registration 10 Nov, full paper 16 Nov 2026, AoE). The auditor's
  view (likely reject without lecture evidence) is recorded; the plan below aims at the missing piece, a
  question-conditioned frame-selection result on unseen lectures against strong matched baselines.
- Plan (dates are targets):
  1. 30 Oct – 1 Nov: finish the confirmation run and the oracle ceiling; H1–H4 table with Holm correction on the
     fixed 894 questions; per-question loss trace on CG-Bench dev (window → pool → chosen → answer).
  2. 31 Oct – 4 Nov: lecture pilot on M3AV (speaker-disjoint official split, bounded subset, cached frozen
     embeddings): train Head A on training speakers, measure on unseen speakers whether the chosen frames show the
     slide that the speech refers to.
  3. 4 – 8 Nov: matched comparisons on lectures and the existing sets (dense option-aware retrieval, a VSI-style
     subtitle + visual search, Head A without its learned residual, Head B without the option term, matched-budget
     speech text), same K, questions and answer model.
  4. 8 – 10 Nov: freeze the claims; register the paper. 10 – 16 Nov: write, figures, supplementary.
- Out-of-memory failures (P26): a question whose prompt runs out of GPU memory is now an answer that failed:
  cached under its full request key (never retried), charged to the ledger and scored as wrong
  (`scripts/v3_baseline_cycle.py`, row field `failed`). `runs/v3_baseline/skipped.jsonl` is kept as the record of
  the earlier skips. The primary confirmation analysis uses all 894 questions; a common-success analysis is
  secondary.
- Oracle ledger (P27): `budget_oracle_4b.json` shows 604 calls and 47,094 s against an 8-hour limit. The answer
  log shows one call of 34,045 s (median call 4.6 s): the machine hung or slept during one call; the ledger checks
  its limit before a call, so that one call overshot it. The other 603 calls took 13,049 s. The oracle continues on
  a new ledger with the same limits (`budget_oracle_4b_cont.json`), and its driver now stops at the first failing
  step, so "DONE" is printed only after every step succeeded. The API confirmation run had stopped with Windows
  process-creation errors (same period).
- Restarted: local chain (4B oracle, then the remaining local confirmation answers) and API chain (Nemotron
  confirmation, then Nemotron oracle). Finished answers come from the cache.

**Step 70. Confirmation run: local part complete; H1–H3 final** (2026-10-31; `reports/v3_confirmation/summary.md`)
- `scripts/v3_confirmation_report.py`: the declared analysis (two-sided paired bootstrap over videos, 10,000
  resamples, Holm over H1–H4) on the fixed 894 eligible questions (500 videos); a failed answer counts as wrong; a
  comparison is final only when both systems answered all 894. Sensitivity: the questions both answered without a
  failure. All 2B and 4B systems are complete; one 4B speech-text question (TMe7oXMJoSM_0) ran out of GPU memory
  in each speech-text system and is scored wrong.

  | Hypothesis | Difference (95% interval) | p | Status |
  |---|---|---|---|
  | H1: Head A + Head B − best framework, 4B | −0.2 (−3.0..+2.7) | 0.92 | final: not supported |
  | H2: Head A + Head B − best framework, Nemotron-120B | +2.1 (−0.8..+5.1) | 0.17 | final: not supported |
  | H3: best framework + speech text − starting design, 4B | **+8.9 (+5.7..+12.2)** | <0.0001 | final: supported (Holm <0.0001) |
  | H4: same, Nemotron-120B | +5.3 (+0.0..+10.4) on 414 questions | 0.05 | pending (Nemotron speech-text run at 414/894) |

- Secondary, 4B: Head A alone +1.2 (−1.5..+3.9); Head B alone −0.3; speech text on Head A + B −1.6; best framework
  vs dense MobileCLIP top-4 −0.1 (−2.9..+2.7); vs 4 evenly spaced pictures +8.6 (+5.1..+12.2). Nemotron: Head A
  alone +2.8 (+0.0..+5.6, p 0.05).
- Reading: on new data the framework's gain over the starting design is confirmed with 4B (H3), but it comes from
  searching the whole video by picture: plain dense MobileCLIP top-4 is as good (no demonstrated difference). The
  trained heads add no confirmed accuracy with either answer model (H1, H2). This matches the auditor's view that
  the publishable contribution must come from the lecture evidence, not from the current heads on general videos.
- 4B oracle ceiling complete (CG-Bench, any of 5 placements inside the marked seconds): dev 72.0 / 72.0 (no text /
  600-word speech text), test 61.8 / 67.3; continuation ledger 492 calls, 15,053 s.

**Step 71. Lecture pilot on M3AV: data checked, design declared before any question exists** (2026-10-31)
- M3AV v1.0 (Chen et al., arXiv 2403.14168; CC BY-NC-SA 4.0, video rights with the owners): no-audio archive
  (2.02 GB, SHA-256 matches the official list) and the official split by speaker (`set_split.json`): 901 train /
  99 dev / 113 test lectures (CHI, NIH, Ubicomp, IPP, MLS talks). Per lecture: slide segments with start and end
  time, one image per segment, its OCR text and the speech said during it (`ocr/seg_speech_ocr.json`), word-level
  transcript (`speech/final+timestamps.json`), YouTube link. Train 19,883 slide segments (275.6 h), dev 2,014
  (32.1 h), test 2,341 (34.3 h); median lecture 5 minutes, 14–15 slides. Sample test videos are still online.
- Why this fills the gap: Video-MMMU has no marked evidence moments, so it cannot show that the right frames
  were chosen. M3AV gives, for each slide, the seconds during which it is shown.
- Declared design (fixed before any question is generated):
  1. Questions: for each dev and test lecture, the large model (Nemotron-120B, API) writes multiple-choice
     questions (4 options) about the content of one slide (its image and OCR), answerable only by seeing that
     slide. Evidence moment = that slide's [start, end] seconds. The question generator never sees the other
     slides' times, and the systems never see the evidence moment.
  2. Filters, applied the same way to every question: drop a question that Qwen3-VL-4B answers correctly from the
     question and options alone (no pictures, no transcript), and duplicates. A random sample of 100 test
     questions is checked by a person (correct answer key; the slide is needed) before any system is run on test.
  3. Videos: downloaded for dev and test lectures only (360p is enough for 4 frames of slides).
  4. Measures: primary = evidence hit (one of the 4 shown frames lies inside the evidence moment), per question,
     paired bootstrap over lectures; secondary = multiple-choice accuracy with Qwen3-VL-4B.
  5. Systems: starting design, best framework (± speech text), Head A as trained on CG-Bench, Head A retrained on
     M3AV train lectures (speech → slide moments, speaker-disjoint), Head B, 4 evenly spaced, dense MobileCLIP
     top-4, dense option-aware top-4, and a VSI-style subtitle + visual search; all with K = 4 and the same answer
     model. Settings are chosen on dev; test is run once.
- Shortcut risk noted: a question that quotes the speech could be found by transcript search alone; this is a
  legitimate route for the speech path, but the share of questions whose evidence is found by transcript search
  alone is reported separately.
- Videos (2026-10-31): 211 of the 212 dev and test lecture videos downloaded (360p). The one missing video is dev
  lecture CHI-36F5D (YouTube tIEjZFlPwcw): YouTube asks for a signed-in account, so it is dropped from dev and its
  questions are not generated. Test is complete (113 of 113 lectures).

## Step 72 (2026-10-31): confirmation on fresh LongVideoBench questions, all four declared hypotheses final
894 eligible questions on 500 videos; paired bootstrap over videos, 10,000 resamples, seed 20261024; Holm over
H1-H4 (`scripts/v3_confirmation_report.py`, `reports/v3_confirmation/summary.md`). A failed answer (GPU out of
memory) counts as wrong; one question failed in each 4B speech-text system.

| Hypothesis | Difference (95% interval) | p (Holm) | Result |
|---|---|---|---|
| H1: heads (S4) − best framework (S1), Qwen3-VL-4B | −0.2 (−3.0..+2.7) | 0.92 | not supported |
| H2: heads (S4) − best framework (S1), Nemotron-120B | +2.1 (−0.8..+5.1) | 0.34 | not supported |
| H3: best framework + speech text (S1t) − starting framework (S0), 4B | +8.9 (+5.7..+12.2) | ≈0.0008 (≈0.0002 before correction; approximate bootstrap-tail estimates) | supported |
| H4: S1t − S0, Nemotron-120B | +8.3 (+4.8..+11.8) | ≈0.0008 (≈0.0002 before correction; approximate bootstrap-tail estimates) | supported |

- Reading: the improved pipeline (whole-video visual search matched to question and options, plus selected
  speech text) beats the starting design on unseen questions with both a 4B and a 120B answer model. The learned
  heads do not add accuracy over the best framework with either model. *(Later reading, step 103: the heads show no
  confirmed increment within the tested contrasts; this does not show they have no effect.)*
- Secondary (no correction): the best framework is no better than dense MobileCLIP top-4 (2B −0.8, 4B −0.1;
  Nemotron pending, −0.7 on 436 questions) and clearly better than 4 evenly spaced frames (2B +7.5, 4B +8.6).
  Head A alone with Nemotron: +2.8 (+0.0..+5.6), p 0.052, not significant.
- So the paper cannot claim that frame selection beats a strong training-free rule; the confirmed result is the
  pipeline against the starting design. This is why the lecture evidence (M3AV, step 71) comes next.
- Pending: the Nemotron runs of the two standard rules (R2, R3), still filling in under the API rate limit.
- Human check of the 100 M3AV test questions (step 71, item 2): the owner decided that the builder and the GPT
  auditor check them independently instead of a person; a question is kept only if both confirm its answer key
  from the slide. This is a protocol change from "checked by a person" and is reported as such.
- Update (same day): a person will also check the 100 test questions, after the project's runs are finished. To
  keep this honest, the test questions are frozen (file hash recorded) before any system is run on test, results
  are reported on all frozen questions, and the person's check is reported as a second analysis (questions the
  person marks wrong are dropped and the results recomputed); both are published.

## Step 73 (2026-11-01): auditor round 3 (summary only; its note was cut off by its usage limit)
The auditor's run stopped before it wrote the full note. Its summary said: Phase 1 is not complete, so Phase 2 has
not started; the confirmation is otherwise verified (894 questions in every primary contrast, the two 4B
out-of-memory answers scored wrong, the Nemotron table labels fixed). Points and what was done:
1. **"Holm p < 0.0001" overstated.** With 10,000 resamples and none crossing zero, p cannot be shown to be below
   about 2/10,000. Fixed: p = (resamples on the other side of zero + 1) / (10,000 + 1), two-sided. H3 and H4: p ≈
   0.0002, Holm-adjusted 0.0008. H1 0.92, H2 0.34 (Holm). The conclusions do not change. Earlier "<0.0001" entries
   (step 70) are superseded by this.
2. **Phase 1 still open:** per-question loss trace on CG-Bench dev, head ablations (Head A without its residual,
   Head B without its option term) and a true matched-budget speech control are still only planned. They are
   done next, before any Phase 2 work.
3. **M3AV visibility check.** The question is written from the slide image saved at the end of a slide segment;
   with slide builds (bullets that appear one by one), the fact may be visible only at the end of the segment.
   Rule: sample the segment at 1 frame per second and keep as the evidence moment only the seconds whose frame
   matches the saved slide image (same picture after downscaling; threshold set on dev by looking at 20 segments
   with builds). A question whose visible span is shorter than 2 s is dropped.
4. **Independent and blind reviews.** The builder and the auditor each review the 100 test questions from the
   question, options, answer key and slide image only, without seeing the other's verdicts or any system's
   answers. The later human review gets the same material in random order, blind to results, and cannot change
   any method: all methods and settings are frozen (and the question file hashed) before the test run.

## Step 74 (2026-11-01): Phase 1 checks declared before running (matched-budget speech, H3 decomposition, head ablations)
All on the fresh LongVideoBench set (894 questions, 500 videos), Qwen3-VL-4B, K = 4, run once
(`scripts/run_phase1.sh`, ledger `budget_phase1_4b.json`, at most 9,000 calls / 24 h). These are follow-up
analyses on the confirmation set, after its results were seen: reported as secondary, with paired bootstrap
intervals over videos and no claim of confirmation.
1. **Matched-budget speech text.** Same pictures (best framework), 600 words each, every arm built from the whole
   transcript, words given recorded per question: `frames+hybrid` (speech near the shown pictures + word/meaning
   search, = S1t, already run), `bm25full` (new: the v2 BM25 ranking over every line, not the 300-word pool
   excerpt), `hybrid` (word/meaning search only), `frames` (speech near the pictures only). Question: does the
   speech selection policy matter once the budget is equal?
2. **H3 decomposition.** H3 changes the pictures, the text policy and the word budget at once. Two arms split it:
   best-framework pictures + the v2 120-word BM25 excerpt; v2 pictures + the S1t 600-word text.
3. **Head B without its option term.** The confirmed Head B setting (`ev_set`, λ = 2, μ = 1) with λ = 0, on both
   pools (compared with S3 and S4).
4. **Head A without its learned residual.** New pool `hybrid_azs`: the untrained MiniLM scorer over the whole
   transcript in place of Head A (compared with S2).

## Step 75 (2026-11-01): per-question loss trace on CG-Bench dev (auditor P28-P30)
`scripts/v3_loss_trace.py` → `reports/v3_loss_trace/summary.md` (82 dev questions; no model calls). Each question
is counted once, at the first step where its answer is lost: **pool miss** (no candidate frame inside the
human-marked seconds), **choice miss** (the pool has one, the 4 chosen frames do not), **reading miss** (a chosen
frame is inside, the answer is wrong, but the same model is right with at least one of the 5 oracle placements),
**model limit** (wrong even with every oracle placement), and the two kinds of right answer.

| 4B | pool miss | choice miss | reading miss | model limit | right, hit | right, miss |
|---|---|---|---|---|---|---|
| S0 starting design | 39 (48%) | 4 (5%) | 4 (5%) | 5 (6%) | 11 | 19 |
| S1 best framework | 23 (28%) | 8 (10%) | 13 (16%) | 8 (10%) | 20 | 10 |
| S1t + speech text | 21 (26%) | 8 (10%) | 12 (15%) | 8 (10%) | 21 | 12 |
| S4 Head A + Head B | 20 (24%) | 5 (6%) | 11 (13%) | 7 (9%) | 21 | 18 |

(2B: same pattern; S1 24 / 3 / 10 / 12.)
- The finite checklist of where answers are lost, largest first:
  1. **Pool miss, about a quarter of questions** even for the best framework: the search never puts a candidate
     inside the marked seconds. This is the largest loss and is a search problem, not a selection problem.
  2. **Reading miss, 11-16%:** an evidence frame is shown, but one frame among four at one instant is not enough;
     frames placed across the marked seconds do succeed. Candidate fix: spend part of the budget around the best
     hit (local refinement), so more than one shown frame covers the moment.
  3. **Model limit, about 10%:** not fixable by choosing frames.
  4. **Choice miss, only 4-10%:** when the pool contains the moment, the plain rule already picks it most of the
     time. This explains why Head B (a better chooser) cannot add much: the stage it improves loses the fewest
     answers.
- Caveats: the oracle counts a question as solvable if any of 5 placements is right, which is lenient (some
  "reading misses" are luck); 82 questions; "right, miss" (10-23%) shows that many right answers do not need the
  marked moment, so accuracy is a noisy measure of localisation. This is why the M3AV pilot uses evidence hit as
  its primary measure.
- Consequence for the plan: improving recall of the search (pool) and covering the found moment with more than one
  frame are the two stages with room; this is input for Phase 2 and the head redesigns.

## Step 76 (2026-11-01): Nemotron confirmation complete; all comparisons final
The API run finished (ledger `budget_confirm_api.json`: 10,199 calls). The last two secondary comparisons are final:
best framework − dense MobileCLIP top-4 with Nemotron −1.1 (−4.1..+1.8), p 0.48; best framework − 4 evenly spaced
frames +5.4 (+2.0..+8.8), p 0.002. With every answer model (2B, 4B, Nemotron) the best framework shows no demonstrated advantage over dense
MobileCLIP top-4 (−0.8, −0.1, −1.1) and beats evenly spaced frames (+7.5, +8.6, +5.4). H1-H4 unchanged.

## Step 77 (2026-11-01): auditor P38 fixes to the Phase 1 runs; reordered; dev first
- **Head A ablation repaired.** `hybrid_azs` (step 74, item 4) is not a residual-only ablation: it takes the top
  line times with 1-second suppression, while the trained Head A turns scores into moments by density voting
  (σ = 4 s). New pool `hybrid_ahead_r0`: the same checkpoints and the same density-voting moments with the learned
  residual zeroed (`LearnedMoments(zero_residual=True)`), so only the residual differs from S2. `hybrid_azs` is kept
  and named a broader zero-shot baseline.
- **Dev first.** Any choice is made on CG-Bench dev (82 questions): Head A residual zeroed, the zero-shot pool, and
  the matched-budget speech arms. The fresh-set runs stay exploratory follow-ups (results already seen); no setting
  is chosen from them.
- **Order and time.** A 600-word call with 4B takes about 20 s (confirmation: 12 s per call on average), so the
  step-74 runs need about 30 hours. The run was stopped after 361 calls (all cached) and restarted in this order:
  dev items; fresh-set no-text ablations (Head A residual zeroed, Head B λ = 0); fresh-set speech arms last. The
  ledger's time limit was raised from 24 to 40 hours (same 9,000 calls), declared here before the restart.
- Naming (P38): `bm25full` is line-level BM25 with one neighbour each side, not the v2 window packing; "S0 pictures +
  S1t text" recomputes the frame-conditioned text around the v2 pictures, so it tests the text policy, not
  identical text. Realized words per question are recorded (`text_words`) and will be reported with the results.

## Step 78 (2026-11-01): loss trace made unconditional (auditor P37)
`scripts/v3_loss_trace.py` now also reports stage coverage over all 82 CG-Bench dev questions and writes per-question
flags (pool, Path A, Path B, chosen inside the marked seconds; correct). Step 75's "pool miss 28%" was wrong answers
whose pool had no candidate inside; over all questions the best framework's pool has none for 40% (10 of them still
answered correctly).

| 4B | pool inside | Path A (speech) inside | Path B (visual) inside | chosen inside | accuracy | acc., chosen inside | acc., chosen outside |
|---|---|---|---|---|---|---|---|
| S0 starting design | 33% | 26% | 18% | 24% | 37% | 55% | 31% |
| S1 best framework | 60% | 26% | 49% | 50% | 37% | 49% | 24% |
| S4 Head A + Head B | 65% | 33% | 49% | 48% | 48% | 54% | 42% |

- The speech path reaches the marked seconds for only a quarter of questions (Head A raises it from 26% to 33%);
  the visual search for half. Accuracy is about 25 points higher when a chosen frame is inside, but many right
  answers do not need the marked moment.
- Still descriptive: the classes and coverage do not say which intervention recovers the most accuracy. S0 is
  compared with an oracle whose text differs; "model limit" means 5 placements failed.
- Owner (2026-11-01, auditor P44): the 100 checked questions are a **random audit sample** of the full generated
  M3AV test cohort, not the whole test set; every accepted test question is used. The sample is drawn with a fixed
  seed, stratified by lecture (at most one question per lecture first), after the cohort is frozen.

## Step 79 (2026-11-01): M3AV lecture protocol declared (auditor P40-P41)
`docs/v3/m3av_protocol.md`, written before any M3AV question exists. Main points: questions generated by Nemotron
from the slide image and OCR only; grounding check of the quoted slide words; every repeated occurrence counts as
evidence; visibility proxy (frames matching the saved slide, τ set on 20 dev segments with builds, then appended);
no answer-model filter, only labelled `q_only` / `t_only` subsets for secondary analyses; primary measure named
**source-slide interval hit@4**; bootstrap over lectures; methods frozen and hashed before test is generated;
declared contrasts L1 (Head A retrained on M3AV − dense MobileCLIP top-4) and L2 (− best framework), Holm over both;
review of a 100-question sample by builder and auditor independently, then a blind human review; primary analysis
on the whole frozen cohort.

## Step 80 (2026-11-01): M3AV videos checked: complete and time-aligned
`scripts/m3av_video_check.py` → `reports/m3av/video_check.json`. Without ffmpeg, yt-dlp kept the 360p picture
stream (`<id>.f605.mp4`) and the sound separately; only the picture is used. For every dev/test lecture: decodable
duration vs M3AV's last slide end, and the correlation (64 × 36 grey) between M3AV's saved slide image and our
frame at the end of 5 segments.
- Five picture streams were truncated (IPP-68329, MLS-ED58D, NIH-95F8B, NIH-B5E52, NIH-EFBA6: video shorter than
  the slides); they were downloaded again and are now complete.
- After that: 211 of 211 videos decode; median correlation 1.00 and no lecture below 0.5, so M3AV's slide times
  fit our copies (no intro offset). Videos running past the last slide are talks with questions at the end.
- Dev lecture NIH-F5E14 has no slide segments in M3AV, so it yields no questions (it has no eligible slide).

## Step 81 (2026-11-01): M3AV dev questions generated; visibility measure replaced on dev (protocol amendment 2)
`scripts/m3av_questions.py --split dev` → `data/m3av/questions_dev.jsonl`: 98 dev lectures, 1,705 eligible slides,
703 asked (Nemotron), 27 failed F1 (format), 134 failed F2 (quoted words not on the slide), **542 kept**. 79 lectures
reached the quota of 6; NIH-306B0 gave none (all 4 eligible slides failed F2) and NIH-F5E14 has no slides. The
generator put the correct option first in 391 of 517 kept questions; after the fixed option shuffle (amendment 1)
the key positions are 124 / 125 / 136 / 132.

F4 (visibility): the declared 64 × 36 grey correlation could not separate a different slide with the same layout
(up to 0.97) from the right slide under burned-in subtitles (0.58), seen on the 20 dev sheets. Replaced, on dev,
by edge-map correlation at 256 × 144 with τ = 0.85 and a neighbour check (the frame must not match the previous or
next segment's slide better). τ = 0.6 from the first set failed a second unseen set; 0.85 was then checked on a
third unseen set of 20 segments (100 frames by eye): 0 of 22 other-slide frames counted visible, 62 of 62 complete-
slide frames counted visible, 8 of 16 build-stage frames counted visible (the open weakness: an incomplete build
can be in the span). Full record in `docs/v3/m3av_protocol.md`, amendment 2; sheets in `reports/m3av/sheets_*`.
`scripts/m3av_visibility.py apply --split dev --tau 0.85` → `questions_dev_visible.jsonl`: **517 questions, 96
lectures**, median visible time 30 s, 1.46 valid evidence segments per question.

Train generation (for retraining Head A) stopped at its time cap after 156 calls: the ledger sums seconds over
the parallel workers, including waits on API rate limits (HTTP 429), so 24 h of summed time ran out in a few hours
of wall time. The cap was raised to 120 summed hours (`GEN_LIMITS`; the 6,000-call cap is unchanged) and the run
resumed with 4 workers from the cache.

## Step 82 (2026-11-01): Phase 1 results so far (CG-Bench dev complete; fresh-set fast arms complete)
4B answers, 4 frames. Paired difference in points with a 95% bootstrap interval over videos (10,000 resamples).
CG-Bench dev has 82 questions, so one question is 1.2 points.

| Set | Contrast | Difference |
|---|---|---|
| dev | Head A full (S2, 45.1%) − Head A without the learned residual (`hybrid_ahead_r0`, 41.5%) | +3.7 (−3.7..+10.8) |
| dev | Head A without residual − best framework (S1, 36.6%) | +4.9 (−1.4..+11.7) |
| dev | Head A without residual − zero-shot moments (`hybrid_azs`, 37.8%) | +3.7 (+1.1..+6.8) |
| fresh | Head A full (57.2%) − without residual (56.8%) | +0.3 (−2.1..+2.8) |
| fresh | Head A without residual − best framework (55.9%) | +0.9 (−1.5..+3.2) |
| fresh | Head B λ = 0 − Head B λ = 2 (on the best framework's pool) | +0.0 (−0.9..+0.9) |
| fresh | Head B λ = 0 − λ = 2 (on Head A's pool) | +0.6 (−0.2..+1.4) |

Matched-budget speech on dev (best framework's frames, 600-word allowance):

| Text policy | Words used (mean) | Accuracy | − frames+hybrid |
|---|---|---|---|
| frames+hybrid (the S1t policy) | 540 | 40.2% | |
| bm25full (BM25 over the whole transcript) | 539 | 41.5% | +1.2 (−5.4..+6.6) |
| hybrid (moment-search lines only) | 541 | 42.7% | +2.4 (+0.0..+5.3) |
| frames (lines near the chosen frames only) | **117** | 42.7% | +2.4 (−2.4..+7.5) |

- Head A on unseen questions: whatever it gains on the fresh set (+1.3 over the best framework, not significant)
  comes mostly from its moment rule (density voting over speech scores), not the learned residual (+0.3).
  On dev the residual looks larger (+3.7) but the interval is wide.
- Head B's coverage term (λ) makes no difference: λ = 0 and λ = 2 pick nearly the same frames.
- Speech text: on dev, no way of choosing the lines beats the others; all are within 3 questions. The `frames`
  policy uses only 117 words because few lines lie near the chosen frames, so that row is not a matched budget.
  The fresh-set text arms are still running.
- Not yet measured: invalid or abstained answers per system on dev (auditor P38). `rows.jsonl` stores only right
  or wrong, so these come from the answer logs.

## Step 83 (2026-11-01): auditor round 4 (P45-P56): visibility spans fixed, answer-content check, dev diagnostics
- **P48, visibility spans.** `stage_apply` had joined every passing second of a segment into one span, so a frame
  in a non-visible gap could be credited. Spans are now unions of consecutive passing samples. Dev rebuilt: 514
  questions on 96 lectures; 54 have split spans; total visible time −9%. Sheets and `apply` are limited to the
  requested split. Protocol amendment 3.
- **P52, repeated occurrences.** A repeated occurrence must now also show the answer's critical tokens, so "10
  times for each object" no longer matches a slide without the 10. Dev rebuilt from the cache with no new calls: 20
  questions lose 29 of 780 occurrences. Per-frame calibration judgments are published, and the tally is computed
  from them (`reports/m3av/visibility_calibration.md`). Correction to step 81: set 3 had 61 complete-slide frames
  (61 counted visible) and 17 build-stage frames (8 counted visible), not 62 and 16. The review sampler is now
  two-stage, with a lecture-balanced and a question-weighted estimate.
- **P46, Head B λ.** "Nearly the same frames" is now measured: λ = 0 and λ = 2 choose the identical 4-frame set
  for 785 of 894 fresh questions (87.8%) on the best framework's pool and 792 (88.6%) on Head A's pool; on average
  3.7 of 4 frames are shared. There is no confirmed gain from λ, which is not the same as showing no effect.
- **P38 / P50, dev answer diagnostics.** The rows now keep the chosen option. All 13 CG-Bench dev arms (1,066 4B
  answers) were replayed from the cache with zero calls: no abstentions and no failed answers. Whether a choice was
  forced (no letter in the reply, so the most likely letter was taken) was never stored; new answers now record it.
  Nothing here calls for a prompt change.
- **P49, wording.** Holm now receives unrounded p-values; the confirmation report is unchanged to 4 decimals.
  Correction to step 80: 211 videos are present but 210 were duration-checked (NIH-F5E14 has no slide metadata and
  was not decoded). Six of the 1,033 sampled slide correlations are below 0.5 (the lowest lecture median is 0.96),
  and that coarse grey measure was later rejected for visibility. So the check supports complete files
  and approximate alignment, not legibility or alignment everywhere. The generation time cap counts worker time
  including waits, not wall-clock or GPU time.
- **P47.** The loss report now opens with the descriptive caveat.
- Phase 1 interruption: the zero-call dev replay loaded the text encoder on the GPU while the 4B speech arm was
  running. That arm hit CUDA out of memory at 10:08, and the driver stopped before the failed request was cached.
  It was restarted at 10:14 and resumes from the cache. GPU work now runs one job at a time.
- M3AV train questions (for retraining Head A): 300 lectures, 5,706 eligible slides, 1,404 asked, 64 failed F1, 193
  failed F2, **1,147 kept on 297 lectures** (`data/m3av/questions_train.jsonl`), rebuilt from the cache under the
  amendment-3 answer-content rule with no new calls. Generation calls in all: 2,107 (dev 703, train 1,404); 1,701
  rate-limit retry lines in the train log.

## Step 84 (2026-11-01): Phase 1 complete; fresh-set speech arms
All Phase 1 runs have finished: 4,413 charged 4B calls in `runs/v3_baseline/budget_phase1_4b.json`, including the
OOM restart in step 83. These runs are on the fresh set, whose results were already seen, so they are descriptive
follow-ups and **no setting is chosen from them**. Fresh set: 894 questions, 4B, 4 frames. Paired difference with a
95% bootstrap interval over videos (10,000 resamples).

Speech text with a 600-word allowance, on the best framework's frames (S1). The fresh transcripts are shorter, so
mean use is 425 words:

| Text policy | Words used (mean) | Accuracy | − frames+hybrid |
|---|---|---|---|
| frames+hybrid (the S1t policy) | 425 | 57.3% | |
| bm25full (BM25 over the whole transcript) | 425 | 57.0% | −0.2 (−1.4..+0.9) |
| hybrid (moment-search lines only) | 425 | 56.7% | −0.6 (−1.7..+0.5) |
| frames (lines near the chosen frames only) | **214** | 57.7% | +0.4 (−0.9..+1.8) |

Where the gain of the best framework with speech (S1t) over the starting design (S0) comes from:

| Contrast | Difference |
|---|---|
| S1t − S0 (all changes) | +8.9 (+5.7..+12.2) |
| S0's pictures + S1t's text − S0 (text change only) | +2.8 (+1.1..+4.4) |
| S1t − S0's pictures + S1t's text (picture change only) | +6.2 (+3.1..+9.3) |
| S1's pictures + S0's 120-word BM25 text − S0 (picture change only, the other order) | +8.2 (+4.9..+11.5) |
| S1t − S1's pictures + 120-word BM25 (text change on S1's pictures) | +0.8 (−1.0..+2.6) |
| S1t − S1 without speech | +1.3 (−0.6..+3.3) |

- Text policy: as on dev, no way of choosing the lines differs from the others (all within 1 point). `frames` again
  uses fewer words (214), so it is not a matched budget.
- The gain over the starting design comes mostly from the pictures (+6.2 to +8.2 points, depending on the order of
  the two changes). The richer speech text helps S0's weaker pictures (+2.8) but adds little on top of S1's pictures
  (+0.8, not significant). Pictures and text are not additive. *(Corrected in step 85: the +6.2 row also changes
  the speech that goes with the frames, so it is not a pure picture effect; the non-additivity sentence is withdrawn,
  since no interaction test was run.)*
- Answer health: 1 failed answer in each of the S1t, `hybrid` and S0-pictures + S1t-text arms. Abstentions are
  counted only for the new arms, which store the option (1-3 per arm).
- Phase 1's runs are all done. The remaining Phase 1 work is the lecture study (M3AV): retrain Head A on the
  M3AV train questions, then R4 and R5, settings on dev, and the method freeze before test is generated.

## Step 85 (2026-11-01): auditor round 5 (P57-P67): wording and count corrections; fresh set in the results table
- **Step 84, frame decomposition (P60).** The row "picture change only" (+6.2) changes the frame selection under a
  fixed text *policy*, and the `frames+hybrid` policy takes speech around the frames actually chosen. So the
  transcript text shown also changes with the frames. It is a frame-selection change including the text changes
  it causes, not a pure visual contribution. The cleaner control is the +8.2 row, where both arms use the same
  120-word BM25 excerpt (it does not depend on the frames). "Pictures and text are not additive" is withdrawn: the
  order dependence is descriptive, and no interaction test was run.
- **Step 84, text policies (P60).** "No way of choosing the lines differs" is too strong. Correct reading: none
  of the three alternative policies shows a confirmed accuracy advantage over frames+hybrid; failing to find a
  difference does not show they are equivalent.
- **Head B frame counts (P58).** Head B chooses at most 4 frames; with λ = 2 it shows fewer than 4 for 81 (best
  framework's pool) and 82 (Head A's pool) of 894 questions, because those pools are small. λ = 0 shows exactly
  the same number of frames for every question (mean 3.85 and 3.86). "Identical sets for 785 and 792 questions"
  means identical selected sets at a maximum budget of 4.
- **Split spans (P60).** "54 of 514 questions have split spans" (step 83, amendment 3) cannot be reproduced and
  is replaced. Definition: a question has a split span when one valid segment holds at least two visible spans.
  Current file: **58 of 514** (57 have more visible spans than valid segments). Total visible time on the same
  514 questions: 27,729.6 s before the fix → 25,327.7 s now (−8.7%). This combines two changes: the span fix
  alone gives 26,586.0 s (−4.1%; recomputed from the pre-F3 questions with the current rule), and the F3
  answer-content rule removes a further 4.7%.
- **Results table (P60, earlier P36).** `docs/report/all_results.csv` now has a stage 13 with the 35 fresh-set
  rows (894 questions each, 2B, 4B and Nemotron), labelled "fresh confirmation". Each row says that only H1-H4 in
  `reports/v3_confirmation` are confirmatory (Holm) and that its own intervals are uncorrected. The ablation rows
  are named as such.
- **Design document (P60).** `docs/v3/heads_design.md` now opens with an adopted/rejected table. Rejected: the
  fine-tuned cross-encoder and the learned offsets (the proposed new part of Head A). Adopted: the light residual
  Head A and the `ev_set` Head B, neither confirmed on the fresh set. No measurable effect: the option term.
  *(Later reading, step 103: no confirmed effect of the option term; λ 0 − 2 was 0.0 (−0.9..+0.9) and +0.6
  (−0.2..+1.4), which is not proof of zero effect.)*

## Step 86 (2026-11-01): lecture study on dev: S2m trained, R4 and R5 built, settings chosen (protocol amendment 4)
- **S2m.** Head A retrained on M3AV train questions with the deployed recipe unchanged (`scripts/run_s2m.sh`:
  1,147 questions on 297 lectures; evidence = the valid slide segments; inner validation on 15% of train lectures
  picks the epoch; 3 seeds). Speech-moment recall on dev (moments inside a visible span): zero-shot density 62.5% at
  4 moments, S2m 76.8% (`reports/m3av/s2m_dev.json`). This is the speech path alone, not the 4 frames shown.
- **M3AV dev as a dataset** (`scripts/m3av_dataset.py`): evidence intervals = visible spans, so the answer stage's
  evidence hit (1 s tolerance) is exactly the primary measure. `scripts/m3av_hits.py` computes it for every
  system from the saved pools without the answer model.
- **Baselines found to be crippled, and fixed.** The first R2-R5 pools held 4 candidates; merging repeated pictures
  left R3 with 1.4 frames on average (hit@4 46.7) and R5 with 2.0 (59.3, against 74.7 for its own 4 times).
  Final versions rank 96 times by the rule's own score and show the first 4 distinct pictures (hash distance > 0;
  0 beat 3 for R3, R4 and R5). R3 rose to 62.6, R4 to 75.1, R5 to 76.1. The general-video R2/R3 pools are
  unchanged (new pool names).
- **R5 setting** chosen on dev: λ = 0.5 with the option-aware visual query (74.7 at 4 raw times; 10 settings in
  `runs/m3av/vsi/dev_report.json`; text only 61.3, visual only 50.2-61.7).
- **Dev, source-slide interval hit@4** (514 questions, 96 lectures; `reports/m3av/hits_dev.md`):

  | System | hit@4 | − S2m (95% interval over lectures) |
  |---|---|---|
  | S2c Head A (CG-Bench) | 77.2 | +0.2 (−2.2..+2.5) |
  | **S2m Head A (M3AV train)** | **77.0** | |
  | R5 VSI-style | 76.1 | −1.0 (−4.7..+2.9) |
  | R4 option-aware dense | 75.1 | −1.9 (−6.5..+2.9) |
  | S3 Head B | 70.8 | |
  | S1 best framework (S1t has the same frames) | 70.2 | −6.8 (−10.7..−3.1) |
  | S0 starting design | 64.2 | |
  | R3 dense MobileCLIP | 62.6 | −14.4 (−19.1..−9.4) |
  | R2 evenly spaced | 26.5 | |

- Reading: a trained speech locator beats the best framework and plain picture search on lecture slides, but the
  training-free subtitle + visual fusion (R5) and option-aware picture search (R4) are as good on dev. Retraining
  Head A on lectures adds nothing over the CG-Bench head (−0.2).
- **Amendment 4** (before any test question): L1 is now S2m − R5, the strongest training-free rule on dev,
  instead of S2m − R3 (the weakest of R3-R5 on dev). S2m − R3, − R4 and − S2c are secondary. Picker for S2m stays
  MMR (77.0 vs 76.7 for Head A's own top 4). Per-system inputs: `docs/v3/m3av_systems.md`.
- **Method freeze** (commit 3a2e911): `docs/v3/m3av_freeze.json` holds every system's pool settings, picker and
  text policy, L1/L2, the R5 setting, and the SHA-256 of the code, configs and checkpoints (both Head A sets, Head
  B scorer, mpnet weights). SHA-256 of the freeze file: `f054ce41038497d8fd6f5e4703ffd8c6bb5e24fa743209b7dec3e0f1861beb3e`
  as committed (LF), `bead18c3f6d12c7f297a268ed4ebdd9e0539214d68d8f9dda4209d0c556fb687` for the Windows working
  copy (CRLF). Test question generation started after the freeze; no system has run on test.

## Step 87 (2026-11-01): lecture test cohort frozen; freeze supplement (auditor P68-P80)
- **Auditor check of step 86.** Amendment 4 (L1 = S2m − R5) and the repair of the fixed-K baselines are approved;
  the auditor recomputed the R2-R5 hits and frame counts from the saved pools and matched all 24 freeze hashes.
  Test runs stay on hold until the R5 description, a freeze supplement and the test reporting path are complete.
- **R5 is VSI-inspired, not VSI** (P71). Step 86 and amendment 4 said R5 keeps VSI's text branch. It does not
  exactly: the frozen R5 boosts a line similarity M > 0.5 to min(1, 2M), so every such line scores 1 (VSI's paper:
  min(1, M + 2 max(M − 0.5, 0)); VSI's released code, checked: min(1, 0.5 + 2(M − 0.5))); its Gaussian covers the
  whole video, where VSI's covers the line ± 2 s; it embeds raw strings, where VSI's code lowercases and removes 12
  stopwords. R5 stays frozen as built and is named a VSI-inspired MPNet/MobileCLIP baseline. No result is claimed
  against the published VSI system. Wording corrected in `scripts/m3av_vsi.py`, `docs/v3/m3av_systems.md` and the
  protocol.
- **Freeze supplement** (`docs/v3/m3av_freeze_supplement.json`, code commit 01e9709; the original record is
  unchanged). Contents:
  - The cache revision and SHA-256 of every snapshot file for MiniLM-L6, bge-reranker-base and MobileCLIP-S2 (all
    three load by name), and for mpnet and Qwen3-VL-4B (pinned).
  - Package versions.
  - The fully resolved pool configuration of every frozen system for test, the picker and Head B arguments, and the
    merged answer-model configuration.
  - The code changed after the freeze, with new hashes.
  - The exact primary test commands (`scripts/run_m3av_test.sh`).

  SHA-256 of the supplement: `60ddc55ccb17b3fedd3bd26c4900e8c8b34a3adb5e02d797fcbec6318fa56347`. Text files are
  hashed with LF line endings.
- **Changes after the freeze**, none affecting which frames a system shows:
  - `m3av_test` added to the dataset registries.
  - `m3av_vsi.py` refuses any test setting other than lam0.5_qo_max (checked: it exits on lam0.5_q and when no
    setting is given).
  - `m3av_hits.py` runs only the frozen systems on test, stops if any of them lacks a cohort question, and reports
    raw p-values and Holm over exactly L1 and L2 in their own table.
  - The test driver checks models, code and test inputs against the freeze. On its first start it refuses any test
    artifact that already exists, and it hashes every artifact before scoring.
  - Dev check: the rewritten report reproduces every step-86 figure. L1 raw p 0.64, Holm 0.64; L2 raw p 0.0006,
    Holm 0.0012. These are dev results, used for choices, not a confirmation.
- **Test cohort** (generated after the freeze; only counts were looked at):

  | Step | Count |
  |---|---|
  | Lectures processed | 113 |
  | Eligible slide segments | 2,027 |
  | Generation attempts | 1,108 |
  | Dropped by F1 (format) | 53 |
  | Dropped by F2 | 168 |
  | Questions after F1/F2 | 887, from 112 lectures (NIH-C45CB has no eligible segment) |
  | Dropped by F4 (visible under 2 s at τ 0.85) | 34 |
  | **Frozen cohort** | **853 questions, 112 lectures** (1-10 per lecture, median 8; no lecture shared with dev) |

  Cohort file `data/m3av/questions_test_visible.jsonl`: SHA-256 `4bd9cd6483aec7203844a0167fdfce857126984bdf85dbe52d8023359c58ff4e`
  (bytes as written), `f97c6ce5d2f3a6c32c28ee369190cd9ead44b952c6dbd4efd1363a279d57bf5d` (LF). The inputs
  manifest (`docs/v3/m3av_manifest_test_inputs.json`) also covers the split file, `data/m3av_test/qa.jsonl`, and
  the 112 transcripts and videos. Dev inputs and dev artifacts have manifests too, and verify.
- **Review sample** (`scripts/m3av_review.py`, protocol two-stage rule): 100 questions, one from each of the first
  100 of 112 lectures in hash order (`reports/m3av/review_sample_test.json`). The packets hold test content and M3AV
  images, so they are kept out of git (`runs/m3av/review_test/`, 100 folders plus an empty verdict form). They
  contain no system names or results. The builder and the auditor review them separately; a person reviews them
  later, blind.
- No system has run on test.

## Step 88 (2026-11-02): builder review of the 100 test packets (sealed)
- The auditor began its review of the packets but hit its usage limit after about 80 of them, and saved none of
  its verdicts. It has not yet checked P71-P74, so the test run stays on hold.
- The builder reviewed all 100 packets alone. For every packet it looked at the frame at the middle of the
  proposed span; when the slide occurred more than once, the span was short, the slide built up over time, or the
  frame was unclear, it also looked at the saved slide and the other frames. Verdicts:
  `collab/review_builder_test.csv`, SHA-256 `8e77f77745f2ad65715662d7a112b215a35ccbee509aacdedaf6245426f699f9`.
- A sealed builder note (`collab/sealed_builder_note_test.md`, SHA-256
  `991ea9ee53973afc75a659014553f5b70dab2434acea55d116db3d7978e277b3`) holds the review summary and a proposed
  sensitivity analysis. Both are declared before any test outcome. They are kept out of the log until the auditor
  has saved its own verdicts, so its review stays independent. The primary measure and L1/L2 do not change.
- No system has run on test.

## Step 89 (2026-11-03): both question reviews saved; S-src sensitivity analysis sealed before test
- **Auditor review.** The auditor reviewed all 100 packets alone and saved its verdicts before opening the builder
  files (`collab/review_auditor_test.csv`, SHA-256 `b60a0bf85e0fba92ef2e2ecc2b12a7736c7092f4cd0b5a77c29cec92befe2cb4`).
  It then checked both builder files against their step-88 hashes; both match.

  | Reviewer | Accept | Reject | Uncertain |
  |---|---|---|---|
  | Builder | 84 | 10 | 6 |
  | Auditor | 77 | 7 | 16 |

  - The two verdicts agree on 84 of 100 packets (both accept 76). They differ on 14, 17, 26, 30, 37, 39, 41, 49,
    60, 70, 84, 86, 89, 91, 98 and 100.
  - Many differences are severity: the builder rejects partial-span problems that the auditor marks uncertain.
  - These are two model reviews, not human judgments. Both files stay unchanged for a later human review, which
    sees neither model's verdicts and no system results.
- **The builder note, unsealed.**
  - Rejects: 8 for evidence spans (5, 17, 35, 63, 70, 84, 98, 100) and 2 for more than one defensible answer
    (24, 99). Uncertain: 4 hard to read at 360p (1, 28, 38, 39), 1 span (41), 1 video-call banner (30).
  - Every key matched its slide. The transcript inside the span gives or hints at the answer in 41 of 100
    (auditor: 49).
  - **False "other occurrences".** A slide counts as occurring again when another slide shares its title or the
    answer word (for example "Poke" on four different charts). Each such span is matched against its own saved
    image, so the visibility check cannot catch it. 158 of 853 test questions (18.5%) have another occurrence
    (dev: 95 of 514). Of the 14 sampled ones, the builder judged 7 false.
  - The auditor judged more of them false: in 89 the repeat shows the label on the other axis, and in 86 it shows
    RQ4 instead of RQ1. The presence of the answer word is not enough. These are exposure counts, not error rates.
- **Decision.** The auditor accepted the proposed sensitivity analysis (P83). It closed P71-P74 and gave the
  go-ahead for the test run (P92), on condition that the analysis is implemented and sealed first. The test
  questions, F3/F4, thresholds and models are not changed because of the review.
- **S-src hit@4** (`scripts/m3av_hits.py`, descriptive only):
  - The same hit test and the same 4 frames, but each visible span is clipped to the question's own source
    segment, and other occurrences are ignored.
  - 3 test questions and 1 dev question have no source-visible part. They stay in the cohort as misses for every
    system.
  - It reports every system and the L1/L2 differences with 95% intervals, with no p-values or Holm.
- **Dev check.** The primary per-question hits, system scores, L1/L2 with Holm and secondary contrasts are
  identical to the step-87 report. No S-src hit occurs without a primary hit (4,626 of 4,626 system-question
  pairs). Dev S-src (used for choices, not a confirmation):

  | Contrast | S-src difference, points (95% interval) | Primary difference |
  |---|---|---|
  | L1: S2m − R5 | +0.0 (−4.1..+4.2) | +1.0 (−2.9..+4.7) |
  | L2: S2m − S1 | +7.0 (+3.3..+10.9) | +6.8 (+3.1..+10.7) |

- **Second freeze supplement** (`docs/v3/m3av_freeze_supplement2.json`, SHA-256
  `25826f95204154d9ade82b2e57f7d04ff4aee6d2fcdb2d5edde7a9284f9ee23c`, code commit aa86a07). It records:
  - the S-src definition;
  - for the changed scorer and test driver, their supplement-1 hashes and their new ones;
  - the new verifier.

  `scripts/m3av_freeze_supplement2.py verify` checks the original freeze and supplement 1 by hash, then the models
  and code against both. It passes on test inputs, dev inputs and dev artifacts, and it rejects a changed scorer.
  The supplement-1 verifier rejects the changed files as it should, so it is no longer the entry point.
- No system has run on test. Next: `scripts/run_m3av_test.sh`.

## Step 90 (2026-11-03): lecture study, frozen test run (primary measure)
- `scripts/run_m3av_test.sh` ran once, from 18:44 to about 21:09 UTC on 2 Nov, with S-src sealed first (auditor
  P92, path 1).
  - Before starting, it verified the models, code and test inputs against the freeze and both supplements.
  - Every artifact was built fresh and hashed before scoring (`docs/v3/m3av_manifest_test_artifacts.json`,
    SHA-256 `ed21653471f132d0b4f12e5dd2407cba23d645db84bc790d195d6103d5ff661d`). The artifacts verify afterwards.
  - Every frozen system covers all 853 questions in 112 lectures.
  - Report: `reports/m3av/hits_test.{json,md}`, JSON SHA-256
    `6ef21809895cab37194152733ba31a9ae4b68ed2f4aef6dafeac980716088ef1`. Nothing was rerun or changed after
    scoring.
- **Primary contrasts** (source-slide interval hit@4; paired bootstrap over lectures; Holm over L1 and L2):

  | Contrast | Test difference, points (95% interval) | Holm p | Dev difference |
  |---|---|---|---|
  | L1: S2m − R5 | −0.5 (−3.8..+2.9) | 0.83 | +1.0 |
  | L2: S2m − S1 | **+5.3 (+2.7..+7.8)** | **0.0004** | +6.8 |

  - L2 is confirmed: Head A trained on M3AV lectures picks frames on the source slide more often than the best
    framework without it.
  - L1 is not confirmed: S2m does not beat the VSI-inspired baseline. Non-significance is not equivalence; the
    interval allows S2m to be up to 3.8 points worse or 2.9 points better.
- **Secondary contrasts** (descriptive): S2m − R3 +15.2 (+11.3..+19.1); S2m − R4 +2.5 (−1.1..+6.0); S2m − S2c
  −0.4 (−2.1..+1.5). Head A trained on CG-Bench does as well on lectures as Head A trained on M3AV.
- **Systems, hit@4 (%):**

  | System | Hit@4 (%) |
  |---|---|
  | S0 | 67.4 |
  | S1 | 74.1 |
  | S2c | 79.7 |
  | S2m | 79.4 |
  | S3 (Head B) | 73.6 |
  | R2 | 22.9 |
  | R3 | 64.1 |
  | R4 | 76.9 |
  | R5 | 79.8 |

  Head B gives no gain over S1. R5 ranks 96 decoded frames, against up to about 54 requested decodes for the
  hybrid systems; it shows fewer than 4 frames for 27 questions.
- **S-src sensitivity** (descriptive): L1 −1.5 (−4.9..+1.8), L2 +5.0 (+2.4..+7.6). Every system falls by 4 to 6
  points. The conclusions are the same as for the primary measure, so the false repeated-slide labels do not
  explain them.
- **Reading.** On lectures, Head A clearly helps the framework. A simple VSI-inspired subtitle and visual search
  is as strong, and the Head A trained on general video matches the lecture-trained one. This is the same pattern
  as dev, and it is a result, not a failure of the run. Still to do:
  - the secondary answer-model runs, whose commands go into a further supplement first;
  - the blind human review of the 100 packets.

## Step 91 (2026-11-03): secondary answer-model evaluation sealed before test
- **Audit of step 90.** The auditor (P94-P97) recomputed every count, interval and Holm value from the saved
  per-question hits, and independently rebuilt most systems' hits from the saved pools.
- **Corrections to step 90's wording** (P97):
  - "Does as well", "is as strong" and "matches" were too strong. The right phrase is **no confirmed
    difference**: S2m − S2c −0.4 (−2.1..+1.5), S2m − R5 −0.5 (−3.8..+2.9).
  - Head B's point estimate is below S1's (628 against 632 hits), but no S3 − S1 test was declared.
  - S-src shows the pattern is **robust to this source-only sensitivity analysis**. It does not show that label
    errors cannot explain the findings: build stages, overlays, small text and ambiguity remain inside source
    intervals.
  - The per-system drops are 4.1 to 6.3 points, not 4 to 6.
  - L2 compares the whole trained locator package with S1. It does not isolate the learned part, show a benefit
    of lecture training over S2c, or show an answer-accuracy gain.
- **Third freeze supplement** (`docs/v3/m3av_freeze_supplement3.json`, SHA-256 LF
  `e7148696d2cc557fa7f01f041843641bc55b9685d7b3b19ec83c426802182b06`, bytes
  `8384225c29ad7e88622901e324873fbbc5e95edf881bf5b8f36320b8148bd4b1`; code commit a8c9eb0).
  - Written after the primary hit outcomes, and before any answer-model call on test.
  - It chains to the freeze, supplements 1-2, both test manifests and the primary report.
  - It records:
    - the 12 conditions with their exact inputs and text policies;
    - the answerer identity, its prompts and processor settings (the 640×360 frames are not resized beyond patch
      rounding);
    - the reuse, failure, ledger, resume, completeness and seal rules;
    - the analysis and the commands;
    - hashes of the QA code and of the code it runs that the earlier records did not cover.
  - `scripts/m3av_freeze_supplement3.py verify` checks the whole chain, the models, code, package and Python
    versions, and the test inputs and artifacts. It rejects a changed QA script.
- **QA driver** (`scripts/m3av_qa.py`). One record per condition and question, with:
  - the frame ids, times and pixel digests;
  - the transcript line ids and realized words;
  - the prompt hash and request key;
  - the raw reply, the parsed option, and abstention, forced-choice and failure flags;
  - prompt text, visual and generated tokens (counted separately);
  - answer, selection and pool-build times.

  Before each answer, the recomputed frames must reproduce the stored primary and S-src hits. Identical requests
  are answered once and later reuses are marked. No earlier answer cache is read.
- **Dev check** on the first 24 dev questions in hash order (288 records; `reports/m3av/qa_dev_check.json`). Every
  check passes:
  - the run is complete, and every frame condition reproduces its hits;
  - S1t shows S1's frames exactly;
  - the controls get no frames, and systems without a text allowance get no text;
  - realized words reach at most S0 118, S1t 599, `t_only` 600;
  - tokens are logged for every new answer;
  - a resumed run makes no new calls;
  - a repeated row and a conflicting reply are both rejected.
- **Dev resources.**
  - New answers took 1.8 s (`q_only`), about 2.4 s (systems without text), 4.0 s (`t_only`) and 7.6 s (S1t).
  - 12 of 288 requests were identical to an earlier condition and were reused.
  - Model loading took 12 s.
  - Peak GPU memory: torch allocated 11.1 GB, and reserved 19.8 GB on the 15.9 GB RTX 5060 Ti. Reserved memory
    above the card's size means Windows let part of it spill into shared system memory. A claim that everything
    fits in 16 GB needs this qualified.
- **Estimate.** About 8 h of answer time for 853 test questions, plus pool loading.
- The 24-question dev accuracies (`reports/m3av/qa_dev.md`) only test the pipeline. They are too few to read.

## Step 92 (2026-11-04): lecture study, secondary answer-model evaluation on test
- *Wording, accounting and resource statements below are corrected in step 93 (auditor P110-P113); the numbers
  stand. Deviation appendix: `docs/v3/m3av_qa_test_deviation.md`.*
- **What ran.** The sealed driver (`scripts/run_m3av_qa_test.sh`, third freeze supplement, step 91) ran the 10
  systems and the two controls on all 853 test questions (112 lectures): 12 × 853 = 10,236 records. Every
  condition used the same answer model, Qwen3-VL-4B (pinned revision, bfloat16, greedy), at most 4 frames and the
  same prompt.
  - `q_only` is the question-only control: it sees the question and the four options, nothing else.
  - `t_only` is the transcript-only control: no frames, and 600 transcript words chosen by BM25 search over the
    whole transcript.
  - Each question has 4 options, so chance is 25%.
- **Incident and deviation.** Session 1 started 2026-11-03 17:23 UTC.
  - **What happened.** After 4,535 records (378 questions; 4,383 new calls, 152 reused, none failed), the
    `t_only` request for m3av:NIH-D369A:1661.45 raised a driver-level CUDA out-of-memory error. The driver's
    handler caught it, but its own `torch.cuda.empty_cache()` call raised the same error. The run stopped with no
    record for that item, and session 1's record (model load, peak memory) was not written.
  - **Handling.** The run was resumed through the sealed driver with the code unchanged, so the seal stays valid.
    Session 2 started 2026-11-04 02:59 local time (20:59 UTC) and wrote the remaining 5,701 records without a
    failure.
  - **Deviation.** The crashed item was attempted again, although the sealed rule says an out-of-memory item is
    "a failed record, wrong, never retried". On the second attempt it was answered, and wrongly.
  - **Sensitivity.** Scoring it as failed instead leaves every accuracy on all questions unchanged, because a
    failure also counts as wrong. It only removes that question from the "without t_only-correct" mask (322
    instead of 323 questions). On that mask every contrast moves by at most 0.3 points, and every interval keeps
    the same sign. For example, S2m − R5 is +5.9 (+1.6..+10.3) and S1t − S1 is −0.9 (−3.8..+2.0).
- **Ledger.** 9,893 new calls and 343 reused identical requests, 0 failed, 27,386 s (7.6 h) of answer time.
  - Session 2 peak GPU memory: torch allocated 11.49 GB, reserved 16.8 GB on the 15.93 GB card. Reserved memory
    again exceeded the card, with spillover into Windows shared memory.
  - "Runs on one 16 GB GPU" holds only with that qualification. The session-1 crash fits the same pressure.
- **Accuracy, test (secondary, descriptive).**

  | Condition | All 853 (%) | Without q_only-correct, 416 (%) | Without t_only-correct, 323 (%) |
  |---|---|---|---|
  | S0 starting design (120 words) | 79.5 | 69.0 | 65.3 |
  | S1 best framework | 82.5 | 75.0 | 71.5 |
  | S1t best framework + 600 words | 83.5 | 75.2 | 70.3 |
  | S2c Head A (CG-Bench) | 88.2 | 82.2 | 80.8 |
  | S2m Head A (lectures) | 87.2 | 81.2 | 80.8 |
  | S3 Head B | 82.8 | 74.3 | 70.6 |
  | R2 4 evenly spaced | 63.5 | 45.7 | 45.8 |
  | R3 dense MobileCLIP top-4 | 78.3 | 67.5 | 68.1 |
  | R4 dense option-aware top-4 | 84.8 | 77.2 | 76.2 |
  | R5 VSI-inspired | 86.5 | 79.6 | 74.9 |
  | q_only | 51.2 | 0 (by definition) | 18.3 |
  | t_only | 62.1 | 36.5 | 0 (by definition) |

- **Contrasts.** Paired bootstrap over lectures, 10,000 resamples, seed 20261101. Points with 95% interval, raw p
  uncorrected. These 21 cells are outside the Holm family and confirm nothing.

  | Contrast | All 853 | Without q_only-correct (416) | Without t_only-correct (323) |
  |---|---|---|---|
  | S2m − R5 | +0.7 (−1.7..+3.0), p 0.58 | +1.7 (−2.1..+5.4), p 0.39 | +5.9 (+1.6..+10.2), p 0.011 |
  | S2m − S1 | +4.7 (+2.5..+6.9), p 0.0002 | +6.2 (+2.8..+9.7), p 0.0004 | +9.3 (+5.0..+13.6), p 0.0002 |
  | S2m − S2c | −0.9 (−2.4..+0.5), p 0.22 | −1.0 (−3.4..+1.4), p 0.50 | +0.0 (−3.0..+2.8), p 1 |
  | S2m − R3 | +8.9 (+5.9..+11.9), p 0.0002 | +13.7 (+9.1..+18.1), p 0.0002 | +12.7 (+7.6..+17.5), p 0.0002 |
  | S2m − R4 | +2.5 (+0.0..+5.0), p 0.060 | +4.1 (−0.2..+8.2), p 0.066 | +4.6 (+0.3..+9.1), p 0.048 |
  | S3 − S1 | +0.2 (−1.9..+2.4), p 0.87 | −0.7 (−4.1..+2.7), p 0.72 | −0.9 (−5.6..+3.8), p 0.76 |
  | S1t − S1 | +0.9 (−0.8..+2.8), p 0.33 | +0.2 (−2.3..+2.8), p 0.93 | −1.2 (−4.1..+1.8), p 0.47 |

- **Reading** (descriptive, in the order the primary results set):
  - **Answers follow the primary pattern.** S2m answers 4.7 points more questions than S1 on all questions, the
    same direction as the confirmed hit@4 gain L2 (+5.3). Against R5 there is no confirmed difference on all
    questions (+0.7, −1.7..+3.0), as in L1.
  - **One mask favours S2m over R5.** On the mask without t_only-correct questions, S2m − R5 is +5.9
    (+1.6..+10.2). It is one of 21 uncorrected cells, and the mask is defined by this answer model's own
    transcript answers. It is a lead to check on another set, not a finding.
  - **No benefit of lecture training.** S2m − S2c is −0.9 on all questions. Head A trained on CG-Bench answers as
    many questions as Head A trained on lectures. As in step 91, L2 compares the whole trained locator package
    with S1.
  - **No difference for Head B or speech text.** S3 − S1 is +0.2. S1t − S1 is +0.9 (−0.8..+2.8): 600 transcript
    words on top of S1's frames add no confirmed accuracy here, and they cost 1,507 instead of 180 prompt text
    tokens.
  - **The questions are guessable without the video.** The question-only control answers 51.2% (chance 25%) and
    the transcript-only control 62.1%. 437 questions are answered from the question alone, so accuracy on all
    questions mixes visual and non-visual items. The masks exist for this reason, but they depend on Qwen3-VL-4B
    and are not verified visually necessary subsets.
  - **Hit@4 and accuracy rank the systems alike.** R2 has the lowest hit@4 (22.9) and the lowest accuracy (63.5).
    R3 is below S1 on both, and S2c, S2m, R4 and R5 lead on both. No item-level link between hits and correct
    answers is claimed here.
- **Cost per answer** (medians):
  - Without text: about 2.4 s, 180 prompt text tokens and 880 visual tokens (4 frames).
  - S1t: 4.26 s and 1,507 prompt text tokens.
  - t_only: 3.88 s. q_only: 1.47 s.
  - Selection cost is in the primary report.
- **Files.**
  - Records seal `docs/v3/m3av_qa_test_outputs.json` (records LF sha256
    `78760d3248ccd6e7ba69dbdf420ad88b037340d44c4c60b57b8cd721f65cc51f`). The seal file itself is LF
    `5791bcb3659a9c262ea209dcea907f78c521f9385b0b8c1d89075bf69b3db414`, bytes
    `51b8bbbd9bbcf413a90a055fcb86842b02df081a76d3c7f8c0fb02cc09436678`.
  - `reports/m3av/qa_test.json`: LF `f65e5c06a5169e8dd5d19c0ff8f657d9d96e125aa2d1e5e5369a1617143e6dd1`, bytes
    `b1d19db1321dcb791c5e132bd128112650d483e4aeca4656f39b97f7a4b16360`.
  - `reports/m3av/qa_test.md`: LF `84fdbc9af85462109efbe669f9c4ae5358f80f0b872b275627e087dc0747774e`, bytes
    `62541ed67d54a8c3a1d66b0c2df3ee85494d8cd306553ef00eaa26d630e039fb`.
  - The run logs (`runs/m3av/qa_test_run.log`, `runs/m3av/qa_test_run2.log`) and records are local and
    git-ignored.
- **Results table.** `docs/report/all_results.*` now has stage 14, the lecture study, built from the two test
  reports.
  - The hit@4 rows label only L1 and L2 as confirmatory.
  - The accuracy rows are labelled secondary, with uncorrected intervals against the best framework.
- **Next.**
  - The auditor checks this step and the deviation.
  - The owner arranges the blind human review of the 100 packets (P105).
  - Writing targets ICMR (31 Jan 2027) or a workshop rather than CVPR main.

## Step 93 (2026-11-04): audit of step 92; deviation appendix; reporting corrections
- **Audit (P106-P109).** The auditor re-ran the freeze verification, which passed. It then independently checked:
  - all 12 conditions on the same 853 questions and 112 lectures, with correctness against the frozen keys;
  - request reuse, the input constraints, S1t's frames being identical to S1's, and the primary and S-src hits;
  - all 21 contrasts (10,000 resamples, seed 20261101);
  - the 19 stage-14 counts;
  - the output and record hashes.

  Everything reproduces. The retry of the crashed item is accepted only as an explicit **protocol deviation**, not
  as compliant execution. The crash-as-failed sensitivity reproduces. No answer-model rerun is needed, and none
  was done.
- **Deviation appendix (P110)** (`docs/v3/m3av_qa_test_deviation.md`; data in the `.json` beside it). It is
  hashed (`.md` LF `e8e64290b5414bfcc75dba4f62874975f030d353c0ec84e958e2404fd2cf5909`, bytes
  `d03964628ae0bfb8f499252fb963e4d839fadc70a8dabc0aa888b90fecc8981a`; `.json` LF
  `d42fd72575e220947cc49796a21d649a58fba58bbb33157130583408e1c0cd76`, bytes
  `e2549a1809f64508ef31c7e1e6468ac11a16c05c9ad786be36d4f52925e19d05`). It is
  generated by `scripts/m3av_qa_deviation.py`, which only reads its inputs. Before writing, the script:
  - checks the records and sessions against the outputs seal;
  - checks that the first record of session 2 is the crashed request;
  - checks that the session-1 log shows exactly one out-of-memory error plus the second one inside the handler;
  - checks that its as-run column equals the saved report.

  The original records, seal and report are unchanged. The appendix holds:
  - the item, the sealed rule and what actually happened;
  - the complete seven-row sensitivity table with denominators;
  - the call accounting, the memory measurements and the new-call timing medians.
- **Accounting corrections (P110).**
  - **Failures.** "0 failed" describes the saved rows only. There was 1 failed physical attempt.
  - **Calls.** The ledger counts 9,893 completed new calls; 343 rows reuse an identical earlier request. At least
    **9,894** answer-model calls were attempted.
  - **Time.** The ledger's 27,386 s exclude the failed attempt, model and pool loading, selection and the restart.
    They are not end-to-end runtime.
  - **Session 1.** Its summary (model load, peak memory) is unavailable and is not substituted by session 2's.
  - **Where it stopped.** Session 1 wrote 4,535 rows: 377 complete questions plus 11 conditions of question 378.
  - **Restarts.** There was one crash and one restart. The log names `run{2..5}` in the collab entry were only the
    restart bound, not restarts.
- **Corrections to step 92's wording (P111).**
  - "No benefit of lecture training" and "answers as many questions" should be **no demonstrated answer-accuracy
    improvement from lecture training in this experiment**. S2c answered 752 questions correctly and S2m 744;
    S2m − S2c −0.9 (−2.4..+0.5) is not equivalence.
  - "No difference for Head B or speech text" should be **no confirmed answer-accuracy improvement**. S3 − S1 is
    +0.2 (−1.9..+2.4) and S1t − S1 +0.9 (−0.8..+2.8). Both intervals allow effects in either direction; they do
    not show the absence of an effect.
  - "Hit@4 and accuracy rank the systems alike" is wrong as an exact ranking.
    - Hit@4 orders R5 (79.8), S2c (79.7), S2m (79.4); accuracy orders S2c (88.2), S2m (87.2), R5 (86.5).
    - S3 is below S1 on hit@4 (73.6 vs 74.1) and above it on accuracy (82.8 vs 82.5).
    - What holds: the broad grouping is similar (R2 lowest; R3 below S1; S2c, S2m, R4, R5 highest), but the order
      within groups differs.
  - The controls show that this answer model answers many questions without the video. They do not show that each
    question a control answers correctly is non-visual. The masks remain Qwen3-VL-4B-dependent.
  - S2m − R4 has a positive point estimate, +2.5 points, but **no confirmed positive effect**. The displayed
    interval is +0.0 to +5.0 and raw p is about 0.060; it stays descriptive. (Wording corrected in step 94; it read
    "is not a positive effect: its interval reaches zero".)
  - In the sensitivity paragraph:
    - "every contrast moves by at most 0.3 points" should be **about 0.34 points at most** (largest: S2m − S1,
      +0.3394; 0.3 at displayed precision);
    - "every interval keeps the same sign" should be **the same zero-crossing classification**.
- **Resources (P113).**
  - **Units.** The memory figures are GiB (divided by 2^30), although the report keys say "gb". Session 2: torch
    allocated 11.49 GiB, reserved 16.8 GiB, card 15.93 GiB.
  - **What they measure.** Allocated memory is tensor storage; reserved memory is what PyTorch's caching
    allocator holds. Neither measures dedicated versus Windows shared memory. Shared-memory backing explains
    reserved memory above the card plausibly, but it was not measured (step 91's wording also said "spilled",
    which overstates it).
  - **What can be claimed.** The run executed locally on this single card, with one out-of-memory crash, one
    restart and incomplete peak accounting. Strict residence within 16 GiB and failure-free completion are not
    shown.
  - **Selection cost.** "Selection cost is in the primary report" was wrong: `hits_test.md` has no cost columns.
    The answer report has only partial selection and pool timings.
  - **Answer time.** Answer seconds are medians of the answer-model call, not end-to-end latency. Reused rows copy
    the originating call's time. The appendix gives medians over new calls only (for example S2m 2.335 s against
    2.363 s over all rows).
- **Results table, stage 14 (P112).** Numbers are unchanged, and only the labels were corrected:
  - **Measure.** The hit rows now say **source-slide interval hit@4 (automatic visibility proxy)**. They do not
    describe frames "with the answer slide", because the labels are automatic and have known false matches.
  - **Pictures.** "Pictures shown" is now "up to 4" with the realized mean, since S0, R3, R4 and R5 are sometimes
    underfilled.
  - **Question set.** The answer rows name the single re-attempted item and link the appendix.
  - **Summary note.** Every row says it is a summary and points to the full reports, which hold the controls,
    masks and S-src.
  - **S2m answer row.** It now also gives S2m − R5 on all questions (+0.7, −1.7..+3.0), next to S2m − S1.
- **Unchanged.** The primary results (L1, L2), the frozen scripts, the raw records and the seals. The safer
  out-of-memory handler is a change for future runs only. It must be checked on dev or a synthetic case and
  cannot cause a replacement test run.
- **Still open.** The blind human review of the 100 packets (P100, P105) is up to the owner.

## Step 94 (2026-11-05): last reporting corrections (auditor P118-P132); project overview updated
- **Audit (P118-P120).** The auditor reproduced all 42 appendix entries and the accounting, and confirmed that the
  original records, seal, report and frozen code are unchanged. P112 and P113 are closed. Two finite corrections
  remained, and both are made here. No answer-model run was needed or done.
- **Log hashes (P121).**
  - **The problem.** The first appendix labelled both run-log hashes `sha256_lf`. They were in fact raw-byte
    hashes, because the frozen helper `sha()` normalizes line ends only for its listed text suffixes, and `.log`
    is not one of them. The logs did not change.
  - **The fix.** `scripts/m3av_qa_deviation.py` now computes both conventions for the logs itself; the frozen
    helper is unchanged. It lists them in the Markdown and says "one recorded crash and one recorded restart"
    instead of "no other restart happened".
  - **New hashes:** `qa_test_run.log` bytes `26740b262df87263200972494ae9cc897dc171f38b9a8f59d954c59577f2a8e5`,
    LF `c87faaff041de06167f00ae71a9bef795ad32de9e106f0b8bd67a74a59f9a85a`; `qa_test_run2.log` bytes
    `fac48bbdf5a8068f1afbd8fc55341b8ef55a9b31858ba71cec285ea65cae7c09`, LF
    `6e021519d5cb7f17e45d90ad3f8d12d2f059d1e21cb25b173daa6f07dac178b6`.
  - **Nothing else changed.** Apart from the log hash fields, the regenerated JSON is identical to the previous
    version.
  - **Appendix hashes.**
    - New: `.md` LF `36237ff8e4423a5af8961de7c37fee7205ac461676e4e67d4c6b806b57c55ac5`, bytes
      `48bf16e53ea7e399f82ea89a009c3a54f811d2524b21a17ba2899a65fd2e321e`; `.json` LF
      `bb28a031a98129c17dd96431a2ba611642d7bed36f26a8271201b137fcf126cd`, bytes
      `ce9bc3a4c1d475fc68d30d8f4a200bce0a2a52e066f348d265062accb47c1f4d`.
    - Previous (step 93, commit f1fe824, kept in git history): `.md` LF `e8e64290…`, `.json` LF `d42fd725…`.
- **Wording (P122).** In step 93, S2m − R4 now reads: a positive point estimate (+2.5) but **no confirmed positive
  effect** (+0.0 to +5.0, raw p about 0.060), descriptive. The old sentence is quoted there. A rounded interval
  endpoint does not show a zero or negative true effect.
- **Project overview (P126).**
  - **`README.md`:**
    - The opening now separates local answering from API-based question generation and the 120B follow-ups. It
      qualifies the 16 GB claim: peak reserved 16.8 of 15.93 GiB, one out-of-memory restart, strict residence not
      shown.
    - Head A's "under confirmation" is replaced by the fresh-set result (no confirmed gain).
    - A lecture-study section gives the confirmed narrow result (S2m − S1 +5.3 hit@4), the absent advantage over
      R5, the descriptive answer results, and the proxy and human-review limits.
    - The licence line is corrected: Video-MMMU annotations are CC BY-NC-ND 4.0 per the maintainers (raised
      earlier as auditor P15), and M3AV is added.
  - **`docs/EXPERIMENTS.md`:**
    - Row 12 now says the oracle is complete (best of 5 placements, no text, CG-Bench test: 2B 70.9, 4B 61.8,
      Nemotron 81.8), described as a lenient diagnostic, not an attainable ceiling.
    - New rows 15 (lecture primary) and 16 (lecture answers, secondary).
  - **Current state** (collaboration file):
    - It now covers the lecture result and the goal question: the owner's CVPR choice, against the ICMR or
      workshop recommendation.
    - It names the biggest gap: human-validated, question-specific visual evidence.
- **Auditor's system verdict (P123-P125), recorded.**
  - **What holds:** the frozen locator package improves lecture frame selection over S1.
  - **What does not:** it has no demonstrated advantage over R5, and lecture-specific training and Head B show no
    confirmed incremental benefit.
  - **Not measured:** the dominant lecture failure stage.
- **Recommended order.**
  1. Blind human review.
  2. A lecture evidence-loss breakdown and resource profile, with decisions on dev only.
  3. Then at most one redesign (visible slide-state ranking for Head A, or set-conditioned fact coverage for Head
     B) under a fresh held-out protocol.

  No redesign is evaluated on the revealed test.
- **Closure (P133-P139).** The auditor closed P121, P122 and P126, and with them the reporting for P110-P113.
  - Two editorial fixes followed (P137). EXPERIMENTS row 16 now says the full-cohort results and the sensitivity
    interpretation are unchanged, with the transcript-only mask at 322 instead of 323 questions. Current state
    names the 459-question set as the earlier mixed benchmark (404 of its questions are Video-MMMU lectures).
  - Still open, scientifically: the blind human review (owner), then lecture evidence-loss diagnostics and
    resource profiling on dev.

## Step 95 (2026-11-05): blind human review set up
- **What.** The 100 sealed review packets (`runs/m3av/review_test/`, step 87) are now a private review page for
  the owner's group. The packets themselves are unchanged. The page is not in git because it holds test
  questions and M3AV images.
- **What a reviewer sees per item.**
  - The question with the key marked, and the evidence text.
  - The saved slide at full resolution, and the three 360p frames from the proposed span.
  - The transcript lines inside the span.
  - YouTube links that open at the slide start, at each proposed span and at each frame. The video IDs come
    from M3AV's `.ytbUrl` files; YouTube times can differ from ours by a second or two (step 80 video check).
- **What a reviewer does not see:** system names, outputs, results, or the builder and auditor verdicts.
- **Order.** Items are shown in the order of SHA-256 of `human-review:<qa_id>`. This differs from the packet
  order the builder and auditor used, as the protocol asks ("in random order").
- **Form.** The form has the same fields as `form.csv`, in plain words:
  - key correct; one best answer; supported by the slide; legible at 360p in the span (slide on screen and fact
    readable);
  - shortcut (none / general knowledge / speech), recorded separately and not part of the verdict;
  - verdict (accept / reject / uncertain) and a reason code (OK, KEY, AMB, UNSUP, ILLEG, SPAN, UI, OTHER);
  - a note, required for reject or uncertain.
- **How answers are stored.** Each reviewer's answers go to their own record, which other reviewers cannot
  read. A copy-to-text fallback covers viewers who cannot save.
- **Blocks and deadline.** Reviewers pick a block (all 100, halves, thirds or quarters) so a group can split the
  work. The deadline is Sunday 8 November 2026, 23:59 (Dhaka time), ahead of the CVPR registration decision
  (10 Nov).
- **Generator.** `scripts/m3av_review_site.py` reads the packets and writes the page data and images to a folder
  outside the repo.
- **Next.** Once the human verdicts are in, compare them with the two model reviews (step 89). Then compute the
  sample rejection rate and the human-review-filtered sensitivity of the automatic source-slide proxy (step 99
  corrects "human-validated S-src sensitivity"), as declared in the protocol, without changing
  L1, L2 or any method.

## Step 96 (2026-11-05): where the source slide is lost (auditor P125)
- **What.** `scripts/m3av_evidence_loss.py` reads saved files only: the candidate pools, the hit reports and the
  test answer records. It runs no model and needs no GPU. For every frozen system and question it asks at which
  stage the source slide is lost, using the primary measure's test (a frame within 1 s of a visible span):
  - **not in pool:** no candidate in the saved pool is inside a span (candidate stage);
  - **in pool, not picked:** some candidate is, but none of the 4 chosen frames (selection stage);
  - **picked:** = source-slide interval hit@4.

  The script checks that its picked column equals the hit reports and the answer records' `primary_hit`. Labels
  are automatic (the visibility proxy), not human-verified. R2-R5 save only the frames they show, so their loss
  cannot be split by stage.
- **Dev (514 questions, 96 lectures): the basis for any redesign.** Percent of questions, 95% lecture-bootstrap
  intervals.

  | System | Slide in pool | Lost at candidates | Lost at selection | Picked (hit@4) |
  |---|---|---|---|---|
  | S1 | 77.8 (73.6..81.9) | 22.2 | 7.6 (5.4..10.0) | 70.2 |
  | S2c | 85.8 (82.5..89.0) | 14.2 | 8.6 (6.0..11.4) | 77.2 |
  | S2m | 85.8 (82.5..89.0) | 14.2 (11.0..17.5) | 8.8 (6.1..11.6) | 77.0 |
  | S3 | 77.8 | 22.2 | 7.0 | 70.8 |
  | R5 | (only the chosen frames saved) | | | 76.1 |

  - S2m's saved pool covers the slide (proxy) for 85.8% of questions instead of S1's 77.8%, partly offset by
    selection losses. "Pool" means the retained pool after proposal, sampling, caps and merging, so this does not
    isolate retrieval, does not show the fact is absent from the video, and does not attribute the gain to the
    learned residual alone (step 99 corrects "Head A's gain over S1 is entirely at the candidate stage"). Selection keeps about 90% of what the pool holds in every framework system (S1 90.2%,
    S2m 89.8%). Head B does not change the pool and picks about as well as S1.
  - S2m's two losses are of similar size: 14.2 points at the candidates and 8.8 at selection. With perfect
    selection from its own pool, S2m would reach 85.8 on dev against R5's 76.1. This is an upper bound, not a
    result.
  - For a redesign, both losses are candidates; pool loss (14.2) is the larger. Selection first is an engineering
    hypothesis that needs cost and fact-visibility evidence, and 85.8 is a proxy upper bound under perfect
    selection, not an attainable gain over R5 (step 99 corrects "dev points first at the selection step"). Any
    choice is made on dev and confirmed on fresh held-out lectures.
- **Test (853 questions): post hoc and descriptive.** The test is revealed, so nothing here chooses a method. The
  stage split matches dev: S2m slide in pool 89.0%, lost at candidates 11.0, at selection 9.6, picked 79.4.
  - Seeing the slide goes with correct answers. When S2m picked the slide, the answer model was right on 93.9% of
    677 questions. When the slide was in the pool but not picked, 56.1% of 82; when it was not in the pool, 66.0%
    of 94.
  - Without questions the question-only control answers correctly: 90.5% of 328, 35.9% of 39 and 55.1% of 49.
    Every framework system and R3-R5 show the same pattern (picked 91-95%, otherwise 49-67%).
  - S2m outcomes on all 853 questions: picked and correct 636; picked and wrong 41 (proxy hit with a wrong answer, 4.8%; cause not yet adjudicated: unreadable frames, false
    visibility matches, missing facts, ambiguous keys or answerer errors; step 99 corrects "answer-stage loss"); not
    picked and correct 108; not picked and wrong 68.
  - This is the item-level link that step 92 did not claim. It is an association (questions where the slide is
    easy to find may also be easier), not a causal effect.
- **Unchanged.** L1, L2, every frozen file and the method. The human review (step 95) is still the check on the
  labels themselves.
- **Files.** `reports/m3av/evidence_loss_{dev,test}.{json,md}`.

## Step 97 (2026-11-06): lecture resource profile (auditor P125)
- **What.** `scripts/m3av_profile.py` measures time and memory per stage. It runs no new method and changes no
  frozen file or output.
- **Time from the frozen runs' own records** (`saved`, CPU only; `reports/m3av/profile_saved.md`). Seconds; test
  figures, dev within a few hundredths:

  | Stage | S1 | S2m | R5 |
  |---|---|---|---|
  | Pool per question: sum of the four instrumented components (retrieval, speech/Head A scorer, frame decoding, MobileCLIP) | 0.67 | 0.97 (scorer 0.34) | 1.42 |
  | One answer call, 4 frames, Qwen3-VL-4B (mean / p95) | 2.47 / 3.63 | 2.41 / 3.58 | 2.44 / 3.57 |

  S1t (600 transcript words, about 1,520 text tokens): 4.55 s per call. q_only 1.48 s; t_only 3.97 s. The 2 s
  feature pass over a whole lecture takes about 9 s (test: 112 lectures, 19 min each, 955 s in all). Per question
  the answer call dominates. The scorer column holds the ordinary speech scorer in every system; with the learned
  head it rises from about 0.04 to 0.34 s. R5's pool is slower than S2m's because it decodes 96 ranked times.
  The pool sum leaves out OCR and text embeddings (about 0.01-0.02 s), serialisation, other uninstrumented work, the
  per-lecture feature index, R5's separate scoring pass (93 s for 514 dev questions), model loading, selection and
  transcript building. Answer times exclude reused replies, so unequal fresh-call subsets do not show that one
  selector's answers are faster (step 100 adds these limits).
- **Memory, measured again on dev** (`gpu`, first 100 dev questions in hash order, R5 scoring over all 514;
  `reports/m3av/profile_gpu.md`).
  - **How.** Each frozen stage is re-run with the frozen code into `runs/m3av/profile/`. For each stage the script
    records the wall time, the torch allocated and reserved peaks, and the process's Windows counters: dedicated
    memory (on the card) and shared memory (GPU memory held in system RAM), sampled each second. The answers run
    twice on the saved pools: first as frozen, then with PyTorch's allocator capped at 14.28 GiB (free card memory
    minus 0.5 GiB).
  - **Checks.** The features (maximum difference 0.0), the R5 times (514 of 514) and the S1, S2m and R5 pools
    (100 of 100 each) are identical to the saved ones. The 144 replies that overlap the earlier dev check records
    are identical to them. No out-of-memory error occurred in either pass.

  | Stage | torch peak allocated | torch peak reserved | dedicated peak | shared peak |
  |---|---|---|---|---|
  | Features | 1.68 | 2.04 | 2.19 | 0.08 |
  | R5 scoring (MobileCLIP + MPNet) | 3.48 | 4.93 | 5.08 | 0.08 |
  | Pools S2m (MiniLM + BGE + Head A) | 1.85 | 3.60 | 3.75 | 0.08 |
  | Answers as frozen (600 calls) | 11.46 | **17.26** | 15.23 | **1.56** |
  | Answers capped at 14.28 GiB (600 calls) | 11.46 | 14.19 | 14.13 | 0.08 |

  All figures are GiB; the card has 15.93 GiB and 14.79 GiB were free at the start.
  - **What this shows.** In this replay, PyTorch's peak allocated memory was 11.46 GiB, and the weights take
    8.4 GiB. The cache grew past the card (17.3 GiB reserved), and Windows reported 1.6 GiB of the process in shared
    system memory. This is consistent with the frozen test run's 16.8 GiB reserved, but does not measure that run's
    mechanism directly (step 100 corrects "tensors never need more than 11.5 GiB" and "That is how the frozen test run
    reached 16.8 GiB").
  - With the cap, all 600 replies are identical to the uncapped pass. Peak reserved memory was 14.19 GiB,
    dedicated 14.13 GiB, and shared memory did not rise above 0.08 GiB at one-second sampling.
  - The capped pass took 6.3% less time (1,518 s against 1,620 s). This is one fixed-order replay: the capped pass ran
    second with the model already loaded, and the auditor's CPU checks overlapped the first pass. It is therefore an
    observed difference, not a speed-up caused by the cap.
  - **Claim, bounded.** This was a 100-question, six-condition dev replay of the answer stage. With a 14.28 GiB cap on
    PyTorch's caching allocator, it completed 600 calls with identical replies and 14.19 GiB peak reserved memory,
    and shared GPU memory did not rise above 0.08 GiB at one-second sampling.
    - The cap limits PyTorch's allocator, not every driver allocation.
    - One-second samples can miss brief spills.
    - The cap was not run end to end for every system.
    - It does not show zero transient spill, a universal memory need, or a resident full test run.
    - The frozen test runs were not capped, so the step-93 qualification and the crash and restart stand for them.

    (Step 100 corrects "the process stays on the card", "about 6% faster" and "the pipeline can run strictly within
    the card".)
- **Report fix before commit.** The per-call peak reset also reset each answer stage's own torch peak. The stage
    rows are therefore built from the maximum over calls (`report` re-renders them from `runs/m3av/profile/answers.jsonl`);
    no re-run was needed.

## Step 98 (2026-11-06): human-review analysis fixed before any human verdict
- **What.** `scripts/m3av_human_review.py` implements the declared analyses (protocol "Question review",
  amendment 3). It was written and committed before any human verdict existed, and it reads saved files only.
- **Inputs.** The review page's saved store, or a reviewer's "Copy my answers" text, or a filled `form.csv`.
  - An answer counts only if it is complete as the page defines it: four checks, the shortcut, a verdict, a
    reason code, and a note for reject or uncertain.
  - Reviewers are reported as H1, H2, and so on; names stay out of git.
- **Combination rule.** If one reviewer judged an item, that verdict stands. If several did, the majority verdict
  stands, and a tie counts as uncertain. Items no one completed are reported as unreviewed and left out of every
  rate.
- **Analyses.**
  1. The sample's rejection rate, lecture-balanced and cohort question-weighted (Σ n_l r_l / Σ n_l), with 95%
     lecture-bootstrap intervals (10,000 draws, seed 20261101). The rate of rejected or uncertain items is also
     reported.
  2. The four checks, the shortcut and the reason codes. Agreement of the combined human verdict with the builder
     and auditor reviews (percent and Cohen's kappa), and between humans where they overlap.
  3. Secondary 1 (sensitivity; never promoted over the primary): every system's hit@4 and S-src hit@4, and L1 and
     L2 with paired lecture-bootstrap intervals (no p-values, no Holm), on the cohort without the human-rejected
     sample questions, with one mask for all systems. The same with rejected or uncertain items removed, hit@4 on
     the accepted sample alone, and the answer model's accuracy (descriptive) under the same masks.
- **Dry run.** The two model reviews stood in for humans, and the output went to a scratch folder, not
  `reports/`. The full-cohort row reproduces the frozen reports (S2m 79.4; L1 −0.5 (−3.8..+2.9); L2 +5.3
  (+2.7..+7.8); answer accuracy S1 82.5, S2m 87.2, R5 86.5). Builder against auditor gives 84 of 100, as in step
  89.
- **Input check.** A synthetic store (60 answers and 1 incomplete) and a copied text (60 answers, overlapping by 20)
  were read correctly. The incomplete answer was excluded, and fed alone, the builder's verdicts reproduce its
  84 / 10 / 6.
- **Next.** When the human answers arrive (owner: about 8 November), save the store to `runs/m3av/review_human/`
  and run the script once. The outputs are `reports/m3av/human_review_test.{json,md}`.

## Step 99 (2026-11-06): human-review analysis corrected before any human verdict (auditor P140-P151)
- **Access record (P146, P151).** Nobody on the analysis side has seen a human answer.
  - The builder read the review page's store once, on 2026-11-05 at 16:43 UTC (22:43 Dhaka), to test the page
    before it was shared. The store was empty ("No documents matched"), and it has not been read since.
  - Known before this plan: the primary and test QA results and both model reviews.
  - This plan is a descriptive supplement written before any human outcome, not a retroactive preregistration of
    the test study. The step-98 version stays in git (commit 560c33f).
- **Corrections to `scripts/m3av_human_review.py`.**
  - **Vote (P143).** A strict majority of an item's valid votes decides; otherwise the item is uncertain, reported
    as "no majority". Step 98's code took a plurality: for example, 2 accept, 1 reject and 1 uncertain gave accept.
  - **Identity and revisions (P143).**
    - A private `identities.json` lists every source and gives each a stable reviewer key; unlisted sources or
      documents stop the run.
    - Each reviewer has one final answer per question. Identical copies count once. Between differing store
      versions, the latest answer time is a revision and wins. Any other difference stops the run unless the
      reviewer's confirmed source is named: a CSV against the store, two rows in one file, or equal times. Import
      order never decides.
    - Reviewers appear as H1, H2, ... with per-item votes. The raw exports and the map stay private and are hashed.
  - **Validity (P144).**
    - An empty field makes the answer incomplete: missing for that reviewer, not a vote.
    - A non-empty value outside the page's vocabulary, or a question outside the sample, stops the run. Reason
      codes are limited to OK, KEY, AMB, UNSUP, ILLEG, SPAN, UI and OTHER.
    - The shortcut never affects validity.
  - **Contradictions (P144).** The following are contradictory:
    - an accept with any check answered n, or with any reason other than OK;
    - a reject or uncertain with reason OK;
    - KEY with key y, AMB with one y, UNSUP with slide y, and ILLEG with legible y.

    Nothing is inferred from the fields. The owner asks the reviewer, blind, to recheck the item. A revision made by
    9 November, 23:59 Dhaka, in a snapshot marked "clarification", replaces the answer. Otherwise the answer is
    unresolved and is not a vote.
  - **Cutoff and eligibility (P146).** Answers count if made (store time) or received (CSV) by 8 November, 23:59
    Dhaka; later ones are reported as late. Reviewers are the owner's group members, blind to systems, results and
    model reviews. The builder, the auditor and the owner are not reviewers.
  - **Coverage (P145).**
    - Missing is kept distinct from uncertain.
    - The rates are the planned estimates only if all 100 items are reviewed. Otherwise they are labelled
      observed-review rates, conditional on the responders, and the missing items and page blocks are listed.
    - The intervals cover sampling only. Check and reason counts are over valid answers, so they can exceed 100.
  - **Masks (P142, P146).**
    - **Secondary 1** is the declared mask: the human-rejected sample questions are removed. Uncertain and
      unreviewed questions are kept, which does not validate them.
    - **Conservative:** the reject-or-uncertain mask, labelled as such.
    - **Accepted sample only:** labelled descriptive, with its question and lecture counts.

    Each mask lists its dropped ids and reasons, the questions and lectures retained and removed, the per-lecture
    attrition, and every system and condition, controls included.
  - **Fail-closed checks (P146).** The run stops unless:
    - the sample is exactly 100 distinct cohort questions from 100 lectures;
    - the hit report's systems are the frozen roster and cover all 853 questions;
    - every QA condition covers every question exactly once;
    - both model reviews cover the sample.

    The output records the SHA-256 of the sample, the hit report, the QA records and seal, the model reviews, the
    page files, the script, every raw export and the identity map.
- **Tests.** `tests/test_m3av_human_review.py` holds 11 synthetic cases, all passing:
  - strict majority, and no majority;
  - identical exports;
  - store revisions;
  - a CSV-against-store conflict, with and without a resolution;
  - a conflict inside one file;
  - invalid reasons and ids;
  - an unlisted document;
  - incomplete and late answers;
  - contradictions, with and without a timely clarification;
  - missing items, with an observed-review label;
  - zero accepted items;
  - no valid vote.
- **Dry run.** The two model reviews stood in for reviewers, and the output went to a scratch folder. The auditor's
  own codes (for example SMALL_TEXT) were stopped by the vocabulary check, as intended; for this dry run only, they
  were mapped onto the page's codes. The full-cohort row reproduces the frozen report: S2m 79.4; L1 −0.5
  (−3.8..+2.9); L2 +5.3 (+2.7..+7.8).
- **Review page (P147).** Its source and a manifest are in the local collaboration folder, outside git:
  `collab/review_page/{index,template}.html`, `items.json` and `manifest.json`. `index.html` has SHA-256
  `941fb430457eb522626446de427643f72562651ba8d54f5abffa31a1a6798159`.
  - **Storage rules.** `people`: read admin, write owner. `people/{self}`: read view, write interact.
  - So a Contributor reads and writes only their own document, and reading all documents needs the owner or an
    Editor. Reviewers must be invited as Contributors, not Editors.
- **Wording (P147, P148).**
  - Step 95 now speaks of a human-review-filtered sensitivity of the automatic proxy, not human validation.
  - In step 96:
    - "candidate stage" becomes the retained pool's proxy coverage, not a gain caused by Head A;
    - the 41 wrong answers with a picked slide are "proxy hit with a wrong answer, cause not yet adjudicated";
    - selection is no longer ranked first.
  - The replaced phrases are quoted in place.

## Step 100 (2026-11-06): input guards, per-reviewer drafts, bounded resource wording (auditor P152-P162)
- **Still no human answer seen.** The store has not been read since step 99's access record.
- **Input guards (P157)** in `scripts/m3av_human_review.py`:
  - **One snapshot, one answer.** Within one snapshot, a reviewer has at most one answer per question. A second,
    different answer, even with a later time, stops the run; latest-time revisions apply only across snapshots.
  - **Raw inventory.** Every file in `raw/` must be listed in `identities.json`, and every listed file must exist.
  - **Model reviews.** Each must give exactly one valid verdict per sampled question. Duplicates or other values
    stop the run. Their own historical reason codes are not read.
  - **Recorded hashes.** The frozen inputs must match the hashes recorded before any human verdict, or the run
    stops (`EXPECTED`):
    - the sample;
    - the hit report: the step-90 byte hash `6ef21809...`, as in supplement 3;
    - the QA records: the seal's `sha256_lf` `78760d32...`;
    - both model reviews: the step-88 and step-89 hashes;
    - the page files: version 4 below.

    The page files are now required, not optional.
  - **Pseudonymous labels.** Output source labels look like "raw source 1 (store, cutoff)", so no file name can
    leak a reviewer's name.
  - **Snapshots (P154).** The owner saves the store at the cutoff (role "cutoff") before any clarification is
    requested, then saves a separate later snapshot for clarifications. Neither snapshot is edited.
  - **Tests.** 5 new synthetic tests cover a conflict inside one store snapshot, an unlisted or missing raw file,
    pseudonymous labels, the model-review checks and the hash check. 16 of 16 pass. The end-to-end dry run still
    reproduces the frozen full-cohort row.
- **Review page, version 4 (P158).** Same link, same storage rules; saved answers are untouched.
  - **The defect.** Versions 1-3 kept one browser draft (`lsc-draft`) per browser, loaded it before sign-in, and
    could upload it into a new reviewer's record.
  - **Now.**
    - Drafts are kept per signed-in reviewer (`lsc-draft:<id>`) and read only after sign-in.
    - The old shared draft is never loaded or uploaded by itself, and it is not deleted.
    - Its owner can restore it only by an explicit click. Each restored answer is then marked `restored`, and the
      analysis reports how many were restored per reviewer.
    - Signed-out or view-only visitors still share one draft per browser (`lsc-draft:this-browser`), so each
      reviewer needs their own browser profile or device in that case.
  - **Files.** The version-3 source is kept in `collab/review_page/v3/`. Version 4: `index.html` SHA-256
    `0f2a69f4c2120a28d7e7b1e4554026dd9b98a0924f0dd4fcd00c37088f19d597`, `template.html`
    `1ebb157d3508d050c84829fca6bdee48c4bb9167500324a7d069e97a6c5fc7b3`.
  - **Limits, disclosed.**
    - The page's "Done" checks completeness only, not the analysis's contradiction rule, so a "Done" item can still
      need clarification.
    - The page suggests a reason from the checks (SPAN whenever legibility is "n", unless the reviewer changes it).
      Reason counts therefore include these defaults and are not wholly unaided judgments; they are not recoded.
  - **Not yet verified.** A synthetic two-account check of the live page has not been run, and the invite list is
    the owner's to confirm.
- **Resource wording (P159, P160).** Step 97 is narrowed to the observed replay, with the old phrases quoted, and the
  README, Current state, both profile reports and the teacher deck say the same.
  - **The memory claim.** This was a 100-question, six-condition dev replay of the answer stage. With a 14.28 GiB
    cap on PyTorch's caching allocator, it completed 600 calls with identical replies and 14.19 GiB peak reserved
    memory. Shared memory did not rise above 0.08 GiB at one-second sampling. This is not proof of:
    - zero transient spill;
    - a universal memory need;
    - a capped end-to-end run of every system;
    - the mechanism of the test run's 16.8 GiB.
  - **The 6.3% time difference** between the two passes is observed in one fixed-order replay, not a speed-up
    caused by the cap.
  - **Pool times** are the sum of four instrumented components. What they leave out is listed. The scorer column
    includes the ordinary speech scorer in every system.
- **Review page, version 5 (same day).** The owner asked what happens when a reviewer signs out. In version 4,
  answers made while signed out stayed in that browser's shared draft, and after signing in they were not offered,
  so they were not lost but did not appear. The explicit "Restore" offer now covers them too; nothing is loaded or
  uploaded without the click.
  - Saved answers in the shared record are untouched by a sign-out: they reappear when the reviewer signs in again
    with the same account.
  - `index.html` SHA-256 `8c6e4b4368b221ad9fe586300db5b0e94b570604e44487f925a4948042ae5ef5`. The version-4 source is kept in `collab/review_page/v4/`,
    and EXPECTED is updated.

## Step 101 (2026-11-06): owner-only progress box on the review page; no answers read
- **Why.** The owner asked how many people have answered. Reading the store would show the builder every verdict,
  so the builder still does not read it. Instead, the review page (version 6, same link) gives the owner a box that
  shows counts only.
- **What the box shows.** The number of reviewers who started. For each reviewer, their name, block and the number
  of items finished, by the page's own completeness rule. The item numbers nobody has finished, and how many items
  have two or more reviewers.
- **What it never shows:** verdicts, checks, shortcuts, reasons or notes.
- **How it works.** The box appears only when `user.isOwner()` is true. The storage rules already let the owner read
  `people`, so the owner's browser loads the documents to count them; the code displays only the counts. Reviewers
  see no change.
- **Why counts are allowed.** Counts say nothing about outcomes, and they let the owner fill gaps before the cutoff.
  Using them to reassign blocks is allowed. Using them to steer what reviewers answer is not: no reviewer is told
  another's verdict.
- **Builder access stays as recorded in step 99.** The builder reads the store once, at the 8 November 23:59
  cutoff, saves it unedited as the cutoff snapshot, and only then runs the fixed analysis.
- **Files.** `index.html` SHA-256 `7f3024ee058a4f2f8d44aa20c13142ce0d05d750d459489e132e856ff2f935cb`. Version 5 is kept in `collab/review_page/v5/`,
  and EXPECTED is updated.


## Step 102 (2026-11-09): cutoff snapshot; reviewer answers arrived as copied text
- **Store snapshot.** Read once, at 2026-11-08 18:33 UTC (00:33 Dhaka on 9 November, 34 minutes after the
  cutoff), saved unedited outside git. It holds one document: the owner's own 12 answers (items 68-79), all saved
  before the cutoff. The owner is not a reviewer, so these are reported separately and are not votes. *(Corrected in
  step 103: by their saved times, 4 of the 12 were saved before the cutoff and 8 after. All 12 are excluded either
  way.)*
- **Why the group's answers are not in the store.** The page's share settings offered the three group reviewers
  view-only access, so the page could not save their answers (the read-only path of step 95). They used the page's
  "Copy my answers" fallback and sent the texts to the owner, who passed them on.
- **Received copies.** Three reviewers, saved unedited with SHA-256 in the private folder: H-a all 100 items, H-b
  items 1-34, H-c items 35-67. Received by the builder at about 00:33 Dhaka on 9 November.
- **Fixed analysis.** With that receipt time all three copies are late, and `scripts/m3av_human_review.py` stops as
  designed: "No valid human vote for any sampled question". No result is reported from it.
- **Open.** The time each copy reached the owner decides whether the copies are on time under the unchanged rule.
  If they reached the owner after the cutoff, accepting them is a deviation decided after the answers were seen, and
  would be reported as such.
- **Update (same day): copies on time; fixed analysis run.** The owner reports that all three group reviewers sent
  their copied texts in the thesis group chat before the cutoff; only the owner's own copy came later (not a
  reviewer). The received time is recorded as 8 November 23:59 Dhaka, as reported by the owner; the chat messages are
  kept as evidence. No rule was changed. Report: `reports/m3av/human_review_test.{json,md}` (pseudonyms only).
  - All 100 sampled items reviewed (67 by two reviewers, 33 by one). Combined verdict: 97 accept, 0 reject, 3 no
    strict majority (1 accept, 1 reject each; all three rejects give reason KEY). No contradictory answer, so no
    clarification is needed.
  - Rejected rate 0.0%; rejected-or-uncertain 3.0% lecture-balanced (0.0..7.0), 2.7% question-weighted (0.0..6.4).
  - Agreement: human vs builder 82%, human vs auditor 77%, builder vs auditor 84%. Kappa is near zero for the human
    pairs because almost every verdict is accept (one reviewer accepted all 100 with every check yes and no notes);
    percent agreement is the more informative number here. *(Corrected in step 103: both numbers are reported;
    neither shows reviewer care, and the high percent agreement mostly reflects the all-accept pattern.)*
  - Masks change nothing material: L2 +5.3 (+2.7..+7.8) on all 853; +5.4 (+2.8..+8.0) without the 3 split items. L1
    −0.5 and −0.4, not significant either way.
  - Reported as notes, not votes: one of the three split items was also rejected by both model reviews as ambiguous;
    one reviewer accepted an item while noting that the slide appears later than the marked start.
- **Update: exact receipt times from the group chat.** Screenshots of the three chat messages are kept privately
  with SHA-256 (`runs/m3av/review_human/evidence/`). Receipt times, Dhaka: H-c 8 November 00:59, H-a 21:42, H-b
  23:59, all before the cutoff. The identity map now holds these times; the rerun gives the same report. Two of the
  texts were posted by group members under a different name from the reviewer name inside the text; the owner is
  asked to confirm these are the same people (nickname and formal name).
- **Update: identities confirmed.** The owner confirms that the two posters are the reviewers named in the texts
  (thesis group members; chat name and nickname differ). The human review is closed with no open item.
  *(Corrected in step 103: collection and the fixed analysis are complete, but reviewer independence is unverified
  and the result has the limits listed in `reports/m3av/human_review_limits.md`.)*

## Step 103 (2026-11-09): final audit answered (auditor P163-P180); limits of the human review stated
- **Final audit.** The auditor reproduced H1-H4, the 15 fresh secondary contrasts, the human-review report and all
  583 count-bearing rows of the results table, and wrote a claim ledger of 58 claims (C01-C58), each marked
  validated, needs rewording or not supported. From now on the paper and the supervisor material use only
  validated claims, or the reworded form where rewording is required. Not to be used: blanket "no failure, no
  retry, strictly resident" claims for the frozen QA run; "0% label error" or "97% of labels verified"; "every run
  stayed within budget" or "fully local"; "first", "state of the art" or a new generic selector.
- **Human review limits.** New file `reports/m3av/human_review_limits.md`; the generated report is unchanged.
  - One reviewer (H1) accepted all 100 items with every check yes and no notes, and was the only reviewer for items
    68-100. Under the fixed strict-majority rule a combined reject was therefore impossible for every item, so the
    0 rejects and the unchanged result after removing rejected items follow from the vote pattern; they are not
    independent evidence that every label is right.
  - Percent agreement and kappa are both reported; kappa is near zero or undefined because almost every verdict is
    accept. Neither shows reviewer care. The 0..0 interval for the rejected rate is degenerate (no rejects in any
    resample), not an upper bound on label error.
  - Independence between reviewers is unverified: the copied answers were posted in a group chat at different
    times, and view-only drafts live in the browser. The owner is asked whether later reviewers could see earlier
    answers and whether each used a separate browser profile or device, and how consent was given.
- **Corrections.** Step 102: the owner's 12 store answers were 4 before and 8 after the cutoff, not all before (all
  excluded either way); "closed with no open item" and "percent agreement is the more informative number" corrected
  with marked notes. Steps 72, 84 and 85: marked notes beside "heads do not add accuracy", "picture change only",
  non-additivity and "no measurable effect": the correct reading is no confirmed increment within the tested
  contrasts.
- **README and index.** Human review complete, with its limits; Head A's CG-Bench result called a dev gain, not a
  confirmation; video length stated per benchmark; budget ledgers described as records that did not stop runs
  (25,373 calls charged against 20,000 declared). Experiments index: the meaning of "test" now names each set's freeze
  and reveal point; the speech-text row refers to the fresh-set follow-ups; the ceiling row labels the with-text and
  without-text conditions.
- **Next.** Supervisor material, then the CVPR paper draft, each sent to the auditor sentence by sentence.

## Step 104 (2026-11-09): supervisor brief and first CVPR 2027 draft
- **Supervisor brief.** `docs/supervisor/brief.md`: the problem, the systems in plain words, the frozen results on
  fresh LongVideoBench and on the lecture test, the loss analysis, the human check with its limits, what can and
  cannot be claimed, limitations and the publication plan. Every number maps to a ledger ID (source table at the
  end). Sent to the auditor before it is shared.
- **CVPR 2027 draft.** `paper/cvpr2027/` uses the official author kit (review mode, anonymous, eight-page limit). Style
  taken from recent CVPR frame-selection papers and from the closest related audit (bold run-in paragraph labels,
  findings and contributions listed in the introduction, null results stated as intervals). Framed as a controlled
  empirical study, not a new method: title "Does the Speech Point to the Slide? A Controlled Study of
  Transcript-Guided Frame Selection for Lecture Video QA".
  - Sections: introduction, related work (every work in auditor note P172), study design (systems, data, protocol,
    statistics), results (fresh LongVideoBench, lecture test, loss analysis, human check, cost), discussion and
    limitations, conclusion, LLM Usage Statement.
  - About 4,600 words of main text with four tables and one figure; the page count must be checked by compiling in
    Overleaf (no LaTeX engine on this machine).
  - References: titles and authors taken from arXiv and CVF Open Access, not from memory.
  - `paper/cvpr2027/CLAIMS.md` maps every numerical or comparative sentence to a ledger ID or source file; numbers
    not in the ledger are marked "new" for the auditor.
  - Open items for the owner: the consent and oversight wording for the human check, and confirmation of the LLM
    Usage Statement (the CVPR guidelines require it, and a paper without it may be desk-rejected).
- **Not done yet.** Supplement (earlier locked test, Head A dev results, ceilings, deviations, per-system inputs),
  due 23 November.
- **Update (same day): owner's answers.** Reviewer independence: the owner states that the three reviewers (thesis
  group members) did not see each other's answers and worked independently. Recorded as the owner's account in
  `reports/m3av/human_review_limits.md`, the brief and the draft; nothing in the collection method enforced it, and
  separate browser profiles or devices are not recorded. Consent and oversight are still open. LLM Usage Statement
  rewritten in the owner's framing (the authors conceived the idea, designed the framework and the experiment plan,
  directed and checked everything; Claude implemented the code, ran experiments and drafted the text, which the
  authors reviewed and edited; GPT double-checked independently; Nemotron generated the questions).
