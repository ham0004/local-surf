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
