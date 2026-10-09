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
