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
