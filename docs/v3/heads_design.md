# Head A and Head B: proposed original designs (v3)

Written 2026-10-09, before implementation, following `FRAMEWORK_HANDOFF.md` (Head A: learn *where* visual
evidence is from timed speech; Head B: choose a frame *set* from both paths using question, image
features, nearby speech, provenance and the already-selected frames). Supersedes the whole-video visual
scorer of steps 18–21, which is re-assigned to Path B. No novelty is claimed before results and a final
prior-work check; each design lists its closest work and the ablation that tests its new part.

## Measurements the designs build on
- Long videos (CG-Bench): frames from the human evidence lift the frozen answerer from 40.2% (MobileCLIP
  frames) to 63.4%; uniform frames and the transcript add nothing.
- Candidate coverage on CG-Bench train: speech-line proposals reach the evidence for 38% of questions (6
  proposals), MobileCLIP for 42% (6), **the pool of both for 64% (6 + 6)**. The paths are complementary.
- Video-MMMU (lectures): frames add +12 to +16 points; selection beats uniform only slightly.
- Narration and picture are often offset in time (TAN, CVPR 2022); v2 always took the frame 0.3 s before
  a line ends.

---

## Head A — "speech-to-sight offset localiser" (proposed)

**Problem it solves.** Path A currently assumes the picture is at the end of the matching speech line.
The speech that matches a question often comes *before* the thing is shown ("now let me draw the graph")
or *after* it ("as you saw, the curve…"). Head A predicts both *which* lines are relevant and *where the
picture is relative to them*.

**Inputs (frozen encoders, label-free at inference):** the question and options; the timed transcript
lines inside a broad BM25 window set (with ±2 neighbouring lines as context); line times.

**Outputs:** for each line i, a relevance p_i, and an **offset distribution** q_i(Δ) over time bins
Δ ∈ [−40 s, +40 s] (4 s bins) giving where the visual evidence lies relative to the line. Proposals are the
top (line, offset) pairs by p_i · q_i(Δ), with temporal non-maximum suppression, so one question can get
several moments.

**Model:** frozen text encoder for question and lines; a small cross-attention block (question attends to
the line and its neighbours) and two heads (relevance, offset bins). A few hundred thousand parameters.

**Supervision:**
- CG-Bench human clue intervals (train videos with English speech): a line is positive if the evidence
  lies within ±40 s; its offset target is the evidence centre minus the line time.
- EduVidQA training questions (3,909, English transcripts after the language filter): the question
  timestamp is a weak evidence position (paper: ~35 s average error), used as weak positives with a wide
  offset target; never the 269 real test questions.

**What is new here (to be tested):** question-conditioned prediction of a *speech-to-visual offset
distribution* per transcript line, used as the proposal mechanism of the speech path. Closest work:
TAN (predicts whether narration is visually alignable and aligns it, but not conditioned on a question and
not for QA proposals); moment retrieval (Moment-DETR, UniVTG: spans from video features); VSI and
Ground-Cover-Refine (use subtitles as anchors, no learned offsets). **Ablation:** offset fixed at 0 (only
line relevance) vs learned offsets.

**Metric:** Path A evidence recall at a fixed proposal budget (6 moments): baseline 38% (query-word lines),
v2's zero-shot MiniLM relevance, BM25 windows.

---

## Head B — "answer-discriminative evidence set selector" (proposed)

**Problem it solves.** Choosing the top-K most relevant frames picks redundant views of the same moment and
frames that match the question but do not separate the answer options. What the answerer needs is a small
*set* of frames that together make one option clearly better supported than the others.

**Inputs per pooled candidate (from Path A and Path B):** MobileCLIP image embedding; its similarity to the
question and to **each answer option** (the option-evidence profile); Head A's score and offset; nearby
verbatim speech embedding; path provenance (A, B, both); time. Plus the set already selected.

**Model:** a small set model: each candidate attends to the already-selected frames (cross-attention, one
layer, width 64) and outputs a marginal score; greedy selection up to K.

**The new ingredient — option-evidence profile and set decisiveness:** for a set S, combine the
candidates' option-similarity profiles into an option distribution; reward sets whose evidence is
*decisive* (separates the options) and *complementary* (each frame adds evidence for options the set does
not already cover), not merely relevant to the question.

**Supervision (two stages):**
1. Human evidence (CG-Bench train): candidates inside a clue interval are positives (relevance pre-training).
2. Answerer set labels (the handoff's contextual set targets): on train questions, swap one frame of a
   selected set with another candidate and record whether the frozen answerer's correctness changes;
   learn pairwise preferences between sets. Ties are not converted into preferences.

**What is new here (to be tested):** a learned set selector driven by per-option evidence profiles
(decisiveness and complementarity across answer options), trained with human evidence plus answerer swap
labels. Closest work: answer-aware query expansion (2609.31668: options appended to the query, no set
reasoning); MarKey / FORTE / GIFT (set-aware but training-free, relevance and coverage only); Frame-Voyager
and ReFoCUS (learned set selection from answer loss or RL, without per-option evidence profiles).
**Ablations:** remove the option profile; remove set context (score frames independently).

---

## Experiment order (the handoff's sequence; corrected 2026-10-09, step 25)
0. Benchmarks and scorer fixed: CG-Bench (dev 8 videos / 82 questions, test 6 / 55) and Video-MMMU (dev 98,
   test 202 per track), multiple-choice accuracy plus open-ended FactQA where references exist (CG-Bench).
   The baseline cycle below comes BEFORE any head is trained.
1. Baseline cycle on dev: path budgets (A and B proposals), K, transcript budget, with uniform, MobileCLIP,
   option-aware MobileCLIP, AKS and the v2 pipeline as baselines.
2. Head A cycle: build, train, tune on train/inner validation, evaluate Path A recall on dev; then pooled
   coverage and QA with the baseline Head B (option-aware MobileCLIP ranking).
3. Head B cycle: build, train (stage 1, then stage 2), tune, evaluate QA on dev with the baseline Path A.
4. Combined A + B; ablations of each new part.
5. Final frozen comparison of all methods on the same untouched test sets: CG-Bench test (6 videos,
   55 questions; small) and Video-MMMU test (202 + 202 questions), plus an external comparator.

All runs budgeted and logged in `docs/v3/research_log.md`.
