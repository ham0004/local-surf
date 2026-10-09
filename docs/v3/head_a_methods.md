# Head A ("where to look"): method survey and chosen design

> **Correction (2026-10-09, same day):** in the project architecture, **Head A is the transcript path**:
> BM25 and related retrieval find speech close to the question, and Head A learns *where the useful
> picture is* relative to that speech (which lines, and how far before or after them). The whole-video
> MobileCLIP scorer and the "retrieve then verify" ranker described below belong to **Path B and Head B**
> (visual candidates; choosing the final frames from the pooled candidates of both paths). The research
> and the diagnostic remain valid; the roles are re-assigned in the section "Roles in the architecture".

Written 2026-10-09, after the Head A pilot (step 19) and before any further Head A result.

## Why Head A first
The gates located the largest measured opportunity in the framework: on CG-Bench's long videos the
frozen answerer scores 40.2% with MobileCLIP-chosen frames but 63.4% with frames from the human evidence
(+23 points, CI +11 to +36), while uniform frames and the transcript add nothing. On short lecture videos
(Video-MMMU) frames help but selection matters less (+4 over uniform, not significant). So improving
"where to look" in long videos is the change most likely to move final accuracy.

## How others build this component (primary sources read)

| Family | Representative work | What is learned | Supervision / cost | Fit for us |
|---|---|---|---|---|
| Training-free relevance + coverage | AKS (2502.21271), MarKey (2609.15408), FORTE (2610.00573), GIFT (2603.25072), Ground-Cover-Refine (2608.01660) | nothing | none | strong, cheap baselines; AKS measured here (no gain over MobileCLIP) |
| Answer-aware query | Query-aligned frame selection (2609.31668, Sep 2026) | nothing | none | published; our option-aware query (+5.4 train hit@4) is this idea, adopted as baseline |
| Temporal search / zoom | T* (CVPR 2025): VLM names target objects, open-vocabulary detector scans frame grids, zooms in | optional RL variant | LV-Haystack: 15,092 human keyframe annotations (mostly Ego4D, licence needed) | detector evidence targets entity questions, our largest gap |
| Learned temporal agent over frozen CLIP | TSPO (2508.04369): local window attention + residual over frame–text similarity, 3.5 M params | the agent | GRPO with VLM correctness rewards, 10 K examples, 8×A800 | closest to our pilot; too costly to train the same way here |
| Grounding / moment retrieval | UniVTG, Moment-DETR family; SeViLA (VLM pseudo-labels) | localiser | human spans or answerer feedback | supervision type we have (CG-Bench intervals) |
| Answer-utility selectors | Frame-Voyager, FrameOracle, ReFoCUS, M-LLM selector (2502.19680) | scorer / policy | answer loss, confidence, or ratings from a larger model | expensive labels; utility ≠ relevance |

## Diagnostic that decides the design (train videos only)
Option-aware MobileCLIP, top-K peaks ≥ 8 s apart, share of questions with a peak inside the human
evidence (`reports/v3_head_a/candidate_recall_train.json`, 373 questions, 37 videos):

| K | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| evidence recall | 36% | 46% | 58% | **71%** | 79% |

The cheap retriever usually *has* the evidence in a 32-candidate shortlist but ranks it too low. Learning
to re-rank a shortlist is a much easier problem than scoring a whole video from scratch (the pilot's
setting, which overfit), and it allows expensive, precise checks on only 32 frames.

## Chosen design: retrieve-then-verify Head A
1. **Retrieve** (cheap, whole video): option-aware MobileCLIP similarity every 2 s → top-32 candidate
   moments ≥ 8 s apart.
2. **Verify** (precise, 32 frames only), frozen signals per candidate, each aimed at a measured failure type:
   - fine image–text matching at full frame resolution with a stronger matcher than MobileCLIP's 256 px
     centre crop (catches small objects; entity questions);
   - on-screen text match: OCR of the candidate vs question and option words (text questions);
   - option discrimination: how differently the frame matches each answer option (evidence for the answer,
     not just relevance to the question);
   - context: the candidate's retrieval rank and score, its neighbours' scores, its position in the video.
3. **Rank** with a low-capacity listwise model (tens to a few hundred parameters) trained on CG-Bench human
   evidence (train videos only; scaled training set when available), choose the top-4 with the same
   separation rule.

What is borrowed: shortlist re-ranking (standard retrieval practice), detector/OCR evidence (T*, VSI), answer
options in the query (2609.31668). What is specific here and untested: a learned *evidence verifier* over a
small set of frozen, failure-targeted signals, trained on human evidence intervals, as the "where to look"
stage for a small frozen VLM under a 4-frame budget. Whether this is a contribution depends on results and a
final prior-work check; no novelty is claimed now.

## Test order and decision rules
1. **Oracle ceiling of re-ranking** (no model): hit@4 if the best 4 of the 32 candidates were chosen
   (≈ recall@32 = 71% vs 36% now). Measured on train.
2. **Signals one at a time** on train (offline hit@4 re-ranking the shortlist): keep a signal only if it
   improves train hit@4 with a CI above zero.
3. **Learned ranker** on train with inner video-level validation; one dev comparison against option-aware
   MobileCLIP; stop rule: CI must exclude 0.
4. **Online QA on dev** with the frozen VLM only if step 3 passes.
5. Cost: report the verifier's added time per question (the shortlist bounds it).

The scaled whole-video Head A (step 21) continues as declared; it shares the same training data and gives a
direct comparison between "score the whole video" and "retrieve then verify".


## Roles in the architecture (corrected) and the path diagnostic

Measured on CG-Bench train questions (373, 37 videos; `reports/v3_head_a/path_recall_train.json`):

| Candidate source | Reaches the human evidence |
|---|---|
| BM25 transcript windows (top 4) overlap the evidence | 55% |
| Path A proxy: top-6 speech lines by query-word overlap, line end within 10 s of the evidence | 38% |
| Path A proxy: top-16 lines | 54% |
| Path B: top-6 option-aware MobileCLIP peaks | 42% |
| Path B: top-16 peaks | 58% |
| **Pool A + B (6 + 6)** | **64%** |
| **Pool A + B (16 + 16)** | **80%** |

The two paths are complementary: their union covers far more evidence than either alone. This supports
the two-path design, and it fixes the jobs:

- **Head A (transcript path):** from the question and the timed transcript, propose the moments where the
  visual evidence is: which speech lines are relevant and the time offset of the picture relative to them
  (before, during or after; TAN found narration and picture are often not aligned). Baseline: query-word /
  BM25 line ranking (38% at 6 proposals) and v2's zero-shot MiniLM relevance. Target: raise Path A's
  evidence recall at a fixed proposal budget. Supervision: CG-Bench human intervals mapped to lines
  (positive lines and offsets), train videos only.
- **Path B:** visual scan with option-aware MobileCLIP (42% at 6).
- **Head B:** from the pooled candidates of both paths, choose the final K frames (the verify-and-rank
  design above: precise checks on a small pool, low-capacity ranker trained on human evidence).

Data limit to resolve: a transcript-based Head A needs English speech. Only 117 CG-Bench videos have English
subtitles (51 held); the 719 streamed training videos mostly lack them. Options: speech recognition on those
videos (only where speech is English), other lecture/long-video sources with evidence times, or EduVidQA's
question timestamps as weak positions. Decided before Head A training.
