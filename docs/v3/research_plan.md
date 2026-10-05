# v3 research plan: learned temporal proposals, multimodal set selection, open-ended answers

Written 2026-10-05, before any v3 implementation, labelling or training. Follows
`FRAMEWORK_HANDOFF.md` and `FRAMEWORK_2_EVIDENCE_REVIEW_2026-10-05.md`.
Nothing here is a result. Every number marked *(measured)* comes from committed reports;
everything else is a plan or a hypothesis.

## 1. Old implementation vs new specification

| Area | v2 as implemented and tested | v3 target | Gap to close |
|---|---|---|---|
| Retrieval | BM25 over 20 s units, top 4 windows | keep; measure missed-evidence rate | measurement only |
| Path A | frozen MiniLM relevance; frame fixed at line end − 0.3 s (`candidates.py`) | **trainable Head A** predicting *where* (offset/subinterval) visual evidence lies relative to speech | new head + integration into candidate extraction |
| Path B | 5 s scan, cap 24, legacy or balanced; MobileCLIP top 6 | duration-proportional allocation under explicit scan budget | small change; baseline only |
| Selector | rules (relevance, MobileCLIP, MMR) or small aggregate-feature heads (≤ 8,333 params), no gain *(measured)* | **trainable Head B**: context-conditioned multimodal set selection over both paths | new representation + supervision |
| Transcript to answerer | fixed BM25 excerpt ≤ 120 words | budgeted verbatim speech around selected moments | policy change; isolate in an ablation |
| Answer | MC letter, accuracy | **open-ended** answer, audited quality scorer | scorer does not exist yet |
| Labels | single-frame / fixed-top-3 fourth-frame correctness, one option order | temporal targets + contextual set preferences from open-ended quality | no reuse of MC labels |

## 2. What the evidence so far says (and does not)

- Frames help over speech alone *(measured)*: LVB +9.1 points; MIT (rotation average) +15.5.
- No learned selector beat a label-free rule *(measured)*; the best rule depends on the data
  (relevance on chalkboard lectures, MobileCLIP/MMR on LVB).
- On lectures the relevance selector is effectively "Path A only": 7 of 788 picks were Path-B-only
  frames *(measured)*. The two paths were never truly combined by any selector.
- The fixed end-of-line Path A frame is a hand-chosen guess. TAN (Han et al., CVPR 2022) reports
  that only about 30% of HowTo100M narration sentences are visually alignable and about 15% are
  naturally well aligned, i.e. speech time is a weak proxy for picture time.
- These are MC results on biased or reused data; they do not measure the v3 system.

## 3. Benchmark decision (gate before any training)

| Candidate | For | Against | Status |
|---|---|---|---|
| **EduVidQA** (EMNLP 2025) | lecture domain; open-ended; 3,909 train / 1,056 synthetic test / 269 expert-verified real test; video-disjoint splits *(checked locally)* | authors' own appendix C.3.3: many "visual" questions are answerable from transcript or question alone; 0/296 videos local, transcripts for 87; official protocol gives the question timestamp (retrieval trivial) | **primary candidate, conditional on gate G1** |
| Neptune (DeepMind) | long videos, open-ended + GEM metric, CC-BY | general video; no subtitles documented (ASR needed); not lecture | fallback for open-ended |
| LongVideoBench | subtitles, long videos, existing pipeline | MC only; 440-question subset already used | secondary MC check only |
| Video-MMLU | lecture boards/slides, open-ended | clips of 10–240 s: little need for long-video selection | not suitable as primary |

**Gate G1 (frame dependence).** On a hash-chosen development sample of EduVidQA *training-split*
videos (never the real test), compare transcript-only vs transcript + frames with the audited
scorer. If frames add nothing measurable, EduVidQA cannot support frame-selection claims; it stays
an answer-quality benchmark and selection is studied on a benchmark where frames matter.

**Protocol.** Two named protocols, never mixed: *timestamp-given* (official; retrieval near-trivial)
and *timestamp-hidden* (whole lecture; our system must find the moment). Head A only matters in the
second.

**Gate G2 (scorer).** FactQA-style precision/recall against the reference (as in the EduVidQA
paper) with a fixed judge, validated by a hand audit of about 50 answers stratified by score;
report agreement and false-accept rate. The earlier Phi-4-mini judge was too lenient and is not
reused.

## 4. Shortlist (three hypotheses, one per component)

### H-A: speech-anchored offset head (proposed original design for Head A)
- **Problem:** Path A always takes the frame 0.3 s before a line ends. TAN's numbers say narration
  and picture are often offset or unrelated.
- **Mechanism:** for each transcript line in the windows, a small transformer over frozen line
  embeddings (question-conditioned, with neighbouring lines and time features) outputs
  (i) an *alignability* score and (ii) a distribution over offset bins (e.g. −30 s … +30 s, 2 s bins)
  for where the useful frame is. Proposals = top alignable lines × their most likely offsets.
- **Borrowed:** alignability + alignment prediction from TAN; temporal attention over text.
  **Changed:** question-conditioned; supervised by *answer utility* of the frozen VLM rather than
  narration–visual similarity; outputs proposals for a QA pipeline, not an alignment matrix.
- **Supervision:** on TRAIN videos only, decode frames at a fixed grid of offsets around candidate
  lines and score the frozen answerer's open-ended answer with each; the best-scoring offsets
  (with ties kept soft) are weak targets.
- **Baselines (same retrieval, same candidate budget):** line end (current), line start, middle,
  uniform in window, AKS-style relevance + coverage.
- **Ablation:** remove question conditioning; remove the alignability output.

### H-B: context-conditioned set scorer (Head B)
- **Problem:** existing heads score frames one at a time from aggregate features and never combine
  the paths.
- **Mechanism:** cross-attention from each remaining candidate to (question, its local verbatim
  speech, path/timestamp, the already-selected frames' features). Output: marginal value of adding
  the candidate. Greedy selection to K.
- **Supervision:** swap/add labels at *varied* selected contexts (not one fixed top-3), from
  open-ended answer quality; listwise/pairwise loss on reliable differences only.
- **Baselines:** relevance, MobileCLIP, MMR, AKS, and the v2 completion head.
- **Ablation:** no selected-set context (unary version); no speech input.

### H-S: operating choices (not novelty; needed for a fair baseline)
- Final K ∈ {2, 4, 8}; candidate budget per path; transcript budget; duration-proportional scan.
- Chosen on development videos by quality vs measured total cost, then frozen.

### Deferred, with reasons
- Encoder LoRA / answerer fine-tuning: changes attribution; only if a frozen encoder is shown to
  be the bottleneck.
- RL (GRPO/DPO): no reliable reward signal yet; supervised ranking is the first control.
- Per-question speech-vs-vision router: motivated by the lecture/LVB reversal, but overlaps with
  modality routing work (arXiv 2607.05438); revisit if H-B fails.

## 5. Order of work and stop rules

1. **CPU, no budget needed:** readiness tooling, open-ended scorer interface, Head A/Head B
   modules with tests, duration-proportional scan. *(next)*
2. **Data access:** check how many EduVidQA videos/transcripts can be fetched; report by split.
3. **G2 then G1** on a small development sample.
4. **Baseline cycle** (H-S) on development videos.
5. **H-A cycle**, then **H-B cycle**, each with its own development tuning and ablations.
6. **Frozen final comparison** of baseline, A-only, B-only, A+B on the same held-out questions.

Stop or report inconclusive if: G1 shows no frame dependence; G2 scorer agreement is poor;
weak labels are unstable across seeds/prompts; or a head cannot beat its baseline on development.

## 6. Budget proposal (requires your approval; nothing runs without it)

Every GPU stage runs under `CallBudget` (fresh calls AND seconds, whichever comes first).

| Stage | Fresh answerer calls | Judge calls | GPU time |
|---|---|---|---|
| G2 scorer audit (≈ 50 answers) | 50 | 50 | 0.2 h |
| G1 frame dependence (≈ 100 dev questions × 2 conditions) | 200 | 200 | 0.5 h |
| Baseline cycle (≈ 100 dev questions × ~6 settings) | 600 | 600 | 1 h |
| H-A weak labels (≈ 300 train questions × ~10 offsets) | 3,000 | 3,000 | 3 h |
| H-B set labels (≈ 300 train questions × ~10 contexts) | 3,000 | 3,000 | 3 h |
| Final comparison (269 real test × 4 systems) | 1,076 | 1,076 | 1 h |

Head training itself is small (minutes on the local RTX 5060 Ti). Judge calls depend on the judge
chosen (local model: GPU time; hosted: per-call cost).
