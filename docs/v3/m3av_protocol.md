# Lecture study on M3AV: protocol (declared before any question is generated)

Declared 2026-11-01 (research log step 79). Answers auditor notes P40–P41 of 2026-11-01. Changes after this date are
appended at the end with a date and a reason; nothing above that line is edited.

## Question
On lectures by speakers never seen in training, does a question-conditioned frame selector put one of its 4 frames
on the slide that answers the question more often than strong training-free rules, with the same frame budget?

## Data
- M3AV v1.0 (CC BY-NC-SA 4.0; source videos keep their own rights), official speaker-disjoint split: train 901,
  dev 99, test 113 lectures. Slide segments come from `ocr/seg_speech_ocr.json` (start, end, OCR text, speech);
  the saved slide image is the **last** frame of its segment.
- Videos: dev and test only, 360p. Dev lecture CHI-36F5D is unavailable (YouTube sign-in) and is excluded; test is
  complete (113 of 113).
- Train lectures (no video) supply questions and speech for training Head A only.

## Questions
1. **Candidate slides.** A segment is eligible when it lasts at least 4 s and its OCR has at least 8 distinct words.
   The first segment of a lecture (title slide) is not used.
2. **Generation.** Nemotron-3.5-Super-VL-120B (NVIDIA API, temperature 0) sees the slide image and its OCR text,
   never the speech or the times. It writes one multiple-choice question with 4 options and returns the exact slide
   words that support the answer (`evidence_text`). It is told: answerable from this slide's visible content; not
   about the title, authors, affiliations or slide numbers; not answerable from general knowledge. Prompt and
   settings are saved with the questions.
3. **Sampling.** Per lecture, eligible slides are taken in a fixed pseudo-random order (SHA-256 of
   `m3av-protocol:<lecture>:<segment start>`) until the quota is met: train 4 questions on 300 lectures chosen by
   the same hash; dev up to 6 per lecture; test up to 10 per lecture.
4. **Automatic filters**, in this order, with the number removed by each reported per split and lecture:
   - F1 format: 4 distinct non-empty options, key in range.
   - F2 grounding: `evidence_text` matches the slide OCR (at least 80% of its words present).
   - F3 occurrences: every segment of the lecture whose OCR contains `evidence_text` (same 80% rule) is a valid
     evidence segment, so a repeated slide is not scored as a miss.
   - F4 visibility (dev and test): within each valid segment, frames at 1 frame per second (360p) are compared with
     the saved slide image (both reduced to 64 × 36 grey, normalised correlation). The **visible span** is the
     seconds whose correlation is at least τ. τ is chosen on dev before test is generated, by looking at 20 dev
     segments with slide builds, and is written here as an amendment. A question whose total visible span is under
     2 s is dropped. This is a proxy for "the slide as saved is on screen"; it does not prove that the specific
     fact is legible.
   - No answer-model filter. Two labels are recorded for every question and used only in secondary analyses:
     `q_only` (Qwen3-VL-4B answers correctly from the question and options alone) and `t_only` (Qwen3-VL-4B answers
     correctly without pictures from 600 transcript words chosen by BM25 over the whole transcript).
5. **Freeze.** The test cohort file (questions, options, keys, evidence text, valid segments, visible spans, filter
   outcomes, generator prompt and settings, and this protocol's τ) is written once; its SHA-256 is recorded in the
   research log before any system runs on test.

## Measures
- **Primary: source-slide interval hit@4**: 1 when at least one of the 4 frames given to the answer model lies inside
  a visible span of a valid evidence segment (1 s tolerance), else 0. Named this way because the visible span is a
  proxy, not checked fact visibility.
- Secondary: multiple-choice accuracy with Qwen3-VL-4B (same prompt as all earlier work), on all questions and on
  the subsets without `q_only` and without `t_only` (these subsets depend on Qwen and are labelled so).
- Uncertainty: paired bootstrap over **lectures** (10,000 resamples, seed 20261101); questions and lecture counts are
  both reported.

## Systems (K = 4, the same answer model and prompt)
S0 starting design; S1 best framework; S1t best framework + speech text; S2c Head A as trained on CG-Bench; **S2m Head
A retrained on M3AV train lectures** (speech lines → evidence segments, same architecture and training recipe; no
dev or test lecture used); S3 Head B; R2 4 evenly spaced frames; R3 dense MobileCLIP top-4; R4 dense option-aware
MobileCLIP top-4; R5 a VSI-style subtitle + visual search (both paths, Gaussian spread of subtitle scores). Settings
are chosen on dev only. Every method and setting is frozen (configs and checkpoints hashed in the research log)
before the test cohort is generated.

## Declared contrasts (primary measure, Holm over both)
- **L1:** S2m − R3 (the trained lecture locator against the strongest training-free frame rule found so far).
- **L2:** S2m − S1 (against the best framework).
All other comparisons are secondary and uncorrected.

## Question review
- **Audit sample:** 100 test questions, one per lecture from 100 of the 113 lectures (lectures and questions picked
  by SHA-256 of `m3av-review:<qa_id>`), drawn after the freeze.
- **Packet per question** (no system names, outputs or results): question, options, key, `evidence_text`, the saved
  slide image at full resolution, 3 frames from the visible span at 360p (the evaluation resolution), the proposed
  visible span, the transcript lines inside it, and the other valid occurrences.
- **Reviewers:** the builder and the GPT auditor, each alone, without seeing the other's verdicts, record for each
  question: accept / reject / uncertain; key correct; one best answer; supported by the slide; fact legible at 360p
  in the proposed span; shortcut (general knowledge, transcript alone); a reason code and a short note. The builder
  reviews only after the method freeze, so nothing from the review can feed back into a method. A person later
  reviews the same packet in random order, also blind to systems and results.
- **Analyses:** primary = the whole frozen cohort. Secondary 1 = the cohort without questions rejected in the
  review sample, with the same mask for every system (sample results are also reported with their rejection rate,
  which estimates the cohort's error rate). Corrected keys or spans, if any, are a separate analysis. Neither
  secondary analysis is promoted over the primary.

## Amendments
- 2026-11-01, before any generation beyond a 2-lecture trial (dev CHI-003EC, CHI-051B1; 16 calls): the 4 options of
  every question are put in a fixed order given by SHA-256 of `m3av-options:<qa_id>:<k>`, so the position of the
  correct option does not follow the generator's habits; the generator's own position is kept as
  `generated_answer_index`. Generation uses at most 3 × quota slides per lecture; the quota counts questions that
  pass F1 and F2.
- 2026-11-01, amendment 2, on dev only, before any test question exists (F4 measure and τ). The declared measure
  (64 × 36 grey correlation) was looked at on 20 dev segments with builds (`reports/m3av/sheets_grey_set1/`) and could
  not work with any τ: a different slide with the same layout scored up to 0.97 (dark-room NIH talks), while the
  right slide under burned-in subtitles scored 0.58. F4 is therefore: **edge-map correlation** (grey 256 × 144,
  Gaussian blur 1 px, absolute horizontal + vertical differences, normalised) between the frame and the saved
  slide, **τ = 0.85**, and the frame must match its own saved slide at least as well as the saved slides of the
  previous and next segments (a neighbour that is itself a valid evidence segment of the question, or has no saved
  image, is not compared). How τ was set: 0.6 was proposed from set 1 and failed on a second unseen set of 20
  (`sheets_edge_set2/`: a slide M3AV never saved scored 0.80-0.82); 0.85 was set from sets 1-2 and checked on a
  third unseen set of 20 (`sheets_edge_set3/`, 100 frames judged by eye): different slide, speaker or intro shown,
  22 frames, 0 counted visible; complete saved slide, 62 frames, 62 counted visible; earlier build stage of the
  slide (part not yet shown), 16 frames, 8 counted visible. So a visible span can include a build stage whose missing
  part may hold the fact; the question review's "fact legible in the proposed span" item measures this. Heavy
  burned-in subtitles (correlation about 0.72) take those seconds out of the span (conservative: a frame on another
  slide is not credited). Dev result: 542 → 517 questions (25 under 2 s visible), 96 lectures.
- 2026-11-01, amendment 3, on dev only, before any test question exists (auditor P48, P52, P53).
  - **F4 spans.** Each 1-per-second sample stands for [t, t + 1 s), clipped at the segment end. Only consecutive
    passing samples no more than 1.5 s apart join one interval, so a failing sample, or a gap in the decoded frames,
    always breaks a span. A question's visible spans are the union of these intervals over its valid segments, and
    its visible time is their total length. The first implementation joined every passing second of a segment into
    one span from the first to the last and so could credit non-visible gaps. Dev after the fix: 54 of 514 questions
    have split spans, total visible time falls 9%, and 514 questions remain (28 have less than 2 s).
  - **Dev only.** Sheets are built from dev lectures only, and `apply` reads only the requested split's curves.
  - **F3 answer content.** A repeated occurrence counts as evidence only if its OCR also contains the answer's
    critical tokens. These are the correct option's words that appear on the source slide, plus every token with a
    digit in `evidence_text`; they are kept as `critical_tokens`. When the key is fully paraphrased (4 of 542 dev
    questions), only the 80% rule applies. Dev: 20 questions lose 29 of 780 occurrences.
  - **Calibration record.** The builder's per-frame judgments for all three sheet sets, and the tally computed
    from them, are in `reports/m3av/visibility_calibration.{csv,md}` (`scripts/m3av_visibility_calibration.py`).
    Corrected set-3 counts: complete slide 61 of 61 counted visible, build stage 8 of 17 (amendment 2 said 62 and
    16: one frame was miscounted by hand). They are judgments on selected dev frames, not population error rates.
  - **Review sample, two stages.** (1) The lectures with at least one question in the frozen test cohort are
    ordered by SHA-256 of `m3av-review-lecture:<lecture>`, regardless of their question counts, and the first 100
    are taken. If fewer than 100 such lectures exist, a further pass in the same order takes a second question from
    each lecture, then a third, and so on, until there are 100 questions. (2) Within a lecture, questions are taken
    in the order of SHA-256 of `m3av-review:<qa_id>`. Two estimates are reported: the lecture-balanced rejection rate
    (unweighted mean over sampled lectures) and the cohort's question-weighted rate, Σ n_l r_l / Σ n_l, where n_l
    is the lecture's number of cohort questions (inclusion probability ∝ 1 / n_l). Removing the rejected sampled
    questions while keeping unreviewed ones is a sensitivity analysis, not a validation of the kept questions.
  - **Per-system table.** At the method freeze, a table is written for every system: candidate pool and search
    work, frames shown, transcript words and tokens shown, the encoders used, and the inputs taken from training
    data. L1 and L2 claim only what they compare; R4 and R5 are shown in the same table as the main results.
- 2026-11-01, amendment 4, on dev only, before any test question exists (dev results in `reports/m3av/hits_dev.md`,
  research log step 86; per-system inputs in `docs/v3/m3av_systems.md`).
  - **Fixed-K rules show 4 distinct pictures.** The first R2-R5 pools held only 4 candidates, and the pool's
    merging of repeated pictures (a slide stays on screen for minutes) left R3 with 1.4 frames on average (hit@4
    46.7). Final versions: R3, R4 and R5 rank 96 times (at least 4 s apart) by their own score and show the first 4
    pictures that differ (perceptual-hash distance > 0, chosen on dev over 3: better for all three rules). R2
    shows its 4 evenly spaced frames unmerged. Dev frames shown: 3.91-4.00.
  - **R4** scores each 2 s frame by its best MobileCLIP match over "question + option k". **R5** keeps VSI's text
    branch (all-mpnet-base-v2, soft threshold θ 0.5 and γ 2, threshold 0.2, Gaussian spread W 2 s) and uses R4's
    score as the visual branch (VSI's object detector is not used). Fusion λ = 0.5, chosen on dev from {0, 0.25,
    0.5, 0.75, 1} × {question, best option} visual queries.
  - **L1 changed to S2m − R5** (was S2m − R3). R3 was declared as the strongest training-free rule found so far;
    on dev it is the weakest of R3-R5 (62.6 vs R4 75.1 and R5 76.1). L1 now uses the strongest training-free rule
    on dev. L2 is unchanged (S2m − S1); Holm over L1 and L2. S2m − R3, S2m − R4 and S2m − S2c are secondary.
  - **S2m settings.** Head A retrained with the deployed recipe unchanged (light_dens, ens, human, width 32, lr
    1e-3, 15 epochs, 3 seeds averaged). Picker kept as MMR (dev 77.0; Head A's own top 4: 76.7).
  - **Correction to amendment 3:** the split-span count is 58 of 514 questions (a valid segment with at least two
    visible spans), not 54; the 9% fall in visible time combines the span fix (−4.1%) and the F3 answer-content rule.
  - Dev results (primary measure, 514 questions, 96 lectures): S2m − R5 +1.0 (−2.9..+4.7); S2m − S1 +6.8
    (+3.1..+10.7); S2m − R3 +14.4; S2m − R4 +1.9 (−2.9..+6.5); S2m − S2c −0.2 (−2.5..+2.2).
