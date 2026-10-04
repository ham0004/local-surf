# v2 novelty check: context-dependent frame-utility selector

Hypothesis under test: *a small selector trained on context-dependent evidence
utility chooses better frames than similarity ranking or independently scored
frames, under the same teacher-labelling budget and measured inference cost.*

Sources were read on 2026-10-03 (arXiv abstracts or HTML, plus the literature
report `Dual path video selection.md`). This is a bounded search. "Not found"
means not found, not "never done".

## Comparison with the closest prior work

| Work | Inputs | Conditions on already-selected frames? | Supervision | How labels are produced | What is trained | Reported cost |
|---|---|---|---|---|---|---|
| **Ross et al., ICML 2013**, Learning Policies for Contextual Submodular Prediction (1305.2532) | item features + current list | **Yes**: predicts the marginal benefit of the next item given the list | observed marginal gains | gains observed along rolled-out lists (no-regret / imitation) | a list policy | n/a (not video) |
| **Frame-Voyager**, ICLR 2025 (2410.03226) | query + candidate frames (bottom LLM layers) | Models frame-frame interaction, but **selects top-T at once** at inference | answer-model loss of frame **combinations** | enumerate combinations, rank by Video-LLM loss | LLM bottom layers / scorer | combinatorial labelling |
| **ReFoCUS**, CVPR 2026 Findings (2506.01274) | query + frame embeddings | **Yes**: autoregressive, masks selected frames | RL reward = frozen reference LMM's confidence margin | online policy gradient, no frame labels | Mamba-based selection policy | RL rollouts |
| **MarKey** (2609.15408) | query + frames + anchors | **Yes**: subset-aware greedy | none (training-free) | n/a: relevance + coverage + redundancy surrogate | nothing | training-free |
| **FORTE** (2610.00573) | query + frames | coverage gains updated after each pick | none (training-free) | n/a | nothing | training-free |
| **M-LLM frame selection**, CVPR 2025 (2502.19680) | SigLIP + Qwen2.5-1.5B | greedy + neighbour suppression (rule) | teacher **usefulness ratings** | larger model rates frames | scorer LLM | 1.5B scorer |
| **TSPO** (2508.04369) | frozen CLIP features | sampling policy over time | answer / localization **rewards** (RL) | rollouts with the frozen Video-MLLM | 3.5M temporal agent | 8×A800 training |
| **SeViLA**, NeurIPS 2023 (2305.06988) | frames + question | no | answerer-derived **pseudo-labels** for keyframes | answerer feedback | localizer (BLIP-2 based) | ≥3.1B |
| **R-VLM** (2312.04931) | CLIP chunk tokens + question | no | answer loss + soft-matching loss | end to end with the LLM | ~6M MLP/projector | end-to-end training |
| **VSI** (2508.06869) | subtitles + object detection | iterative sampling, not learned history | none (training-free) | n/a | nothing | 31.7 s search at 64 frames |
| **Q-Gate** (2604.17422) | 3 streams incl. subtitles | no | none; query-dependent GPT-4o weights | n/a | nothing | GPT-4o in the loop |
| **Modality Relevance ≠ Utility** (2607.05438) | draft answer + evidence | n/a (modality choice, not frames) | counterfactual keep/escalate outcomes | both outcomes observed per question | calibrated router | 72B answerer |

## Our mechanism vs. this table

What we do: a ~13k-parameter MLP over frozen MobileCLIP / OCR / transcript
features predicts **signed answer-utility gain** R(S+{c}) − R(S) of a frozen
Qwen3-VL-2B, **conditioned on the retained transcript and the selected-frame
history**. It is trained by **supervised** regression + within-context ranking
on labels from **prefix chains** whose evaluations are **cached and reused**,
and compared with single-frame labels at the **same teacher-call budget**.

Already covered by prior work, so we do **not** claim these as new:
- Predicting marginal benefit given the current selection, trained from gains
  along rolled-out lists: **Ross et al. 2013** (general setting).
- Frame selection conditioned on previously selected frames: **ReFoCUS** (RL),
  **MarKey** / **FORTE** (training-free).
- Learning frame selection from a frozen answer model's signal: **Frame-Voyager**,
  **TSPO**, **SeViLA**.
- Dual subtitle + visual streams: **VSI**, **Q-Gate**, HiMu.

Not found in this search (candidate distinctions, to be confirmed by results):
1. **Supervised, signed** (+1/0/−1) answer-utility labels, conditioned on **both
   the retained transcript and the selected frames**, as opposed to RL rewards
   (ReFoCUS, TSPO), relevance ratings (M-LLM) or training-free proxies (MarKey).
2. A **matched teacher-call budget comparison** of prefix-chain vs single-frame
   supervision for frame selection.
3. **Per-segment speech-vs-frame credit** (Head A's two separately intervened
   targets). Modality-bias studies measure this per dataset, not per segment
   as a learned selector target.
4. A fully local ≤4B stack with measured end-to-end latency on lecture video.

Verdict before any result: the history-conditioned selector is **not a new
algorithm**. At most it is a new *empirical* finding (does supervised
context-dependent utility beat similarity, diversity and independent labels at
equal cost?), plus items 1–4, which are narrow. Claims will follow the pilot's
numbers, not this table.

## Update after an adversarial audit (2026-10-03)

A second, adversarial literature audit (primary-source sections cited there)
looked for counterexamples. It adds prior art and
corrections, and it **withdraws the broad claims**.

Additional prior art:

| Work | What it establishes |
|---|---|
| **FrameOracle** (arXiv 2510.03584) | A lightweight frame selector that also predicts *how many* frames are needed. Curriculum from similarity proxies to validated minimal-sufficient keyframe labels; the audit reports leave-one-out downstream-loss changes as contextual importance targets (§4.2). Contextual answer-utility supervision for frame selection exists. |
| **Acquisition Conditioned Oracle**, ICML 2024 (2302.13960) | Learned acquisition policies imitating an expensive training-time oracle under acquisition cost. Replacing an expensive evaluator with a learned selector is not new. |
| **Sampling Permutations for Shapley Value Estimation** (2104.12199) | Marginal contributions from adjacent permutation prefixes. Prefix reuse is established estimation arithmetic. |
| ReaSon (2511.12530); multimodal moment-retrieval cascade (2512.12935) | Not exact matches (no per-prefix answer-gain head), but they strengthen the architecture precedent. |

Corrections to the table above: VSI also reports **8-frame** QA, so its 31.7 s
timing must not be used as an 8-frame comparison. R-VLM trains a visual
projector as well as its retrieval MLP. SeViLA fine-tunes the answerer in its
forward chain, and only its reverse chain uses a frozen answerer for labels.

**Withdrawn as claims:** a new controller-training principle; history-conditioned
selection; marginal-utility regression or ranking; prefix-label reuse; two-path
fusion; OCR fusion; two regression outputs. All of these are established.

**Exact method differences that remain (a combination, not an invention):**

| Element | Closest work | Difference that remains |
|---|---|---|
| Separate measured gains for a timed transcript line vs its nearby frame, against the same context | 2607.05438 (keep vs escalate per question) | Per-moment, temporal, raw-frame granularity |
| Conditioning on the transcript actually retained for the answerer | ReFoCUS / FrameOracle condition on query and frames | Retained speech as conditioning input |
| Signed per-addition gains on a changing selected set | Ross 2013; ReFoCUS (trajectory RL); FrameOracle (deletion from the full pool) | Addition to a variable prefix, supervised |
| A compact frozen-feature implementation on one 16 GB GPU with end-to-end latency | TSPO (3.5M agent, 8×A800), FrameOracle | Systems operating point, if measured |

**The one bounded, falsifiable question this project now tests:** *does the
retained speech change which additional frames are useful, and can a compact
learned scorer exploit that change better than strong baselines at equal cost?*
It is tested in two ways:
1. Directly from labels (`scripts/v2_transcript_conditioning.py`): agreement
   (Cohen's κ) between frame usefulness with and without the retained transcript.
2. Through selection: Head B with vs without speech features (`C_noocr` vs
   `C_noocr_notext`), with pixels, retained text, answerer, K, labels and
   parameter count fixed.

If the effect is absent, or explained by extra evidence or compute, that is the
result that gets reported.

## Fourth-frame completion (declared 2026-10-04, before its labels were collected)

Single-frame labels R(T, {c}) − R(T, {}) did not produce a better four-frame
selector (Head B, and the residual scorers in `review_20261004.md`). The next
test changes the **label target**, not the model size:

- The strongest simple baseline (zero-shot transcript relevance) fixes three
  frames. Every other pool candidate *c* is labelled with R(T, top-3 + {c}):
  four frames plus the retained transcript, exactly as deployed.
- Labels are exhaustive per question, so every fourth-frame policy (learned,
  CLIP, MMR, random, oracle) is scored **exactly offline** from one table.
- Budget: 1,697 answerer calls, vs 1,770 for the single-frame labels.
- Primary test: completion head − relevance rank-4 (lecture-bootstrap CI).

Closest prior work and what remains different:

| Work | Overlap | Difference that remains |
|---|---|---|
| Ross et al. 2013 | learn the marginal benefit of the next item given the list | not video; no retained-speech context |
| FrameOracle (2510.03584) | contextual importance from leave-one-out loss on the full pool | removal from the full pool vs addition to a fixed, deployed 3-frame set under the retained transcript |
| ReFoCUS (2506.01274) | history-conditioned selection | RL reward from a reference model vs exhaustive supervised completion labels |
| MarKey / FORTE / GIFT (2603.25072) | relevance + coverage + redundancy for the next frame | training-free surrogates; no measured answer outcome |
| Question-aware keyframes with synthetic supervision (2603.14953) | LMM-derived keyframe labels | labels from LMM rationales (cited timestamps), not measured answer change |

Specific, testable distinction: *measured, exhaustive, deployment-matched
completion utility for one slot, including negative effects of the retained
transcript and redundancy with chosen frames, plus an exact headroom (oracle)
for that slot.* This is an adaptation of contextual utility learning, not a
new algorithm. It counts as a contribution only if the completion head beats
both the baseline and single-frame-trained scorers on held-out lectures.

## Local board features for the fourth frame (declared 2026-10-04, before analysis)

The global completion head (26 scalar features) fell back to the baseline in
18/20 folds. Hypothesis: global features cannot see *where* question-relevant
content is on the board, or whether a candidate shows board content absent
from the chosen frames. Test, with zero new answerer calls, on the exhaustive
completion table:

- Representation only: 8 local features from frozen MobileCLIP on 8 square
  tiles covering the whole frame (no centre crop) plus chalk-ink maps against
  the three anchor frames (novel ink, novel ink in relevant tiles, local tile
  redundancy, camera misalignment). `src/videoqa/v2/local_features.py`.
- Same labels, loss, ridge grid, nested leave-one-lecture-out and fallback as
  the global head.
- **Primary:** global+local head − relevance rank-4. **Attribution:**
  global+local − global. Local-only head and single-feature zero-shot rules
  are reported as exploratory.
- Cost charged: 8 extra image encodings per candidate plus ink maps.

Closest work: TranSTR (ICCV 2023) learns spatio-temporal rationales end to
end; Visual Transcripts (SIGGRAPH Asia 2015) extracts board entities and aligns
them with speech; GIFT / MarKey score global irreplaceability and redundancy.
Difference under test: training-free local novelty relative to the frames
already chosen, conditioned on the question, used to pick the final frame for a
frozen answerer. If it does not beat relevance, the hypothesis is reported as
refuted, not renamed.
