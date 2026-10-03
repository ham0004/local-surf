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
