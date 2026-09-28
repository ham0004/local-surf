# Related work refresh (2026-09-28)

Scope of the check: work that (a) decides when to spend visual computation
versus relying on text or transcripts, (b) trains that decision from paired or
counterfactual evidence conditions, or (c) is cost-aware. I read the abstract
pages, not the full papers. **This is not a novelty claim.** Several of these
papers overlap with parts of our design, and a full-text read is still needed
before any claim.

| Paper | What overlaps with ours | What differs (per abstract) |
|---|---|---|
| CARGO-VL, arXiv 2608.04509 (Jiang et al., Aug 2026) | Counterfactual matched variants trained as one bundle (image-correct / text-correct / both / neither); a primal-dual controller trading accuracy against deferral | Arbitrates conflicting image vs. text evidence for a VLM answer; no transcripts, ASR or video frame acquisition mentioned |
| Modality Relevance is not Modality Utility, arXiv 2607.05438 (Li and Gai, Jul 2026) | Cost-aware escalation to a VLM only when needed; a calibrated value-of-escalation router; "relevance ≠ utility" matches our gain-based labels | Multimodal RAG on MultiModalQA; post-hoc verifier over a draft answer; oracle headroom rather than paired counterfactual training |
| Which Modality Decides? (CMA), arXiv 2608.00076 (Hasic et al., Jul/Aug 2026) | Counterfactual modality attribution | Analysis and audit only, not a controller; image and text only |
| CRAFT, arXiv 2605.19075 | Query-conditioned keyframes combined with ASR transcripts for video QA | A fixed pipeline, not a learned when-to-look controller under transcript damage |
| Adaptive Greedy Frame Selection, arXiv 2603.20180 | Question-conditioned frame selection with a text-only router | Selects a frame set; does not model the value of looking against transcript content |
| A.I.R., arXiv 2510.04428; HiMu, 2603.18558; query-based selection, 2601.07459 | CLIP-style query–frame similarity for selection | Always look; no stop decision driven by speech |

Consequence for this repository: *counterfactual paired training for a
controller* (CARGO-VL) and *cost-aware selective escalation* (2607.05438) both
exist. The remaining difference to test is narrower: pairs built by
**transcript** interventions (targeted vs. severity-matched control speech
damage) for a **when-to-look** controller over video frames under a measured
cost model. Whether that difference matters is an empirical question, and the
paired-training experiment is how we test it. The words "novel" or "first"
should not appear anywhere until the full texts are checked.

Sources: https://arxiv.org/abs/2608.04509, https://arxiv.org/abs/2607.05438,
https://arxiv.org/abs/2608.00076, https://arxiv.org/html/2605.19075,
https://arxiv.org/abs/2603.20180, https://arxiv.org/pdf/2510.04428,
https://pith.science/paper/2603.18558, https://arxiv.org/abs/2601.07459

## Closer reading for the dual-stream idea (transcript scorer + CLIP stream)
Read from arXiv HTML/abstract pages through a summariser, not a full manual
read; claims marked (?) need checking in the PDF.

| Paper | Transcript/subtitle use | Visual stream | Trained? | Answerer | Latency reported |
|---|---|---|---|---|---|
| VSI 2508.06869 | all-mpnet-base-v2 query–subtitle similarity + Gaussian spread around subtitle times | YOLO-World-110M object detection (target/cue objects) | no (plug-and-play) | GPT-4o | only relative (+9.7% time), FLOPs |
| VideoAgent 2403.10517 | none | uniform frames, captions by LaViLa / CogAgent-18B, CLIP retrieval | no (zero-shot) | GPT-4 agent with 3-level self-confidence stop | no |
| R-VLM 2312.04931 | none | MLP over CLIP chunk tokens, soft-matching loss | yes, end-to-end with the LLM | LLM | 2.56 s per 60 s video |
| VideoStir 2604.05418 | abstract: "without relying on auxiliary information" (ASR use unclear (?)) | clip-level spatio-temporal graph + MLLM intent scorer on frames, IR-600K data | yes (scorer) | MLLM | (?) |

Not found in these four: (a) a TRAINED scorer over transcript segments
supervised by the frozen answerer's utility; (b) a fully local ≤4B stack
with absolute wall-clock on a consumer GPU; (c) lecture videos, where speech
carries answers. VSI's LongVideoBench gains come from its text-relevant
subset, where (as measured here) subtitles serve as temporal anchors. A
dual-stream design is still VSI-like; any claim must be differential
against VSI as a baseline.

## Full-text reading (PDFs in cache/papers/, text via pypdf), 2026-09-28
- VSI 2508.06869: training-free. The subtitle branch is all-mpnet-base-v2
  cosine with a soft threshold (θ=0.5, γ=2) and a Gaussian spread (W=2 s).
  The visual branch is YOLO-World-110M on target/cue objects that a VLM
  (GPT-4o in their figure) lists first, with iterative spline-updated
  sampling (~26 iterations). Search latency is 31.7 s per question at 64
  frames (Table 3). Answerers: GPT-4o, LLaVA-Video-7B, Qwen2.5-VL-7B. There
  is no OCR and no lecture data. Their subtitle gains concentrate on the
  LongVideoBench text-referred subsets (subtitle as a temporal anchor).
  Removing subtitles entirely changes F1 only from 83.7 to 83.2 (Table 6).
  Code is public.
- VideoStir 2604.05418: no speech or subtitles ("native input"). The frame
  scorer is Qwen2.5-VL-3B with LoRA, distilled from Qwen2.5-VL-72B
  RELEVANCE levels 1–5 (IR-600K), on 8×A100. Latency is listed as a
  limitation and not reported.
- VideoAgent 2403.10517: GPT-4 agent, CogAgent-18B/LaViLa captions,
  EVA-CLIP-8B retrieval, 3-level self-confidence stop, zero-shot. No speech.
- R-VLM 2312.04931: 1.8 M MLP on CLIP text features selects 4 s visual
  chunks, trained end-to-end with a soft-matching loss. No speech.
- 2607.05438: relevance ≠ utility for modality escalation (MultiModalQA,
  text+table vs image). Post-draft calibrated value router trained on
  counterfactual keep/escalate labels. The verifier predicts relevance
  (F1 0.87) but not utility (F1 0.32). Utility oracle: 12% escalation. The
  answerer is 72B and latency 4.49 s. Not video, not speech.
- 2608.03161: lecture→knowledge graph (Faster-Whisper, EasyOCR,
  Qwen2.5-VL-7B). Anchors combine visual change, transcript keywords and
  first mention. 3 lectures, 3 seed questions. No QA benchmark, no
  conflict, no cost.
- CARGO-VL 2608.04509: image vs RETRIEVED TEXT conflict bundles (A/V/T/N)
  on TextVQA/ScienceQA stills. Qwen3.5-9B trained with GRPO over bundles and
  primal–dual abstention. No video, speech, lectures or acquisition cost.

Gaps across all seven: lecture video QA where speech carries answers; a
transcript-window localiser trained on answer UTILITY (the 2607.05438
insight applied to temporal speech); board-text-aware cheap frame scoring;
speech-vs-board conflict (CARGO-VL's A/V/T/N transposed to temporal speech,
including ASR errors); end-to-end local ≤4B latency. Each piece is
incremental on its own; the combination is the candidate contribution.
