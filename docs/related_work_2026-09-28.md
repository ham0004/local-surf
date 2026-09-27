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
