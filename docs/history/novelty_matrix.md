# Novelty matrix (M0)

Status words: **Covered** = direct prior art exists, not a contribution.
**Uncertain** = no exact match found in the sources we checked; still needs full-method
comparison and experiments. **Engineering** = useful, but framed as a systems contribution.

Nothing below is a claim that something has "never been done". A finite search cannot show that.

| Component of our system | Closest prior work | Status |
|---|---|---|
| Transcript retrieval → candidate time windows | VSI, GCR, LongVideoAgent, Q-Gate | Covered |
| Small "brain" choosing where to look | LongVideo-R1, VideoAgent, A.I.R., HORNet, EviSelect | Covered |
| Cheap visual scout (CLIP-style similarity, scene change) | MobileCLIP-based selectors, FFS, Q-Frame, CSES | Covered |
| Cost/budget-aware reward or stopping | EviSelect (accuracy+efficiency reward), CSES, TRACE, BCEA | Covered |
| "Only look when vision is needed" routing | Caption-once Visual-Need Router, Q-Gate, F10 | Covered (by question type / post-hoc verifier) |
| Utility ≠ relevance labels from frozen answerer before/after | F10, Frame-Voyager, SeViLA, ReaSon | Covered |
| Contrastive decisive vs. misleading evidence | REVEAL (training-free, over evidence sets) | Covered as a pattern |
| Counterfactual intervention on **frames** | ReaSon | Covered |
| Robustness test with noisy/missing subtitles | VSI (training-free, untargeted) | Covered as an evaluation |
| **Paired training on the same (video, question) under CLEAN / TARGETED_DAMAGE / MATCHED_CONTROL_DAMAGE transcripts, supervising the controller's visual-acquisition decisions with measured utility and cost** | Nearest: F10 + REVEAL + VSI combined | **Uncertain** — no exact match found |
| Evaluation that separately reports over-spending under control damage | — | **Uncertain** (part of the same hypothesis) |
| Timestamped evidence ledger + full cost trace | Many agent papers report partial costs | Engineering |

## The one claim we will make (for now)

> We propose and **test** whether a small controller trained with paired transcript
> interventions — answer-relevant damage vs. an equal amount of unrelated damage on the same
> video and question — learns to acquire visual evidence when speech that matters is lost,
> without over-spending when unrelated speech is lost.

### What would falsify / shrink it

1. Full read of *Caption-once, Frames-on-Demand* (2609.11899) or F10 reveals paired transcript
   damage supervision → reposition as replication + video extension.
2. The trained controller does **not** beat an "unpaired" controller (same labels, no
   matched-control pairing) at equal cost → the pairing adds nothing; publish as a negative
   result / systems study.
3. The gain appears only under synthetic damage and vanishes under natural ASR errors.

### What the "matched control" must guarantee (checked by tests)

- Same number of transcript segments (or words) damaged as the targeted condition.
- Same damage *type* (deletion vs. substitution vs. timestamp shift).
- Control segments must not overlap the answer-relevant segments.
- The video and gold answer are unchanged.

## Evidence so far (2026-09-27)

LongVideoBench (252 videos, 330 fair triples) gave a **null** paired-vs-unpaired result, but it
neither supports nor falsifies the claim: measured on real labels, removing the answer-relevant
line did not make looking more valuable there (−0.022 [−0.050, +0.004]), so the condition the
hypothesis depends on is absent. Falsification criterion 2 above still needs a dataset where that
condition holds. See `reports/lvb_full/RESULTS.md`.
