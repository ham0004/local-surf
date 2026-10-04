# Residual probe

CPU-only nested video CV on existing random single-frame labels. These are proxy ranking metrics, not four-frame QA accuracy. No teacher or encoder calls. Full-pool selections are exported for later exact evaluation.

Completed nested CV. 197 questions, 20 videos, 1573 single-frame labels.

- clip: mean selected single-frame correctness 0.2513; pairwise concordance 0.5061
- mmr: mean selected single-frame correctness 0.2538; pairwise concordance 0.5149
- relevance: mean selected single-frame correctness 0.2716; pairwise concordance 0.5903
- residual_base: mean selected single-frame correctness 0.2792; pairwise concordance 0.6409
- residual_context: mean selected single-frame correctness 0.2652; pairwise concordance 0.5957

These exploratory results carry no novelty or performance promise.
