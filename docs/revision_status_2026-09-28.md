# Revision status (2026-09-28)

This follows the order of the revision plan. Each stage states
what was implemented, what was measured, and what is blocked. Starting point
was 1ba3c59 (148 tests); see docs/corrections/2026-09-28.md for the audit.

## Stage 1: audit, correction note, regression tests (done)
- scripts/audit_artifacts.py reproduces every historical label count exactly.
- The correction note documents the real defects (visual_change neighbour
  dependence, per-condition pools, unequal-damage triples) and withdraws the
  overstated claims.

## Stage 2: matching, identities, provenance, traces, costs (done)
- Damage matching now works on effective changed tokens, with a predeclared 20% tolerance
  and no-op rejection (damage.build_triple, with drop reasons).
- official_split and experiment_split are separate fields; test labelling is refused; there is a
  media-fingerprint overlap check (assay.check_media_overlap).
- Answer traces now keep abstentions (they are no longer forced to option A),
  validate citations, count tokens exactly before generation (verified 440 = 440 on GPU),
  and store raw text in eval records and assay rows.
- One CostModel is used in labels, inference and calibration. Its costs were measured on the
  RTX 5060 Ti (configs/cost_model_rtx5060ti.yaml): decode 43.3 ms/frame,
  scout 46.8 ms/frame, prefill 0.363 ms/visual token (220 tokens per 640x360 frame),
  generation 38 ms/token.

## Stage 4: genuine selective search (done)
- acquisition.py: text preparation decodes nothing; frames and scout calls
  are made lazily and metered; each policy pays only for its own scout seed.
- Evaluation runs every (policy, condition) independently, rotates policy order,
  and reports warm_ms separately from model load.
- **Measured** (12 LongVideoBench dev questions, 11 with a fair triple, lazy path,
  mean warm ms per query on the clean condition):

  | policy | decoded frames | scouted frames | warm ms |
  |---|---|---|---|
  | transcript_only | 0 | 0 | 973 |
  | uniform | 424 | 0 | 1357 |
  | retrieval | 328 | 0 | 1396 |
  | heuristic | 584 | 4 | 1815 |
  | scout_similarity | 1600 | 15.9 | 2718 |

  Accuracy on these 11 questions is 0.09–0.27 for every policy. The new transcript-only
  answers match the historical labels on all 11 questions, so this is not a regression.
  The sample is far too small for any accuracy comparison.

## Stage 3: real-data feasibility pilot (running; see pilot section)
- **Blocker, measured.** On LongVideoBench the "important-speech failure"
  stratum (transcript-only right on clean, wrong on targeted) holds 2 of 208 historical
  train questions and 2 of 48 dev questions. The quoted subtitle is a temporal
  anchor; the answer is visual. The primary endpoint cannot be powered on this
  dataset. Data in which the answer is spoken is needed (e.g. EduVidQA, which
  requires downloading YouTube videos and is the user's decision), or a synthetic spoken-answer set
  that needs a human audit.

## Stage 5: controllers (implemented; pilot training pending labels)
- assay.py: common candidate pool, exact canonical keys, shared histories,
  EXPAND excluded from visual pairs. On the fixture: 93 exact triplets, 0 pair drops.
- train_assay.py: MLP V0/V1/V2 on identical rows and seeds; decision lambda calibrated on
  dev at a declared price; Pareto curve.
- llm_controller.py: Qwen3-0.6B (rev c1899de2) + LoRA r=8 on q/v projections,
  1.15 M trainable parameters, allowlisted text input (serialize.py). Fixture GPU smoke test:
  17 ms per scored action (batched). Training peak was 13.5 GB before gradient
  checkpointing, measured while sharing the GPU with labelling.

## Literature
docs/related_work_2026-09-28.md. CARGO-VL (counterfactual bundle training +
controller) and 2607.05438 (cost-aware modality escalation) overlap
substantially. No novelty claim is made.
