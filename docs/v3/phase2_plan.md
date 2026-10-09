# v3 phase 2 plan: learned "where to look" (Head A) for long videos

Written 2026-10-09, after the two gates, before any Head A code or result.

## Why this, and not something else
- EduVidQA's synthetic split cannot show a frame effect (gate failed; `gate_g1_report.md`).
- Video-MMMU (lectures, ~8 min): frames help (+12 to +16 points); selection beats even spacing only
  slightly (+4.1, CI includes 0) because short videos leave little room (`gate_videommmu_report.md`).
- CG-Bench (long videos, median 39 min, evidence windows median 14 s): even spacing fails, MobileCLIP
  selection gains +24, and frames from the human evidence gain +48 — **+23 points of headroom over
  MobileCLIP** (`gate_cgbench_report.md`). CG-Bench also provides human evidence intervals that can
  supervise "where to look" directly.

So the most promising place for a learned component is Head A on long videos.

## Head A: question-conditioned temporal localisation over frozen features

- Inputs (all frozen, label-free at inference): MobileCLIP-S2 image embeddings every 2 s of the whole
  video; MobileCLIP text embedding of the question (and, as a variant, of each option).
- Model: projection + time encoding + a small transformer over the time axis, conditioned on the
  question (FiLM), giving one score per 2 s bin. Selection: top-K peaks at least 8 s apart (same rule as
  the MobileCLIP baseline), K = 4.
- Supervision: CG-Bench human clue intervals on TRAIN videos only (bins inside an interval are
  positives, softened by ±2 s); listwise cross-entropy over time.
- What is borrowed: temporal grounding / moment retrieval (e.g. Moment-DETR, UniVTG) and question-aware
  frame selection with learned scorers. What is specific here: trained and evaluated as the frame
  selector for a frozen small VLM answerer under a fixed 4-frame budget, with frozen CLIP features
  only (cheap at inference). No novelty is claimed before results and a prior-work check.

## Evaluation
- Offline (no answerer calls), on dev: **hit@4** = share of questions with at least one chosen frame
  inside a human clue interval; also mean distance to the nearest clue.
- Online, on dev: four-frame QA accuracy with the frozen Qwen3-VL-2B (same protocol as the gate).
- Baselines, same features and selection rule: uniform, MobileCLIP top-K, temporally smoothed MobileCLIP,
  AKS-style relevance + coverage; upper reference: frames in the human evidence.
- Final: CG-Bench test videos (6 videos, 55 questions; small) and Video-MMMU test two-thirds (transfer,
  lecture domain), each used once with frozen configurations.

## Data and scale (pilot before scale)
1. Pilot: the 51 English-subtitled videos we hold (train 37 videos / 373 questions). Train Head A,
   measure dev hit@4 against MobileCLIP. Stop if it does not beat MobileCLIP on dev.
2. Scale only if the pilot is promising: Head A needs no subtitles, so all 770 released CG-Bench videos
   (minus the 14 dev/test videos) can provide ~7,000 training questions with human intervals
   (~168 GB of archives; features are kept, videos can be deleted after extraction).

## Budget for this phase
- Feature extraction: CPU decoding + MobileCLIP on GPU; no answerer calls.
- Training: minutes per run on the local GPU.
- Online dev evaluation: ≤ 400 answer calls per compared method set.
