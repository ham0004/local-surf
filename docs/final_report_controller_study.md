# Final report: speech-failure-aware visual acquisition controller (closed study)

Status: **closed on 2026-09-28**. The code is kept working and tested. The
research hypothesis was **not supported, and could not be tested, on the
data available**. Follow-up work (lecture-video QA) builds on this code but
is a separate study.

- Git: the last commit of branch `QAframework`. New work continues on
  branch `framework-2`, created from this commit. The revision work spans
  32 commits from `1ba3c59` (the audited snapshot).
- Tests: `195 passed` (`.venv/Scripts/python -m pytest -q`).
- Hardware: RTX 5060 Ti 16 GB, Windows 11, Python 3.12, torch 2.11 cu128,
  transformers 5.17.

## 1. What the framework is (kept as a working project)
Transcript → BM25 windows → candidate moments → lazy PyAV decode →
frozen MobileCLIP-S2 scout (5 signals) → controller (LOOK / EXPAND / STOP,
net = gain − λ·cost) → evidence packing → frozen Qwen3-VL-2B answer.

Usable entry points:
- `python -m videoqa.cli ask …` answers one question on one video
  (lazy path, metered cost).
- `python -m videoqa.cli evaluate …` runs every policy × condition
  independently and reports warm ms, frames and the primary endpoint.
- `python -m videoqa.cli profile …` gives a per-stage cost breakdown.

Measured costs (`configs/cost_model_rtx5060ti.yaml`): decode 43.3 ms/frame,
scout 46.8 ms/frame, prefill 0.363 ms/visual token (220 tokens per 640×360
frame), generation 38 ms/token. Warm time per query on LongVideoBench dev (11 q):
transcript-only 973 ms, uniform 1357, retrieval 1396, heuristic 1815,
scout-similarity 2718.

## 2. Hypothesis tested
Training the controller with *paired* labels (targeted speech damage vs.
severity-matched control damage on the same frame and history; losses V1/V2)
makes it better than pointwise training (V0) at recovering answers when
important speech fails, at equal measured cost.

## 3. What was implemented (by stage of the revision prompt)
1. Audit, correction note, regression tests (`docs/corrections/`,
   `scripts/audit_artifacts.py`).
2. Severity-matched damage with drop reasons, split provenance,
   answer traces with abstentions, and one CostModel everywhere.
3. Feasibility pilot with a support audit (§4).
4. Lazy selective search (`acquisition.py`), independent per-policy runs
   and fair timing.
5. Assay-v2 paired labels (`assay.py`). V0/V1/V2 heads: tiny MLP
   (`train_assay.py`) and Qwen3-0.6B + LoRA r=8 (`llm_controller.py`,
   1,147,905 trainable parameters, allowlisted text input `serialize.py`).
   λ calibrated on dev.
6. DAgger/RL: not started. Not justified without a testable signal.
7. Locked test evaluation: not run. The test split was never labelled.

## 4. Real-data results
LongVideoBench pilot (train 26 q / 12 videos, dev 14 q / 8 videos after
predeclared drops): 848 + 478 exact triplets, 0 pair drops. The
targeted−control gain is ≠ 0 in only 13% (train) / 4% (dev) of triplets.

| dev (mean of 3 seeds; LLM seed 0) | T−C rank acc (n=20) | look rate targeted / control | utility |
|---|---|---|---|
| MLP V0 / V1 / V2 | 0.37 / 0.48 / 0.62 | 0.60/0.58 · 0.77/0.77 · 0.88/0.89 | 0.060 / 0.057 / 0.075 |
| LLM V0 / V1 / V2 | 0.40 / 0.40 / 0.75 | 0.71/0.71 · 0.89/0.86 · 0.82/0.82 | 0.068 / 0.112 / 0.089 |
| oracle | – | 0.25 / 0.32 | 0.298 |

No variant looks more under targeted than under control damage. Even the
oracle does not.

Text-only screen for the "important-speech failure" stratum (right on the
clean transcript, wrong after targeted damage; `scripts/speech_stratum_check.py`):

| dataset | stratum rate | reason |
|---|---|---|
| LongVideoBench | 2/50 | answers are visual; subtitles are time anchors |
| Video-MME (subtitled) | 2/50 | mostly visual; 38% have no answer words in subtitles |
| EduVidQA (synthetic train) | 0/43 | reference answers not grounded in the speech at the asked time |
| TVQA val | 2/50 | speech informative (+0.18 over prior) but redundant across the clip |

## 5. Verdict on the hypothesis
**Not supported, and not testable on these data.** The primary endpoint
needs questions where losing a *local* piece of speech breaks the answer
and a frame can restore it. None of the four benchmarks provides enough such
questions (0–4%). The V2 ranking numbers above rest on 20 pairs and do not
count as evidence. No improvement or novelty is claimed.

A structural limit was also identified. The controller is a text model
that sees only the transcript and five scout numbers, so it cannot read
board or slide content. Its achievable gain over simple training-free rules
(scout top-k, a score threshold, answer-confidence escalation) is therefore
small. Closely related prior work: VideoAgent, VSI, VideoStir, 2607.05438,
CARGO-VL (see `docs/related_work_2026-09-28.md`).

## 6. Reproduce
```
.venv/Scripts/python -m pytest -q
python scripts/speech_stratum_check.py --dataset lvb --n 50 --out runs/stratum/lvb50.json
python -m videoqa.cli build-assay --data data/longvideobench_full --split train --config configs/gpu_12gb.yaml --relevance annotation --limit 40 --out runs/assay_v2/pilot_train
python -m videoqa.cli build-assay --data data/longvideobench_full --split dev   --config configs/gpu_12gb.yaml --relevance annotation --limit 20 --out runs/assay_v2/pilot_dev
python -m videoqa.cli train-assay --train-labels runs/assay_v2/pilot_train --dev-labels runs/assay_v2/pilot_dev --config configs/gpu_12gb.yaml --seeds 0,1,2 --out runs/assay_v2/mlp
python -m videoqa.cli train-llm   --train-labels runs/assay_v2/pilot_train --dev-labels runs/assay_v2/pilot_dev --config configs/gpu_12gb.yaml --variants V0,V1,V2 --seeds 0 --out runs/assay_v2/llm
```
Set `PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1` for GPU runs. data/, cache/
and runs/ are git-ignored; dataset manifests record sources and licences.

## 7. Limitations
Small pilot (40 questions labelled); one answerer and one scout; LLM
controller single seed; the free-form scorer (EduVidQA) is token F1, not an
audited rubric; YouTube blocked transcript fetching after 87 videos; the
locked test split was never evaluated.

## 8. What carries over to the next study
The lazy acquisition path, the measured cost model, the evaluation
harness, the stratum screen, the paired-label machinery (reusable for
utility labels on transcript windows), and the mined MIT OCW lecture data
(`scripts/fetch_lectures.py`, `mine_moments.py`, `verify_moments.py`;
pilot in `docs/mined_lectures_pilot_2026-09-28.md`).
