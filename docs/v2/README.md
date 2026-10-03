# Framework 2 (v2): learned evidence selection for long-video QA

v1 searched the transcript and then *chose frames by rule*. v2 asks a research
question on top of the same pipeline: **can small, cheap models learn which
evidence actually helps a frozen answerer, and is the speech or the frame at a
moment the useful part?** Everything heavy stays frozen; only two small heads are
trained.

## Pipeline

```
question + transcript                                        video
      │                                                        │
      ▼                                                        │
BM25 windows (v1 retrieval) ─── retained transcript (fixed, ≤120 words)
      │                                                        │
      ├────────────────────────────┐                           │
      ▼                            ▼                           │
PATH A  hot moments           PATH B  visual scan  ◄───────────┤
Head A scores each line       frames every 5 s in the windows, │
in the windows; top lines     one per stable slide/shot        │
give one frame each ◄─────────(perceptual hash), ranked by ◄───┘
      │                       MobileCLIP similarity
      └──────────┬─────────────────┘
                 ▼
   merged, de-duplicated candidate pool (≈10–12 frames)
   + one frozen pass per frame: MobileCLIP embedding, OCR text
                 ▼
   SELECTOR (one interface, several strategies)  →  K = 4 frames
                 ▼
   frozen Qwen3-VL-2B: retained transcript + chosen frames → answer
```

## Modules (`src/videoqa/v2/`)

| Module | Role | Main types / functions |
|---|---|---|
| `records.py` | Plain data shared by all stages (no logic) | `FrameCandidate`, `QuestionPool`, `UtilityRow` |
| `candidates.py` | Windows → Path A + Path B → merged pool; save/load pools | `build_pool`, `PoolConfig`, `save_pool`, `load_pool` |
| `encoders.py` | Frozen MobileCLIP embeddings + RapidOCR, with timing counters | `FrozenEncoders` |
| `head_a.py` | Transcript hot-moment scorer with two outputs: **text credit** and **frame credit** | `HotMomentScorer` (`feature_mode="small"` or `"full"`) |
| `head_b.py` | Frame scorer: utility(candidate given question, transcript, selected frames) | `featurize`, `HeadB`, `HeadBConfig` |
| `selectors.py` | Selection strategies behind one interface (Strategy pattern) | `SimilarityTopK`, `MMRSelector`, `UnaryUtility`, `GreedyUtility`, `ScoreTopK` |
| `teacher.py` | Frozen answerer as label source; exact-identity answer cache | `CachedTeacher`, `evidence_key` |
| `labeling.py` | Utility labels under explicit call budgets | `independent_labels`, `prefix_labels`, `head_a_labels` |
| `pilot.py` | The first controlled pilot (history-conditioned selection) | stages `pools/audit/label/eval/head-a` |
| `experiment.py` | Main experiment (speech-vs-frame credit; cost-matched selection) | stages `pools/label/eval` |

Design choices:
- **Stages depend only on `records.py`.** Each can be unit-tested with fakes
  (`tests/test_v2.py` uses a fake answerer and random embeddings).
- **Selectors share one interface.** Every arm receives the *same*
  `QuestionPool` and frame budget, so comparisons isolate the selection rule.
- **All teacher calls go through one cache keyed by everything that can
  change the answer** (question, options, transcript, exact pixels, model,
  prompt version). Call counts are therefore exact, and repeated evaluations
  are free.
- **Every run is resumable and charged to a persisted GPU-time budget**
  (`<run>/budget.json`). A stage stops cleanly when the budget is spent.
- **Labels never become inputs.** Gold answers and gains are only targets.

## Labels: what "utility" means here

`R(q, T, S) = 1` if the frozen answerer gets question *q* right from transcript
*T* and frames *S*. Labels are measured changes:

- **Head B:** `R(T, S + {frame}) − R(T, S)`. Positive, zero and negative gains
  are all kept.
- **Head A**, with separate interventions on an empty context:
  text `R({line}, {}) − R({}, {})` and frame `R({}, {frame}) − R({}, {})`.
  Line and frame are never added together, so neither takes the other's credit.

## Running

```
# data: lectures and mined questions
python scripts/research/fetch_lectures.py --limit 20 --out data/mit_lectures
python scripts/research/mine_moments.py --lectures data/mit_lectures --out runs/mined/v2 --limit-windows 32
python scripts/research/verify_moments.py --lectures data/mit_lectures --mined runs/mined/v2 \
       --qa-out data/mit_lectures/qa_v2.jsonl --drop-prior

# main experiment
python -m videoqa.v2.experiment pools --data data/mit_lectures --qa-file qa_v2.jsonl --run runs/v2_main
python -m videoqa.v2.experiment label --run runs/v2_main
python -m videoqa.v2.experiment eval  --run runs/v2_main
```

## Reports
- [pilot_report.md](pilot_report.md): pilot (history-conditioned selection). Not supported at pilot scale.
- [novelty_check.md](novelty_check.md): closest prior work and what is not claimed.
- `main_report.md`: main experiment (written when it completes).
