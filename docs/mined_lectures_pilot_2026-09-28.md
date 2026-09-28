# Mined lecture questions: pilot (2026-09-28)

Source: 5 MIT OpenCourseWare lectures by Gilbert Strang (CC BY-NC-SA 4.0), from
the olive5/ml-lectures archive, with YouTube ASR captions ("en"), de-rolled.
Generator: Qwen3-VL-4B-Instruct @ebb281ec (Apache-2.0), greedy decoding,
frames at 896 px. Answerer: the frozen Qwen3-VL-2B @89644892, frames at 640 px.
All items are SYNTHETIC and unaudited.

Commands:
    python scripts/fetch_lectures.py --limit 5 --out data/mit_lectures
    python scripts/mine_moments.py --lectures data/mit_lectures --out runs/mined/pilot --limit-windows 40
    python scripts/verify_moments.py --lectures data/mit_lectures --mined runs/mined/pilot

## Funnel
| step | count |
|---|---|
| 30 s windows mined (40 per lecture, after skipping the OCW boilerplate) | 177 |
| generator says "none" / parse failure | 31 / 1 |
| rejected: not answerable by the generator from speech alone OR from the frame alone | 9 |
| rejected: grounding (speech quote not in captions, answer not in speech, duplicate options) | 4 |
| **kept** (measured relation: both 66, speech_only 52, visual_only 14) | **132** |

The generator's own claim of "both" (132 of 146) is unreliable. The measured
relation (the generator answers from one source at a time) is used instead.

## Frozen 2B answerer on the kept items
| relation | n | right with no evidence (prior) | speech only | frame only | speech + frame |
|---|---|---|---|---|---|
| both | 66 | 40 | 50 | 50 | 54 |
| speech_only | 52 | 17 | 24 | 13 | 19 |
| visual_only | 14 | 4 | 4 | 9 | 8 |

Strata that matter for the paired experiment (prior wrong, so the evidence matters):
- speech needed and sufficient (prior wrong, speech right): 21 of 132 (16%)
- of those, the frame also rescues them (frame alone right): 7 of 132 (5%)
- speech fails but the frame works: 15; speech works but the frame fails: 14

## Reading
- There are far more usable situations than in any benchmark screened (LongVideoBench / Video-MME / TVQA 4%, EduVidQA 0% for "speech right, damage wrong").
- Main loss: 46% of items are answerable from the question and options
  alone (the 2B model knows linear algebra). The next generator prompt must
  ask for lecture-specific facts (this example's numbers, this board's
  entries), and prior-answerable items are filtered by a rule declared now.
- A 640 px frame often makes board text illegible for the 2B answerer
  (14 speech-right items fail frame-only). This is a property of the frozen
  answerer and is kept, not tuned.
- Projection to all 106 MIT lectures (~100 windows each): ~10,600 windows,
  about 28 GPU hours at 9.5 s/window, which must run in resumable chunks.
  At pilot rates that gives ~1,700 speech-dependent and ~560 visually
  rescuable items before any prompt improvement.
- Label quality is unknown until a human audit (runs/mined/pilot/audit.jsonl
  has timestamps for every item).
