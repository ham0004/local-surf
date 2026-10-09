# Gate on Video-MMMU: frames help (gate passes)

**Answer: yes.** On Video-MMMU's Perception questions (development third, 97 questions on 97 videos),
adding 4 frames to the transcript raises the frozen Qwen3-VL-2B's accuracy by 12–16 points with 95%
intervals above zero. This is the first benchmark in v3 on which frame selection *can* show an effect.

## Setup (declared before running: `scripts/v3_gate_videommmu.py`, commit `2104677`)
- Data: Video-MMMU Perception track, multiple choice (mostly 10 options), 89 OCR questions (text that
  must be read from the screen) and 8 ASR questions. Video-level split: this is the 1/3 dev part; the
  2/3 test part (202 questions) is untouched. One question was not run (97 of 98).
- Transcript: our Whisper large-v3-turbo ASR, word timestamps grouped into lines, full transcript capped
  at 2,000 words (declared protocol; the paper used a different Whisper setup).
- Answerer: frozen Qwen3-VL-2B, multiple-choice letter prompt, exact letter scoring, no judge.
- Budget: 388 answer calls, 1,552 s (cap 400 calls, 3,600 s).

## Results

| Condition | Accuracy | − transcript only (95% CI) |
|---|---|---|
| Q: question only | 28.9% | −11.3 (−21.6 to −2.1) |
| T: transcript only | 40.2% | — |
| U4: + 4 evenly spaced frames | 52.6% | **+12.4 (+2.1 to +22.7)** |
| C4: + 4 MobileCLIP-selected frames | **56.7%** | **+16.5 (+6.2 to +26.8)** |

- MobileCLIP selection vs even spacing: C4 − U4 = +4.1 (−3.1 to +12.4), not established.
- C4 vs T per question: 23 answers fixed by the frames, 7 broken.
- By type: OCR 38.2% (T) → 50.6% (U4) → 53.9% (C4); ASR (8 questions) 62.5% → 75.0% → 87.5%.
- Cost: median answer time 2.5 s with transcript only, 3.4 s with 4 frames (plus the MobileCLIP scan,
  not timed here).
- Question-only accuracy (28.9%) is well above the 10% chance level of 10 options, so some questions
  are partly guessable from the options; this affects all conditions equally (paired comparison).

## Reading
- Unlike EduVidQA's synthetic questions, these questions need the screen, and the answerer uses the
  frames. The gate passes for both frame placements.
- The selection gain over even spacing (+4.1) is small and uncertain on 97 questions. That gap, and the
  human-evidence reference measured on CG-Bench, decide how much room a learned selector has.
- Video-MMMU has no training split, so it serves as a development benchmark (dev third) and an untouched
  final test (test two thirds); training data for learned heads must come from elsewhere (CG-Bench train
  videos, or other sources).

Data: `reports/v3_gate_videommmu/` (summary, per-question answers, chosen frame times).
