# Gate on CG-Bench: in long videos, where to look is almost everything

**Answer: frames help a lot, but only if they are the right ones.** On 82 dev questions (8 long videos,
median 39 min, human evidence windows median 14 s), the retrieved transcript and evenly spaced frames
add nothing over guessing, MobileCLIP-selected frames add +24 points, and frames taken from the
human-marked evidence add +48 points. A perfect frame selector would gain another **+23 points** over
MobileCLIP.

## Setup (declared before running: `scripts/v3_gate_cgbench.py`, commit `0ff0c4d`)
- Data: CG-Bench, English-subtitled videos we hold; dev split (8 videos, 82 questions; split declared
  in commit `633dd8a`). Multiple choice, 6–8 options (chance ≈ 14%).
- Transcript: v1/v2 retrieval (BM25 over 20 s units, top 4 windows, ≤ 300 words).
- Answerer: frozen Qwen3-VL-2B, multiple-choice letter prompt, exact letter scoring.
- Budget: 410 answer calls, 447 s (cap 420 calls, 3,600 s).
- E4 places 4 frames inside the human clue intervals. It uses gold evidence locations, so it is an upper
  reference for frame selection, not a method.

## Results (95% CIs: video-clustered bootstrap over only 8 videos, so intervals are wide)

| Condition | Accuracy | Difference |
|---|---|---|
| Q: question only | 15.9% | |
| T: retrieved transcript | 15.9% | T − Q = 0.0 (−4.9 to +5.1) |
| U4: + 4 evenly spaced frames | 17.1% | U4 − T = +1.2 (−6.0 to +9.7) |
| C4: + 4 MobileCLIP frames | 40.2% | C4 − T = **+24.4 (+14.9 to +32.5)** |
| E4: + 4 frames in the human evidence | **63.4%** | E4 − T = +47.6 (+31.3 to +63.5) |
| | | **E4 − C4 = +23.2 (+11.3 to +36.5)** |

By question type (small groups): text perception T 15% → C4 54% → E4 85% (13 questions); entity
perception 12% → 30% → 58% (33); event cognition 14% → 14% → 71% (7).

## Reading
- In these long videos the BM25-retrieved transcript does not help this answerer at all (many CG-Bench
  questions are about things seen, not said).
- Four evenly spaced frames almost never contain a 14 s evidence window in a 39-minute video; choosing
  frames by question–image similarity recovers about half of the achievable gain.
- The remaining +23 points is measurable headroom for a better "where to look" component (Head A). It is
  an upper reference measured with human evidence; a learned method will not reach all of it.
- Limits: 8 dev videos, one answerer, multiple-choice only, English-subtitled subset of CG-Bench.

Data: `reports/v3_gate_cgbench/` (answers, chosen frame times).
