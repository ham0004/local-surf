# The starting framework for v3 (baseline), and what the baseline cycle tunes

Written 2026-10-10. This is the v2 two-path pipeline as already implemented (`src/videoqa/v2/candidates.py`,
`selectors.py`), run on the new benchmarks with **rules only** (no trained head). Head A and Head B later
replace exactly one box each; everything else stays as frozen here after the baseline cycle.

```
question + options + timed transcript                                video
          |                                                             |
   [1] BM25 retrieval: transcript split into 20 s units, top 4 windows  |
          |        (+1 neighbour unit each side)                        |
          |------------------------------------------------.           |
          v                                                 v           |
 [2] PATH A (speech)                           [3] PATH B (vision)  <---|
     MiniLM cross-encoder scores each line         scan frames inside the windows
     in the windows (zero-shot);                   (every 5 s, cap 24; "legacy" or
     top 6 lines -> 1 frame each, taken            "balanced" spread); one frame per
     0.3 s before the line ends                    run of near-identical frames;
          |                                        MobileCLIP ranks them; keep top 6
          |                                             |
          '------------------.     .--------------------'
                             v     v
         [4] CANDIDATE POOL (~10-12 frames): merge, de-duplicate
             (same pixels hash or within 1 s), MobileCLIP embedding per frame
                             |
         [5] SELECTOR -> K frames (rule): MobileCLIP top-K | MMR (similarity + diversity)
             | transcript relevance; frames >= minimum gap apart
                             |
         [6] TRANSCRIPT for the answerer: BM25 excerpt (<= 120 words in v2)
                             |
         [7] FROZEN ANSWERER: Qwen3-VL-2B (K frames + excerpt + question) -> answer
```

Where the heads go later (not in the baseline):
- **Head A replaces box [2]'s rule** "top lines, frame at line end" with learned moments (which lines, and how
  far before/after them the picture is).
- **Head B replaces box [5]'s rule** with a learned set selector over the pool.

## Benchmarks and metrics (same for every method)
- CG-Bench (long videos): dev 8 videos / 82 questions; test 6 / 55 (locked).
- Video-MMMU (lectures): dev 98 questions per track; test 202 per track (locked).
- Metrics: multiple-choice accuracy; open-ended FactQA on CG-Bench (reference answers available; validated
  judge); evidence recall of the pool (CG-Bench human intervals); measured time per question.

## Baseline cycle: settings tuned on dev only, then frozen

| Setting | Values tried | Why |
|---|---|---|
| Search scope of the paths | BM25 windows only (v2) vs Path B also scanning the whole video at a coarse step | BM25 windows contain the CG-Bench evidence only 55% of the time; the handoff asks to measure this retrieval limit openly, not hide it inside a head |
| Path B sampling | legacy, balanced, duration-proportional; step 2/5 s | coverage of long windows |
| Candidates per path | A 0/3/6, B 0/6/12 | how much each path contributes |
| Selector rule | MobileCLIP top-K, option-aware MobileCLIP, MMR, relevance, AKS | strongest rule becomes the bar to beat |
| Final frames K | 2, 4, 8 | quality vs cost |
| Transcript to answerer | none, 120, 300 words, verbatim around chosen moments | the handoff's transcript policy |

Procedure: change one setting at a time from the v2 defaults, keep a change only if it helps on dev, then
test the combination; all trials logged with their numbers. The chosen configuration is frozen as **the
baseline** that Head A and Head B must beat.

## Frozen baseline (2026-10-10, after the baseline cycle; research_log steps 27–36)

| Box | Frozen setting |
|---|---|
| [1] retrieval | BM25 over 20 s units, question + options as query, top 4 windows (+1 neighbour) — unchanged |
| [2] Path A | MiniLM zero-shot line relevance inside the windows, 6 lines, frame at line end − 0.3 s — unchanged |
| [3] Path B | **hybrid**: best 3 frames of the scan inside the windows + best 3 of the top-24 dense whole-video MobileCLIP times (2 s features, ≥ 4 s apart); one per stable run |
| [4] pool | merge + de-duplicate (hash ≤ 3 or ≤ 1 s) — unchanged |
| [5] selector | **MMR (0.7 relevance / 0.3 redundancy) on MobileCLIP similarity to question + options** |
| K | **4** |
| [6] transcript to the answerer | **none** (multiple choice and open-ended both scored best without the excerpt) |
| [7] answerer | frozen Qwen3-VL-2B (unchanged) |

Dev results of the frozen baseline (v2 default in brackets): CG-Bench multiple choice **40.2%** (28.0),
paired +12.2 (+4.6..+21.5); Video-MMMU Perception **63.3%** (56.1), +7.1 (−1.0..+15.3); CG-Bench open-ended
(judge agreement with hand grading 40/40 on a sample) **19.5%** (13.4), +6.1 (−1.3..+13.3); CG-Bench pool
evidence recall **60%** (33%). Head A replaces box [2]'s rule, Head B box [5]'s rule; everything else stays
as above for every later comparison, including the final test.
