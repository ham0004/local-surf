# Lecture study: what each system sees (frozen with the methods)

Answers auditor note P53 and the protocol's per-system table (`docs/v3/m3av_protocol.md`, amendment 3). Every
system gives the same answer model (Qwen3-VL-4B, same prompt) at most **4 frames** (K = 4). The exact settings and
the hashes of every file and checkpoint are in `docs/v3/m3av_freeze.json`. Dev figures are from M3AV dev (514
questions, 96 lectures).

Shared inputs:
- **Dense picture features:** MobileCLIP-S2 image embeddings of one frame every 2 s over the whole video, computed
  once per video. Used by S1, S1t, S2c, S2m, S3, R3, R4 and R5.
- **Transcript:** M3AV's timed sentences (the speech recognition text M3AV ships).

| System | How the frames are found (search work per question) | Final picker | Frames shown on dev (mean) | Transcript text shown | Encoders | Trained on |
|---|---|---|---|---|---|---|
| S0 starting design | BM25 over 20 s transcript units (question + options) → 4 windows. Speech path: MiniLM cross-encoder over the lines in the windows, 6 frames. Picture path: a frame every 5 s inside the windows (at most 24 decoded), 6 frames | top 4 by MobileCLIP similarity to the question | 3.94 | 120-word BM25 excerpt | MiniLM-L6 cross-encoder, MobileCLIP-S2 | nothing |
| S1 best framework | speech path as S0. Picture path: 3 frames from the window scan + 3 from the dense features over the whole video | MMR on MobileCLIP similarity to question + options (0.7 / 0.3) | 4.00 | none | as S0 | nothing (rules tuned on CG-Bench and Video-MMMU dev) |
| S1t | S1's frames | as S1 | 4.00 | up to 600 words: lines around the chosen frames + best lines by word and meaning search | as S0 | as S1 |
| S2c Head A (CG-Bench) | speech path: Head A over the whole transcript, 6 moments. Head A is a light residual on frozen MiniLM and bge-reranker scores, with density voting. Picture path as S1 | as S1 | 4.00 | none | MiniLM-L6, bge-reranker-base, MobileCLIP-S2 | CG-Bench train human clue intervals |
| S2m Head A (M3AV train) | as S2c, Head A retrained with the same recipe | as S1 | 4.00 | none | as S2c | 1,147 generated questions on 297 M3AV train lectures (evidence = the question's valid slide segments). No dev or test lecture |
| S3 Head B | S1's pool | Head B (`ev_set`, λ 2, μ 1): learned frame score + option-evidence set term | 4.00 | none | MobileCLIP-S2 | CG-Bench train pools |
| R2 | 4 evenly spaced frames, shown unmerged | n/a | 4.00 | none | none | nothing |
| R3 | dense features: the 96 times most similar to the question, at least 4 s apart, decoded. The first 4 distinct pictures in that order (perceptual hash differs) | n/a | 3.91 | none | MobileCLIP-S2 | nothing |
| R4 | as R3, each time scored by its best match over "question + option k" | n/a | 3.91 | none | MobileCLIP-S2 | nothing |
| R5 VSI-style | text branch as in VSI: all-mpnet-base-v2 line similarity to question + options, soft threshold, Gaussian spread (W = 2 s), maximum over lines. Visual branch: R4's score. Fusion 0.5 · z(text) + 0.5 · z(visual). Then as R3 (96 ranked, first 4 distinct) | n/a | 3.96 | none (the transcript is used only for search) | all-mpnet-base-v2, MobileCLIP-S2 | nothing (λ and the visual query chosen on dev) |

Notes:
- R3, R4 and R5 decode up to 96 frames per question, more than S0-S3 (at most about 30). They search more,
  not less.
- Frames shown below 4 happen when a talk has fewer than 4 distinct pictures in the searched range (dev: 13-31
  questions for R3-R5; 0-8 for the S systems).
- VSI's visual branch (a VLM proposes objects; YOLO-World detects them) is replaced by MobileCLIP, so R5 tests VSI's
  subtitle-plus-visual fusion idea, not the published system.
- What L1 and L2 claim: L1 compares the trained lecture locator with the strongest training-free rule on dev (R5);
  L2 compares it with the best framework (S1). Neither claims anything about answer accuracy; accuracy is secondary.
