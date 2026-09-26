# Progress log

## 2026-09-27 — session 2: first real dataset

### Built

- Hugging Face access confirmed for LongVideoBench (gated, CC-BY-NC-SA-4.0).
- `videoqa.sources.tar_range`: HTTP Range-based selective extraction from LongVideoBench's
  161.6 GB, 31-part video archive, avoiding a full download for a small subset. Retries transient
  network failures with backoff; persists a resumable, crash-safe directory catalog
  (`build_catalog`/`load_catalog`/`save_catalog`) so a full archive scan is never repeated.
- `videoqa.sources.longvideobench`: converts `lvb_val.json` + subtitle files into our `QAItem`/
  `Transcript` types. Extracts the exact subtitle span a T*-category question quotes
  (`extract_quoted_span` + fuzzy `find_evidence_interval`) to locate answer-relevant time
  intervals — the dataset-native analogue of our synthetic "spoken fact" questions.
- `scripts/download_longvideobench_subset.py`: end-to-end fetch + convert, resumable, with a
  dry-run mode.
- 21 new tests (tar_range, the adapter, including a real local-HTTP-server test that forces a
  mid-scan failure and checks correct resume).

### Real data obtained

40 real LongVideoBench videos (≤ 70 s each, 98 MB total), 40 real questions, real auto-generated
subtitles. See `docs/DATA_FEASIBILITY.md` for the full table. 23/40 questions have a
correctly-located, answer-relevant transcript span after the fix below.

### Bugs found against real data (each fixed with a regression test, each its own commit)

1. **A single network timeout killed a 40-video download after 1 file.** No retry logic existed.
   Fixed: exponential-backoff retries on transient failures in `RemoteMultipartTar.read()`.
2. **Selecting videos whose id happened to be absent from the archive silently turned into a full
   ~3,992-entry, ~2-hour scan** (searching for something that will never be found forces scanning
   to the true end). Fixed: excluded that id pattern (confirmed absent by the completed scan) and
   **[CORRECTED in session 3: wrong — those videos are stored under `video_path`, not `video_id`;
   all 753 val videos are present. The exclusion was removed.]**
   added a persisted, resumable catalog so this cost is paid at most once, ever.
3. **A second subtitle JSON shape** (`{"timestamp": [start, end], "text": ...}` for TikTok-sourced
   clips, vs. `{"start", "end", "line"}` for YouTube-sourced ones) raised `KeyError`. Fixed.
4. **Null end-timestamps** in that second shape (`{"timestamp": [23.0, None], ...}`) raised
   `TypeError`. Fixed: treated as a zero-duration point event; a null *start* is dropped (cannot
   be placed in time at all).
5. **Critical, found by manually inspecting one converted item's evidence interval (419.9s on a
   9.0s video):** LongVideoBench ships each short clip's subtitle file for the clip's *entire
   original source video*, not just the clip. Every one of the 40 downloaded videos had a nonzero
   `starting_timestamp_for_subtitles` (range 30–2871s). Uncorrected, every transcript-only
   baseline, retrieval window, damage assignment and answerer prompt would have silently included
   speech from parts of the source video never present in the file we have — a research-invalidating
   bug that would not have been visible from summary statistics alone. Fixed in
   `subtitles_to_transcript` (rebase by `offset_s`, drop/clamp by `clip_duration_s`); verified
   afterward that zero segments or evidence intervals exceed any video's real decoded duration
   across all 40 videos.

### What this means for trust in the pipeline

Bug 5 specifically was **not** caught by any test, type check, or automated pipeline stage — it
surfaced only because a human (well, an agent instructed to be suspicious) looked at one real
example's output and asked "does 419.9 seconds make sense for a 9-second video?" This is a
concrete argument for manually auditing a sample of any newly-integrated real dataset before
trusting pipeline output from it, exactly as `docs/DATA_FEASIBILITY.md` and the original research
plan both required.

### Bug 6: matched-control separation too strict for short clips

Running `build-labels` on the real subset first yielded only 5/25 train and 0/4 dev usable
questions (everything else hit `skipped_no_matched_control`). Diagnosis: `select_matched_control`
required a control segment to be >= 10s from the targeted one — a sane default for a lecture, but
almost impossible on a 9-70s clip with only 1-5 total transcript segments. Fixed with
`auto_min_distance_s`: scales the requirement to 15% of the transcript's own covered span (floor
1.0s, cap 10.0s), so a lecture-length fixture is unaffected (span large enough to hit the cap) but
a short clip gets a proportionally smaller requirement. Recovered usable triples from 7/40 to
24/40 (train 5->14, dev 0->2, test 2->7).

### First real pilot result (train 14 / dev 2 / test 7 questions — see caveats)

> **[INVALID — see session 3.]** The answerer received no transcript on 35/40 of these videos
> (BM25 IDF bug), so the damage conditions did not change its input at all. The numbers below
> are kept only as a record; `reports/pilot_lvb_real/INVALID.md` explains why.

Full tables: `reports/pilot_lvb_real/summary.md`. Checkpoints and manifests also archived there.

| | paired | unpaired |
|---|---|---|
| dev pair-difference MAE (predicting gain_targeted - gain_control on HELD-OUT questions) | 0.0199 | 0.0421 |
| test selectivity (targeted_response - control_overspend) | **+0.14** | **+0.00** |

The paired head predicts the held-out targeted-vs-control difference about twice as accurately as
the unpaired head, and on the test set it shows the hypothesized pattern (reacts to targeted
damage, does not overspend on control damage) while the unpaired head shows no reaction to either.
This is the first result in the direction the hypothesis predicts.

**This is not evidence, and must not be reported as such.** With 7 test questions, the paired vs.
unpaired frame-count difference under targeted damage has a bootstrap 95% CI of **[0.0, 0.43]**
(computed by the evaluation harness's own cluster bootstrap) — it does not exclude zero, and the
two heads produce byte-identical ANSWER QUALITY in every single condition (the extra frame changed
nothing about correctness here). Both heads' dev-tuned STOP threshold came out at the top of the
search grid (0.8, i.e. "almost never look"), which is itself a symptom of too little dev data (2
questions, 18 pairs) to calibrate a threshold meaningfully. Read this as: the plumbing produces a
directionally sensible signal on real data and did not break; it says nothing yet about whether
the paired-training hypothesis holds.

### Next

**Scale up the dataset.** The bottleneck that made the first LongVideoBench download slow (a
~2-hour archive scan) is now solved — the full 3,992-entry directory is cached locally, so
fetching more videos costs only their (small, ~2 MB average) download bandwidth. A run with
~150-300 usable questions (proportionally ~250-500 videos given the ~57% "no matched control /
no relevant segment" attrition rate measured here) is the minimum for the bootstrap intervals to
plausibly separate from zero. After that: multiple training seeds, a wider dev-tuning threshold
search, and only then treat any surviving effect as a claim worth writing up.

## 2026-09-26 — session 1

### Built (see `git log` for one commit per unit)

M0 audit docs → schemas → config → transcript → damage triples → costs → retrieval → frames →
fixtures → scout → packing → answerers (fixture double + Qwen3-VL-2B) → features → NumPy
utility head with paired loss → controller policies → pipeline + export → splits → labels →
training → evaluation → datasets → CLI → dataset audit → device profiles. 103 tests pass.

### Measured

| What | Result | Source |
|---|---|---|
| GPU warm / cold per question (RTX 5060 Ti, heuristic, 6 frames) | 1.18 s / 14.8 s; 5.45 GB peak VRAM allocated | `reports/profile_gpu_12gb_fixture.json` |
| CPU + real VLM (2 frames) | 10.5 s warm / 32.1 s cold; 8.8 GB RSS; **answer wrong** | `reports/profile_cpu_vlm_fixture.json` |
| Visual tokens per 640×360 frame (Qwen3-VL-2B) | 220 (counted) | answerer smoke run |
| Label generation cost with the real VLM | 1,327 answerer calls / 1,126 s for 18 train questions; 872 / 783 s for 12 dev questions | `reports/pilot_vlm_synthetic/labels_*_manifest.json` |

### Pilot: real frozen VLM, synthetic lectures (18 train / 12 dev / 18 test questions, 6 test videos)

Full table: `reports/pilot_vlm_synthetic/summary.md`. **This is a plumbing pilot on synthetic
data, not evidence for or against the hypothesis.**

| Policy | clean Q / frames | targeted Q / frames | control Q / frames | selectivity |
|---|---|---|---|---|
| transcript_only | 0.444 / 0 | 0.333 / 0 | 0.444 / 0 | 0 |
| uniform (6) | 1.000 / 6 | 1.000 / 6 | 1.000 / 6 | 0 |
| heuristic | 1.000 / 4.67 | 1.000 / 5.00 | 1.000 / 4.44 | +0.56 |
| learned, paired | 0.944 / 0.67 | 0.722 / 0.72 | 0.778 / 0.67 | +0.06 |
| learned, unpaired | 1.000 / 4.00 | 1.000 / 5.00 | 0.944 / 3.56 | +1.44 |

What it does and does not show:

1. **No evidence that the paired loss helps.** Dev pair-difference MAE: paired 0.339 vs
   unpaired 0.341. The large behavioural difference comes from the **STOP thresholds**
   tuned on only 12 dev questions (paired 0.50, unpaired 0.00) — a confound. Next: compare heads
   at matched thresholds / matched frame budgets, and tune on far more dev data.
2. **The synthetic task is too easy for vision**: uniform sampling with 6 frames scores 1.0 in
   every condition, so it cannot discriminate policies. Real data is required.
3. The unpaired learned head shows the *intended pattern* (more frames under targeted, fewer
   under control), but with 6 test videos the intervals are wide and this may not replicate.
4. Retrieval collapses under targeted damage (0 frames): BM25 finds nothing once answer-bearing
   lines are deleted — itself an illustration of why transcript-only gating needs rescue.

### Bugs found and fixed this session (each has a regression test)

* label answer-cache ignored the question id → answers shared across questions
* fixture spec registry silently fell back to the default lecture in a fresh process
* fixture answerer matched options as substrings ("red" in "covered")
* uniform baseline covered only the start of the video
* decode seek loop when the keyframe is far before the target

### Open problems (priority order)

1. **Decoding cost.** The scout requires decoding every candidate. Keyframe seeking helps
   short-GOP video (test: < 1/3 of frames), but on the fixture's 50 s GOP it *increased* decoded
   frames (≈ 423 → 500–680 per question in the pilot). Needs a GOP-aware rule (probe keyframe
   spacing once per video) and/or scout probes as metered, lazily decoded actions with a fixed
   probe budget.
2. **Real data.** LongVideoBench (gated, CC-BY-NC-SA, 161.7 GB) needs a Hugging Face login and
   an adapter; EduVidQA needs a decision on YouTube download terms and an audited free-form
   grader. See `docs/DATA_FEASIBILITY.md`.
3. **Fair paired-vs-unpaired comparison**: matched thresholds, more dev data, several seeds,
   cluster bootstrap over many videos.
4. **Label audit**: manually check a sample of real-VLM labels (teacher noise; the VLM answered
   a visual-only question wrongly with 0.95 confidence without the frame).
5. Text-LLM controller (Qwen3-0.6B + LoRA) — deferred until the numeric head is validated.
6. Natural ASR errors (faster-whisper vs reference subtitles) — `asr_style_noise` is synthetic.
7. `gpu_8gb.yaml` is unmeasured; Qwen3.5-2B and SmolVLM2 answerers not yet run.

### Caveat on the pilot artefacts

The train labels were produced before the seek-decoding change, dev labels and evaluation after
it. Served frames and timestamps are identical under both decoders (tested); only the
decode-time component of measured action cost differs.
