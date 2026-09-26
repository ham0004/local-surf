# Data feasibility (measured 2026-09-26)

Every number below comes from `scripts/audit_datasets.py` → `reports/data_audit.json`
(live GitHub / Hugging Face API calls). **No video was downloaded.** Re-run the script before
relying on these numbers.

```
uv run python scripts/audit_datasets.py --out reports/data_audit.json
```

## Summary table

| Source | Questions (measured) | Videos | Answer type | Timed transcript? | Temporal evidence? | Access / license (measured) | Fit for us |
|---|---|---|---|---|---|---|---|
| **EduVidQA** (EMNLP 2025) | real_world_test 269; synthetic_train 3,909; synthetic_test 1,056 | 99 / 157 / 40 (YouTube ids) | long-form free text | **No** — README recommends `youtube-transcript-api` | one `timestamp` per question (e.g. `2:06:36`), no interval | Repo MIT (code/annotations). Videos are third-party YouTube content — **not** covered by the repo license. oEmbed: 27/30 real-world ids respond, 3/30 HTTP 401 | Lecture domain ✔. Open-ended answers need audited grading (see below). |
| **NExT-GQA** | train 34,132; val 3,358; test 5,553 | 3,870 / 567 / 990 | 5-way multiple choice | No (everyday clips, little speech) | **Yes**: `gsub_*.json` intervals per question (val/test) | Repo MIT; videos come from VidOR/NExT-QA (separate terms) | Clean MC scoring + grounding. Not lectures, little speech → weak for the transcript-damage hypothesis. |
| **QVHighlights** | (val sample checked; 3 real rows in report) | ~150 s YouTube clips | query → moments (not QA) | No | **Yes**: `relevant_windows`, saliency | Repo MIT; YouTube-derived features/videos | Retrieval/grounding pre-training only. |
| **LongVideoBench** | parquet val/test (counts not yet read) | — | MC | **Yes**: `subtitles.tar` | referring queries | HF dataset **gated ("auto")**, license **CC-BY-NC-SA-4.0**, **161.7 GB** | Best long-video + subtitle fit; needs HF login + non-commercial use. |
| **Video-MME** | 1 parquet (counts not yet read) | — | MC | **Yes**: `subtitle.zip` | no | HF `lmms-lab/Video-MME` not gated, no license field on card, **101.0 GB**; repo has no license | General-video generalisation test; clarify terms before use. |

## Real rows (verbatim excerpts from the report)

EduVidQA `real_world_test.csv` — columns `url, id, question, answer, timestamp`:
```
id=F9-yqoS7b8w  timestamp=2:06:36
question: "Is it necessary to dereference the pointers name and number when using the %s format specifier in printf, ..."
answer:   "When using the %s format specifier in functions like printf, the corresponding argument is expected to be a pointer ..."
```
EduVidQA `synthetic_train.csv` — columns `vid_id, vid_url, vid_title, timestamp, final_answer, final_question`:
```
vid_id=8jqKOqwOwA4  vid_title=mod07lec31  timestamp=3:54
final_question: "At <timestamp>, can you explain how the Fixed Universe Successor problem differs from ..."
```
NExT-GQA `val.csv` — columns `video_id, frame_count, width, height, question, answer, qid, type, a0..a4`:
```
video_id=4882821564  type=CW  question="why did the boy pick up one present from the group of them and move to the sofa"
answer="unwrap it"  options=[share with the girl, approach lady sitting there, unwrap it, playing with toy train, gesture something]
gsub_val[10001787725] = {"duration": 34, "location": {"1": [[1.2, 5.8]], "3": [[12.1, 17.1], ...]}, "fps": 29.97}
```
QVHighlights `highlight_val_release.jsonl`:
```
qid=2579  vid=NUsG9BgSes0_210.0_360.0  duration=150
query="A girl and her mother cooked while talking with each other on facetime."  relevant_windows=[[82, 150]]
```

## What no source provides (and we generate)

None of these datasets contains our **action-usefulness labels** (answer quality before/after
looking at a specific moment, under clean / targeted / control transcripts). Those are produced
automatically by `videoqa build-labels` on TRAIN videos only. This is a design fact, not a gap
in the audit.

## Decisions (provisional, pending access checks)

| Role | Source | Why | Blocking issue |
|---|---|---|---|
| Plumbing / CI | synthetic lectures (`videoqa make-synthetic`) | fully licensed, deterministic | not research evidence |
| Controller training (pilot) | **LongVideoBench val** (subtitles + MC) | timed subtitles + objective MC scoring | HF login + gated access; NC license; 161.7 GB → download a subset |
| Lecture-domain evaluation | **EduVidQA real_world_test** | real student questions on lectures | YouTube download terms; no transcripts shipped; free-form answers need an audited grader; only one timestamp per question |
| Grounding diagnostics | NExT-GQA val/test | real evidence intervals | little speech → transcript damage mostly irrelevant |
| Held-out generalisation | Video-MME (subtitle setting) | broad domains | license/terms to clarify |

Splits: always by original video (course when known) **before** creating transcript variants
(`videoqa.splits`). Official test sets are touched once, at the end.

## Open-answer grading (EduVidQA)

EduVidQA answers are long-form, so `answer_quality` falls back to normalised exact match / token
F1, which is **not** a reliable correctness measure for explanations. Before EduVidQA labels are
used for training, we need: a written rubric, an automatic grader, and a manual audit of a
representative sample (≥ 50 items, two annotators). Until then, EduVidQA is evaluation-only
and reported separately.

## LongVideoBench: real subset downloaded (2026-09-27)

Access granted (Hugging Face login + accepted dataset terms). 40 real videos, 40 real
multiple-choice questions, real (often noisy, auto-generated) subtitles downloaded via
`scripts/download_longvideobench_subset.py` — see [docs/progress.md](progress.md) for the full
account of what broke and was fixed along the way. Summary of the final, corrected dataset:

| | Value |
|---|---|
| Videos | 40, all duration ≤ 70 s, all with a T*-category (subtitle-quoting) question |
| Total video size | 98 MB |
| Questions | 40 (one per video in this pass) |
| Questions with a correctly-located answer-relevant transcript span | 23 / 40 (mean fuzzy-match ratio 0.80) |
| Splits (video-level hash) | train 25, dev 4, calibration 1, test 10 |
| License | CC-BY-NC-SA-4.0 (non-commercial), recorded per item |

**Critical bug found and fixed before any experiment touched this data:** LongVideoBench trims
short clips out of longer source videos but ships each clip's *full original* subtitle track
under the same id — a 9.0s clip arrived with subtitle text timestamped up to 553s. Every one of
the 40 videos had this offset (30–2871s). Uncorrected, transcripts would have contained speech
never actually present in the clip, invalidating every downstream measurement. Fixed in
`videoqa.sources.longvideobench.subtitles_to_transcript` (rebases + clips timestamps using the
record's `starting_timestamp_for_subtitles` and `duration` fields); verified afterward that zero
transcript segments or evidence intervals exceed any video's real decoded duration.

Two other real format quirks were found and fixed along the way: a second subtitle JSON shape for
TikTok-sourced clips, and null end-timestamps in that shape. The dataset's TikTok-sourced videos
("@username-..." ids) are excluded from selection — a full scan of the video archive confirmed
they are not present in it (only YouTube-sourced videos are).

The download is resumable and caches the archive's directory listing locally, so scaling from 40
to more videos does not repeat the (slow, ~1–2 hour) full-archive scan.

## Next concrete steps

1. ~~Authenticate to Hugging Face, accept LongVideoBench terms, download a small video subset.~~ **Done.**
2. Run the controller pipeline (`build-labels` → `train-controller` → `evaluate`) on this real
   40-video subset as a pilot, same as was done on synthetic data.
3. Decide whether to scale the subset up (more videos, and/or raise `--max-duration` past 70s)
   before or after seeing pilot results.
4. For EduVidQA, decide with the project owner whether downloading the YouTube videos is
   permissible for this research; if yes, fetch transcripts and record per-video availability.
