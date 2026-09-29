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

## LongVideoBench: real data (updated 2026-09-27, session 3 audit)

Access granted (Hugging Face login + accepted dataset terms). `scripts/download_longvideobench_subset.py`
fetches videos straight out of the 161.6 GB archive by byte range, using a complete, cached
directory of the archive (3,992 entries). Measured eligibility (has a T*-category question that
quotes a subtitle, and a subtitle file):

| | Value |
|---|---|
| Eligible videos | **252** (all durations; 128 are <= 70 s) |
| T*-category questions on them | **440** |
| ... with a locatable quote | 360 |
| ... with the quote located in the clip's own transcript | **352** |
| Total video size | 9.22 GB |
| License | CC-BY-NC-SA-4.0 (non-commercial), recorded per item |

Real-data pitfalls found and fixed (details in `docs/progress.md`, each with a regression test):

1. **Subtitle timeline offset.** Short clips ship their *source* video's full subtitle track;
   times must be re-based by `starting_timestamp_for_subtitles` and clipped to `duration`.
2. **Archive names use `video_path`, not `video_id`.** The 135 TikTok-sourced videos (id
   `@user-<n>`) are stored as `videos/<n>.mp4`. An earlier version looked them up by id, failed,
   and wrongly concluded they were absent.
3. **Two subtitle JSON shapes** and **null timestamps** in the TikTok shape.
4. **Quote extraction**: apostrophes taken as quote marks (29/440 questions), one-letter quotes
   swallowing the next quote, only the first of several quotes used, long lines diluting the
   fuzzy-match score.
5. **Retrieval on tiny transcripts**: most clips form a single retrieval unit, where the
   `rank_bm25` IDF goes negative and nothing is retrieved — the answerer then got no transcript.

**Relevance for damage.** Gold answers here are visual ("eggs"), so only the dataset-native
quote-anchored evidence is used to decide what "answer-relevant speech" is
(`--relevance annotation`), never word overlap with the answer.

## Next concrete steps

1. ~~Authenticate to Hugging Face, accept LongVideoBench terms, download videos.~~ **Done** (all 252).
2. ~~Run the controller pipeline on all 252 videos.~~ **Done** — `reports/lvb_full/RESULTS.md`. Finding:
   LongVideoBench cannot test the hypothesis (answers are visual; removing the quoted line does not
   make looking more valuable). A spoken-and-visible dataset is needed; check it first with
   `scripts/analyze_premise.py` on a small pilot.
3. For EduVidQA, decide with the project owner whether downloading the YouTube videos is
   permissible for this research; if yes, fetch transcripts and record per-video availability.
