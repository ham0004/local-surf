# Dataset selection rule (declared 2026-09-28, before any V0/V1/V2 result)

Required by section 5 of the revision prompt: "Declare inclusion rules on
development material ... do not repeatedly switch datasets until significance
appears."

## Candidates
Named in the prompt: EduVidQA, LongVideoBench, QVHighlights (retrieval only,
not a QA-label source), NExT-GQA. Added by us: Video-MME (public on Hugging
Face, has subtitles), and TVQA (licence request needed). The additions are not
in the prompt and are reported as additions.

## Screen (text only; trains nothing)
scripts/speech_stratum_check.py, 50 questions, one per video, fixed hash
order, frozen answerer, seed 0. The measure is the stratum-1 rate: the
transcript-only answer is right on clean and wrong on targeted damage. Drops,
relevance methods and the control-damage failure rate are reported for every
dataset.

## Rule
1. Take the dataset with the highest stratum-1 rate, provided the rate is at
   least 0.25 and the videos are accessible under their licence. If nothing
   qualifies, the blocker is reported and no dataset is chosen.
2. The rule uses ONLY the screen. It never uses V0/V1/V2 or controller
   results. The choice is made once and recorded here. If the chosen
   dataset later fails for access or licence reasons, that is reported
   before the next candidate is used.
3. Splits are by source video (or course), made inside the chosen dataset.
   The test split is labelled only once, for the locked final evaluation.
4. Every stratum is kept in train, dev and test (speech and vision,
   visual-only, spoken with no visual replacement, unrelated damage, and
   natural ASR error). The final test is not filtered to stratum 1 or by
   answerer success; stratum 1 is the primary endpoint.
5. Datasets are never pooled. The LongVideoBench pilot is reported
   separately as feasibility evidence only.

## Screen results
| dataset | n | attempted | drops | text-only acc clean / targeted / control | stratum 1 | rate | qualifies |
|---|---|---|---|---|---|---|---|
| Video-MME (subtitled) | 50 | 84 | no answer words in subtitles 32, no matched control 2 | 0.54 / 0.50 / 0.54 | 2 | 0.04 | no |
| LongVideoBench | 50 | 60 | no relevant segment 5, no matched control 5 | 0.22 / 0.26 / 0.22 | 2 | 0.04 | no |

Video-MME: 38% of the attempted questions have no subtitle line containing
the answer's words, and targeted damage changes almost no answers. Clean
text-only accuracy of 0.54 on 4-way questions largely survives targeted
damage, so it probably comes from the question and options, not from the
speech. Neither dataset meets the rule. EduVidQA and NExT-GQA are screened
next (runs/stratum/*.json).

## Release audit of the remaining candidates (no screen possible yet)
- **EduVidQA** (github sourjyadip/eduvidqa-emnlp25, MIT licence for the code).
  The release contains CSVs only: synthetic_train 3,909 questions from 157
  videos, synthetic_test 1,056 from 40, real_world_test 269 from 99. Each row
  has a YouTube id, a timestamp and a question. The answers are free-form long
  answers; there are no multiple-choice options. Videos and transcripts are
  NOT included; the README says to fetch them from YouTube with yt-dlp and
  youtube-transcript-api. So the screen needs a YouTube transcript fetch
  first, and scoring needs an audited free-form rubric (prompt section 5)
  instead of option accuracy. Blocked on the user's go-ahead for YouTube
  access.
- **NExT-GQA**: from the NExT-QA source videos, with grounded temporal spans
  for visual QA. Transcripts or subtitles are not part of the release as far
  as the repository describes, and the questions target visual causal and
  temporal events, so stratum 1 is expected to be near zero. Not a candidate
  for the primary endpoint; possibly useful for stratum 2 (answer mainly
  visual). Unverified: the repository has not been fetched.
- **QVHighlights**: a retrieval dataset, not QA (per the prompt). Excluded.

## EduVidQA screen (2026-09-28)
Transcripts: 87 of 296 videos fetched before YouTube blocked the IP
(IpBlocked). 69 of the 87 are manual English captions and 18 are ASR. There is
no train/test video overlap. The screen used 43 official-train questions (one
per video with a transcript), free-form, token F1 against the reference, with
the question's own timestamp window added to the excerpt.

| | F1 clean | F1 targeted | F1 control | F1 drop ≥ 0.10 under targeted | qualifies |
|---|---|---|---|---|---|
| EduVidQA synthetic_train | 0.278 | 0.280 | 0.278 | 0 / 43 | no |

Diagnosis (runs/stratum/eduvidqa50.json):
- The synthetic reference answers are long general explanations (median 121
  words). Their content words (excluding the question's words) occur in the
  timestamp window no more often than in a random 40 s window (6% vs 6%;
  28% anywhere in the transcript). The answers are not grounded in the
  speech at the asked time, so deleting that speech cannot hurt them.
- Targeted damage deletes at most 3 of the median 9 segments in the window.
- Token F1 against a 121-word reference is a weak scorer (prompt section 5
  asks for an audited rubric for free-form answers).
- Human-written real_world_test questions are official test and stay locked.
  They are not used for screening or training.

Status: no candidate meets the rule. The next step is not another dataset
switch. It is a controlled, marked-synthetic spoken-answer set built from
these lecture transcripts, with a human audit, or TVQA if its licence is
obtained. Both are proposals for the user to decide.
