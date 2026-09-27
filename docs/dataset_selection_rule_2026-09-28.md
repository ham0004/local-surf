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
(filled in when runs/stratum/*.json are complete)
