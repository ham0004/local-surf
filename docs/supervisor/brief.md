# Thesis brief: using lecture speech to choose the frames a small video model sees

Status on 9 November 2026: all experiments are finished and an independent audit of every claim is complete.
Each number below is from a frozen test set unless it is marked otherwise; the source table at the end gives the file
for each one.

## The problem

A small vision-language model that runs on one 16 GB graphics card can only look at a few pictures from a long video.
We give it four. In a lecture, what the speaker says usually points to what is on the slide at that moment. The
thesis asks two questions:

1. Can the timed transcript be used to choose those four frames well enough to raise answer accuracy?
2. Does a small trained component beat hand-built rules at this choice?

## What we built

- **Answer model.** Frozen and never trained by us: Qwen3-VL-4B on the local card (2B in earlier stages), and the
  120B Nemotron vision-language model through NVIDIA's API as a check with a much larger model.
- **S0, starting design.** Our previous version. Keyword search (BM25) over the transcript finds four windows;
  frames are taken from inside those windows; the four most similar to the question are shown, with a 120-word
  transcript excerpt.
- **S1, best rule-based framework.** The same transcript search, plus a picture index of the whole video (one
  MobileCLIP image fingerprint every 2 s). Candidates from both are merged, and a diversity-aware picker that scores
  frames against the question and the answer options chooses four. **S1t** adds up to 600 words of speech chosen
  around those frames.
- **Head A.** A small trained locator (1,889 trained parameters on top of frozen text rerankers) that reads every
  transcript line and votes for the moments where the answer should be visible. It replaces the speech part of S1.
  **S2c** is Head A trained on the CG-Bench training questions; **S2m** is Head A retrained on our lecture training
  questions.
- **Head B.** A trained picker that replaces S1's rule for choosing the final four frames (**S3**). **S4** uses both
  heads.
- **Baselines without our search:** R2, four evenly spaced frames; R3, the four frames most similar to the question in
  the dense picture index; R4, the same with each answer option scored separately; R5, a rule adapted from the recent
  VSI method that combines picture and subtitle scores. R5 was the strongest training-free rule on lecture dev
  questions. It is our adaptation, not a reproduction of VSI.

All systems show the answer model at most four frames. Some baselines sometimes show fewer (R3 3.87 frames on
average on the lecture test, R4 3.88, R5 3.94). The baselines search more frames than our systems do, so the search
work is not matched.

## Data

- **Fresh LongVideoBench set:** 894 questions on 500 videos from the official validation set, held out and not used
  for any decision (897 planned; 3 questions on one unavailable video excluded). This is our own held-out subset, not
  an official test-server result.
- **Lecture set (M3AV):** a large public collection of recorded talks with slides and transcripts. M3AV has no
  multiple-choice questions, so we generated them from slides with the Nemotron 120B model and filtered them. Train:
  1,147 questions on 297 lectures; dev: 514 on 96; test: 853 on 112. No lecture appears in more than one split. The
  protocol, systems and settings were frozen before the test questions were generated.
- **Primary lecture measure, source-slide hit@4:** at least one of the four chosen frames falls inside an
  automatically labelled time interval when the question's source slide is on screen (±1 s). This is an automatic
  proxy for "the right slide was shown". It does not prove that the needed fact was visible.

## Result 1: fresh LongVideoBench (answer accuracy, %)

| System | 4B (local) | Nemotron 120B (API) |
|---|---|---|
| S0 starting design | 48.3 | 49.3 |
| S1 best framework | 55.9 | 56.5 |
| S1t best framework + speech text | 57.3 | 57.6 |
| S4 both heads | 55.7 | 58.6 |
| R2 evenly spaced | 47.3 | 51.1 |
| R3 dense picture top-4 | 56.0 | 57.6 |

Four declared tests, Holm-corrected, 95% intervals from resampling whole videos:

| Declared test | Difference, points (95% interval) | Holm p | Outcome |
|---|---|---|---|
| H3: S1t − S0, 4B | +8.9 (+5.7..+12.2) | 0.0008 | confirmed |
| H4: S1t − S0, Nemotron 120B | +8.3 (+4.8..+11.8) | 0.0008 | confirmed |
| H1: S4 − S1, 4B | −0.2 (−3.0..+2.7) | 0.9185 | not confirmed |
| H2: S4 − S1, Nemotron 120B | +2.1 (−0.8..+5.1) | 0.3424 | not confirmed |

- The complete new pipeline with speech text beats the starting design with both answer models. This compares two
  packages: pictures and text both change.
- The trained heads add no confirmed accuracy over the rule-based framework.
- Against plain dense picture search (R3), S1 shows no advantage (4B −0.1, −2.9..+2.6; 120B −1.1, −4.1..+1.8). This
  does not show the two are equal. Against evenly spaced frames S1 is clearly better (4B +8.6, +5.1..+12.2).

## Result 2: lectures (M3AV test, 853 questions, 112 lectures)

| System | Source-slide hit@4, % (primary) | 4B accuracy, % (secondary) |
|---|---|---|
| S0 starting design | 67.4 | 79.5 |
| S1 best framework | 74.1 | 82.5 |
| S2c Head A, trained on CG-Bench | 79.7 | 88.2 |
| S2m Head A, trained on lectures | 79.4 | 87.2 |
| S3 Head B | 73.6 | 82.8 |
| R2 evenly spaced | 22.9 | 63.5 |
| R3 dense picture top-4 | 64.1 | 78.3 |
| R4 option-aware dense | 76.9 | 84.8 |
| R5 VSI-inspired rule | 79.8 | 86.5 |
| Question only, no video (control) | – | 51.2 |
| Transcript only, no frames (control) | – | 62.1 |

Two declared tests, Holm-corrected, 95% intervals from resampling whole lectures:

| Declared test | Difference, points (95% interval) | Holm p | Outcome |
|---|---|---|---|
| L2: S2m − S1, hit@4 | +5.3 (+2.7..+7.8) | 0.0004 | confirmed |
| L1: S2m − R5, hit@4 | −0.5 (−3.8..+2.9) | 0.8253 | not confirmed |

- Head A puts the source slide among the four frames more often than the rule-based framework. This compares the whole
  Head A package with S1 (it also changes how much of the transcript is searched and how frames are scored), so it
  does not isolate the learned part.
- Head A does not beat the strongest training-free rule, R5. The two are within about half a point, and the interval
  is wide enough that neither is shown to be better.
- Answer accuracy is a secondary, uncorrected measure: S2m − S1 +4.7 (+2.5..+6.9); S2m − R5 +0.7 (−1.7..+3.0). There is
  no confirmed accuracy gain from lecture-specific training (S2m vs S2c −0.9, −2.4..+0.5), from Head B (S3 − S1 +0.2,
  −1.9..+2.4) or from speech text (S1t − S1 +0.9, −0.8..+2.8).
- The controls matter: the answer model gets 51.2% from the question alone and 62.1% from the transcript alone, so many
  generated questions can be answered without seeing the slide.

## Result 3: where the right slide is lost (lecture test, after the fact)

| | S1 | S2m |
|---|---|---|
| Source slide among the candidates before the final choice, % | 81.2 | 89.0 |
| Source slide among the four chosen frames, % | 74.1 | 79.4 |

For S2m, answers were correct for 93.9% of the 677 questions where the source slide was chosen, 56.1% of the 82 where
it was among the candidates but not chosen, and 66.0% of the 94 where it never reached the candidates. This is an
association, not proof that showing the slide caused the right answer. Losses at the candidate stage and at the final
choice are of similar size.

## Human check of the lecture questions

Three thesis group members, who were not given any system output, reviewed 100 sampled test questions (one per
lecture), checking the key, the options and the source slide. Under the rule fixed in advance: 97 accepted, 0
rejected, 3 without a majority. Removing those 3 leaves L2 at +5.4 (+2.8..+8.0) and L1 at −0.4 (−3.7..+3.0).

This check is weaker than it looks. One reviewer accepted all 100 and was the only reviewer for 33 questions, so a
combined reject was not possible under the fixed rule. Agreement with the two model reviews was 82% and 77%, but kappa
was near zero. The reviewers worked independently and did not see each other's answers, according to the owner; since
answers were shared in a group chat at different times, this rests on that account rather than on a technical control. The selected frames were not reviewed. The review supports the questions on a sample; it does not validate
all 853 labels.

## What the thesis can and cannot claim

Can claim:
- A careful empirical study of transcript-guided four-frame selection for lecture QA, with frozen protocols, held-out
  tests, controls and an audit trail.
- The complete rule-based pipeline with speech text beats our starting design on fresh data with a 4B and a 120B
  answer model (+8.9 and +8.3).
- A trained speech-to-moment locator improves source-slide hit over the rule-based framework on lectures (+5.3) but
  only matches a strong training-free rule.
- Small trained heads add no confirmed accuracy on fresh long-video questions.

Cannot claim: a new state of the art; an advantage over R5 or over dense picture search; that the two are equivalent;
that the chosen frames contain the answer; that the whole project ran locally (question generation and the 120B runs
used an API).

## Limitations

- The lecture labels are automatic slide intervals, and the questions are machine-generated; the controls show many
  can be answered without video.
- The human check covers a sample, with the weaknesses above.
- Several comparisons are packages that change more than one thing.
- Deviations are recorded: the declared call allowance for the confirmation runs was exceeded (25,373 ledger-charged
  calls against 20,000 declared); one crashed control answer in the lecture QA run was retried, against the rule; the
  frozen lecture QA run reserved more memory than the card has (16.8 of 15.93 GiB) and needed one restart after an
  out-of-memory crash. Each is documented and none changes a primary result.

## Publication plan

The owner has chosen CVPR 2027 (paper deadline 16 November 2026). The auditor and I both think acceptance at the
main conference is unlikely, because the trained part does not beat the strongest training-free baseline and recent
work (VSI, GCR, KFS-Bench, VideoRouteAudit) covers nearby ground. The paper is therefore written as an empirical
study, with no claim of a new best method. A multimedia venue (ICMR 2027) or a CVPR workshop remain the more
realistic targets if it is not accepted.

## Sources

Claim IDs refer to the audited claim ledger (auditor note P167).

| Section | Claim IDs | Files |
|---|---|---|
| Fresh LongVideoBench data and results | C09, C10, C11, C12, C13, C14, C15, C17 | `reports/v3_confirmation/summary.md`; `docs/v3/confirmation_deviations.md` |
| Lecture data | C25 | `docs/v3/m3av_protocol.md`; `docs/v3/m3av_systems.md` |
| Lecture hit@4 and declared tests | C26, C27, C28, C29, C30, C55 | `reports/m3av/hits_test.md` |
| Lecture answer accuracy | C32, C33, C34 | `reports/m3av/qa_test.md`; `docs/v3/m3av_qa_test_deviation.md` |
| Where the slide is lost | C39, C40 | `reports/m3av/evidence_loss_test.md` |
| Human check | C41, C42, C44, C45 | `reports/m3av/human_review_test.md`; `reports/m3av/human_review_limits.md` |
| Deviations | C37, C53 | `docs/v3/m3av_qa_test_deviation.md`; `docs/v3/confirmation_deviations.md` |
