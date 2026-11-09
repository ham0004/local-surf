# Thesis brief: using lecture speech to choose the frames a small video model sees

Status on 9 November 2026: all experiments are finished. The claim ledger was audited by a separate AI auditor
(OpenAI's GPT), and this brief has been corrected after its sentence-by-sentence check (auditor notes P181–P195). The
accuracy and hit figures come from the frozen test sets. Design details, the loss analysis and the resource figures
are marked as such. The source table at the end gives the file for each claim.

## The problem

A small vision-language model that runs on one 16 GB graphics card has to answer questions about long videos. In this
thesis it is shown at most four frames per question. That is a budget we chose, not a measured hardware limit. In a
lecture, what the speaker says often refers to what is on the slide at that moment. The thesis asks two questions:

1. Can the timed transcript be used to choose those four frames well enough to raise answer accuracy?
2. Does a small trained component do better than hand-built rules at this choice?

## What we built

- **Answer model.** Frozen and never trained by us: Qwen3-VL-4B on the local card (2B in earlier stages), and the
  120B Nemotron vision-language model through NVIDIA's API as a check with a much larger model.
- **S0, starting design.** Our previous version. Keyword search (BM25) over the transcript finds four windows;
  frames are taken from inside those windows; up to four that are most similar to the question are shown, with a
  transcript excerpt of up to 120 words.
- **S1, best rule-based framework.** The same transcript search, plus a picture index of the whole video (one
  MobileCLIP image fingerprint every 2 s). Candidates from both are merged, and a diversity-aware picker that scores
  frames against the question and the answer options chooses four. **S1t** adds up to 600 words of speech: lines near
  the chosen frames plus lines retrieved from the whole transcript by keyword search (BM25) fused with a MiniLM cross-encoder.
- **Head A.** A small trained locator that reads every transcript line and votes for the moments where the answer
  should be visible. Each of its three members is a small network with 1,889 trained parameters on top of frozen text
  rerankers (5,667 across the three). It replaces the speech part of S1. **S2c** is Head A trained on the CG-Bench
  training questions; **S2m** is Head A retrained on our lecture training questions.
- **Head B.** A trained picker that replaces S1's rule for choosing the final four frames (**S3**). It adds frames
  greedily, scoring each by its learned relevance, plus how much it makes the answer options easier to tell apart,
  minus its similarity to frames already chosen. **S4** uses both heads; it was tested on fresh LongVideoBench, not
  on the lecture test.
- **Baselines without our search:** R2, four evenly spaced frames; R3, the four frames most similar to the question in
  the dense picture index; R4, the same with each answer option scored separately; R5, a rule adapted from the recent
  VSI method that combines picture and subtitle scores. R5 was the best of ten settings we tried on lecture dev
  questions. It is our adaptation, not a reproduction of VSI. The baselines differ between the two studies. On fresh
  LongVideoBench, R3 takes the four top-ranked times and merges repeated pictures, so it can show fewer than four. On
  the lecture test, R3–R5 go down a ranked list of up to 96 times until they have four perceptually distinct pictures.

All systems show the answer model at most four frames. On the lecture test, some baselines show fewer on average
(R3 3.87, R4 3.88, R5 3.94). The lecture baselines R3–R5 also decode more frames per question than our systems (up
to 96 against at most about 54), so the search work is not matched.

## Data

- **Fresh LongVideoBench set:** 894 questions on 500 videos from the official validation set. No method was chosen
  or tuned on them before the confirmation run (897 planned; 3 questions on one unavailable video excluded). Later
  analyses on this set are marked as post hoc. This is our own held-out subset, not an official test-server result.
- **Lecture set (M3AV):** a large public collection of recorded talks with slides and transcripts. M3AV has no
  multiple-choice questions, so we generated them from slides with the Nemotron 120B model and filtered them. Train:
  1,147 questions on 297 lectures; dev: 514 on 96; test: 853 on 112. No lecture appears in more than one split. The
  protocol, systems and settings were frozen before the test questions were generated.
- **Primary lecture measure, source-slide interval hit@4:** at least one of the chosen frames has a timestamp inside
  an automatically labelled interval in which the question's source slide is inferred to be on screen (±1 s). The
  labels have known false matches. This is a proxy for "the right slide was shown"; it does not prove that the slide,
  or the needed fact, was actually visible.

## Result 1: fresh LongVideoBench (answer accuracy, %)

| System | 4B (local) | Nemotron 120B (API) |
|---|---|---|
| S0 starting design | 48.3 | 49.3 |
| S1 best framework | 55.9 | 56.5 |
| S1t best framework + speech text | 57.3 | 57.6 |
| S4 both heads | 55.7 | 58.6 |
| R2 evenly spaced | 47.3 | 51.1 |
| R3 dense picture top-4 | 56.0 | 57.6 |

Four declared tests, with 95% intervals from resampling whole videos (not adjusted) and Holm-adjusted p-values:

| Declared test | Difference, points (95% interval) | Holm p | Outcome |
|---|---|---|---|
| H3: S1t − S0, 4B | +8.9 (+5.7..+12.2) | 0.0008 | confirmed |
| H4: S1t − S0, Nemotron 120B | +8.3 (+4.8..+11.8) | 0.0008 | confirmed |
| H1: S4 − S1, 4B | −0.2 (−3.0..+2.7) | 0.9185 | not confirmed |
| H2: S4 − S1, Nemotron 120B | +2.1 (−0.8..+5.1) | 0.3424 | not confirmed |

- The complete new pipeline with speech text beats the starting design with both answer models. This compares two
  packages: pictures and text both change.
- The trained heads add no confirmed accuracy over the rule-based framework.
- Secondary comparisons, not corrected for multiple tests: against plain dense picture search (R3), S1 shows no
  demonstrated advantage (4B −0.1, −2.9..+2.6; 120B −1.1, −4.1..+1.8). This does not show the two are equal. Against
  evenly spaced frames, S1 is higher (4B +8.6, +5.1..+12.2).

## Result 2: lectures (M3AV test, 853 questions, 112 lectures)

| System | Source-slide interval hit@4, % (primary) | 4B accuracy, % (secondary) |
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

Two declared tests, with 95% intervals from resampling whole lectures (not adjusted) and Holm-adjusted p-values:

| Declared test | Difference, points (95% interval) | Holm p | Outcome |
|---|---|---|---|
| L2: S2m − S1, hit@4 | +5.3 (+2.7..+7.8) | 0.0004 | confirmed |
| L1: S2m − R5, hit@4 | −0.5 (−3.8..+2.9) | 0.8253 | not confirmed |

- The Head A package has a higher interval hit rate than the rule-based framework. This compares the whole package
  with S1 (it also changes how much of the transcript is searched and how frames are scored), so it does not isolate
  the learned part.
- Head A shows no demonstrated advantage over the strongest training-free rule, R5. The point estimates are about half
  a point apart, but the interval is wide; this is not evidence that the two are equivalent.
- Answer accuracy is a secondary, uncorrected measure: S2m − S1 +4.7 (+2.5..+6.9); S2m − R5 +0.7 (−1.7..+3.0). There is
  no confirmed accuracy gain from lecture-specific training (S2m vs S2c −0.9, −2.4..+0.5), from Head B (S3 − S1 +0.2,
  −1.9..+2.4) or from speech text (S1t − S1 +0.9, −0.8..+2.8).
- The controls matter: the answer model answers 51.2% correctly from the question alone and 62.1% from the transcript
  alone. Many generated questions can therefore be answered without frames; this does not tell us which questions
  logically need the slide.

## Result 3: where the labelled slide is missed (lecture test, post hoc)

| | S1 | S2m |
|---|---|---|
| A candidate frame hits the source-slide interval, before the final choice, % | 81.2 | 89.0 |
| A chosen frame hits the source-slide interval, % | 74.1 | 79.4 |

For S2m, answers were correct for 93.9% of the 677 questions where a chosen frame hit the interval, 56.1% of the 82
where a candidate hit it but none was chosen, and 66.0% of the 94 where no candidate hit it. This is an association,
not proof that showing the slide caused the right answer, and an automatic hit can itself be a false match. The
shares of questions lost at the candidate stage and at the final choice are of similar size.

## Human check of the lecture questions

Three thesis group members, who were not given any system output, reviewed 100 sampled test questions (one per
lecture), checking the key, the options and the source slide. Under the rule fixed before any reviewer answered: 97
accepted, 0 rejected, 3 without a majority. Removing those 3 moves the estimates slightly (L2 +5.4, +2.8..+8.0; L1
−0.4, −3.7..+3.0) and leaves their interpretation unchanged.

This check is weaker than it looks. One reviewer accepted all 100 and was the only reviewer for 33 questions, so a
combined reject was not possible under the fixed rule. Agreement with the two model reviews was 82% and 77%, but kappa
was near zero. According to the owner, the reviewers worked independently and did not see each other's answers;
because answers were posted in a group chat at different times, this rests on that account rather than on a technical
control. Whether each used a separate browser or device was not recorded. The selected frames were not reviewed. The
review supports the questions on a sample; it does not validate all 853 labels. How the reviewers' consent was
obtained, and whether any institutional review applied, is still to be stated.

## What the thesis can and cannot claim

Can claim:
- A careful empirical study of transcript-guided four-frame selection for lecture QA, with frozen methods and declared
  primary tests, held-out tests, controls and an audit trail.
- The complete rule-based pipeline with speech text beats our starting design on fresh data with a 4B and a 120B
  answer model (+8.9 and +8.3).
- On lectures, the Head A package raises the source-slide interval hit over the rule-based framework (+5.3), with no
  demonstrated advantage over a strong training-free rule.
- In the two tested contrasts, small trained heads add no confirmed accuracy on fresh long-video questions.

Cannot claim: a new state of the art; an advantage of Head A over R5; that Head A and R5 are equivalent; an advantage
of S1 over dense picture search on fresh LongVideoBench; that the chosen frames contain the answer; that the whole
project ran locally (question generation and the 120B runs used an API).

## Limitations

- The lecture labels are automatic slide intervals with known false matches, and the questions are machine-generated;
  the controls show many can be answered without frames.
- The human check covers a sample, with the weaknesses above.
- Several comparisons are packages that change more than one thing.
- Deviations are recorded:
  - The declared total of 20,000 calls for the confirmation runs was exceeded. Each ledger had its own 20,000 cap, so
    the total was not enforced, and the ledgers charged 25,373 calls. These include API follow-ups on the earlier test,
    so they are not all LongVideoBench calls.
  - One crashed control answer in the lecture QA run was retried, against the rule.
  - In the frozen lecture QA run, the observed reserved memory reached 16.8 GiB on a 15.93 GiB card. One
    out-of-memory crash and restart is recorded.
  None of these changes a primary result, and each is documented.

## Publication plan

The owner has chosen CVPR 2027 (paper deadline 16 November 2026). The auditor and I both think acceptance at the
main conference is unlikely, because the trained part shows no demonstrated advantage over the strongest
training-free baseline and recent work (VSI, GCR, KFS-Bench, VideoRouteAudit) covers nearby ground. The paper is
therefore written as an empirical study, with no claim of a new best method. In our opinion an empirical multimedia
venue or a workshop is a better fit. If it is not accepted, we would choose the next venue only after checking its
dates against the CVPR decision, and would not submit to two venues at once.

## Authorship and tool use

The authors conceived the research idea, designed the framework and the experiment plan, directed every experiment
and checked all results. Within this plan, Anthropic's Claude, used through the Claude Code agent, implemented the
code, ran the experiments and drafted the text of the paper, which the authors reviewed and edited. A second model,
OpenAI's GPT, acting as a separate AI auditor, double-checked the numbers, statistics and wording against the
experiment records and helped search the related work. The lecture questions for training, development and test
were generated by the Nemotron-3.5-Super-VL 120B model, which is also one of the answer models, and the two model
reviews used for agreement statistics were produced by Claude and GPT. The authors take full responsibility for the
content of the paper.

## Sources

Claim IDs refer to the audited claim ledger (auditor notes P167 and P184).

| Section | Claim IDs | Files |
|---|---|---|
| Fresh LongVideoBench data and results | C09, C10, C11, C12, C13, C14, C15, C17 | `reports/v3_confirmation/summary.md`; `docs/v3/confirmation_deviations.md` |
| System design (Head A size and voting, Head B rule, input caps, baselines per study) | C61, C62, C63, C64, C67 | `src/videoqa/v3/head_a.py`; `scripts/v3_head_b.py`; `scripts/v3_baseline_cycle.py`; `docs/v3/m3av_systems.md` |
| Lecture data and question rules | C25, C65, C66 | `docs/v3/m3av_protocol.md`; `scripts/m3av_questions.py`; `scripts/m3av_visibility.py` |
| Lecture hit@4 and declared tests | C26, C27, C28, C29, C30, C55 | `reports/m3av/hits_test.md` |
| Lecture answer accuracy | C32, C33, C34 | `reports/m3av/qa_test.md`; `docs/v3/m3av_qa_test_deviation.md` |
| Where the labelled slide is missed | C39, C40, C60 | `reports/m3av/evidence_loss_test.md` |
| Human check | C41, C42, C44, C45 | `reports/m3av/human_review_test.md`; `reports/m3av/human_review_limits.md` |
| Deviations and memory | C37, C38 (observed peak only), C53 | `docs/v3/m3av_qa_test_deviation.md`; `docs/v3/confirmation_deviations.md` |
