# Lecture study: limits of the blind human review

This file goes with [`human_review_test.md`](human_review_test.md). That report is the fixed output of the analysis
declared before any human answer was seen (research log steps 98-99). It is left as generated. This file states what
the result can and cannot show (auditor note P168, research log step 103). Pseudonyms H1-H3 only.

## What was reviewed

- 100 test questions, one from each of 100 sampled lectures, out of 853 test questions on 112 lectures.
- Each reviewer judged the question, its answer key and its source slide: is the key right, is exactly one option
  right, is the source slide the one the question uses, is it legible, can the question be answered without the
  video. Reviewers did not judge the frames chosen by any system, and did not check the other 753 questions.

## How the votes fall

| Reviewer | items answered | accept | reject | notes written |
|---|---|---|---|---|
| H1 | 100 (items 1-100) | 100 | 0 | 0 |
| H2 | 34 (items 1-34) | 31 | 3 | 3 |
| H3 | 33 (items 35-67) | 33 | 0 | 1 |

- H1 accepted every item, answered yes to every check and wrote no notes.
- Items 1-67 have two votes (H1 plus H2 or H3). Items 68-100 have one vote, from H1 alone.
- Under the fixed strict-majority rule, an item with H1's accept can only end as accept (one vote, or two accepts) or
  as no majority (one accept, one reject). A combined reject was therefore impossible for every item. The
  0 combined rejects and the unchanged result after removing rejected items follow from this vote pattern. They are
  not independent evidence that every label is correct.
- The 3 rejects all come from H2 and all give reason KEY (answer key wrong). They are the 3 no-majority items.

## Agreement and kappa

Both are reported in `human_review_test.md` and both are kept: human vs builder 82% (kappa 0.019), human vs auditor
77% (kappa 0.074), builder vs auditor 84% (kappa 0.525), H1 vs H2 31 of 34 (kappa 0.0), H1 vs H3 33 of 33 (kappa
undefined, because neither reviewer gave anything but accept on those items). Almost every verdict is accept, so
percent agreement is high and kappa is near zero or undefined. Neither number shows how carefully a reviewer worked.

## Interval

The rejected rate is 0.0 with a 95% interval of 0.0..0.0. With zero rejects observed, every bootstrap resample also
has zero rejects, so the interval collapses to a point. It is not an upper bound on the true error rate of the labels.
The prespecified estimator is kept as reported.

## Independence of reviewers: unverified

- The review page hid all system results and both model reviews from the reviewers.
- The three reviewers had view-only access to the page, so their answers could not be saved to the shared store.
  They used the page's "Copy my answers" fallback and posted the texts in the owner's thesis group chat, at different
  times on 8 November 2026 (Dhaka): H3 at 00:59, H1 at 21:42, H2 at 23:59.
- It is not known whether a later reviewer could read an earlier reviewer's posted answers before finishing their
  own. In view-only mode, unsent answers are kept in the browser, so reviewers sharing one browser profile could see
  each other's drafts; whether each used a separate profile or device is not known. The owner has been asked both
  questions (auditor note P180).
- Until those facts are established, the review is described as blind to system results, with independence between
  reviewers unverified. No contamination is claimed either.

## Timing and provenance

- Eligibility uses the time each text reached the owner, as the fixed rule says. These minute-level times come from
  private screenshots of the chat (kept with SHA-256, not in git) and the owner's confirmations of dates, time zone and
  which chat name belongs to which reviewer.
- The first run of the fixed analysis used the builder's later forwarding time, found all copies late and stopped
  without a result. The receipt times were then corrected. Both runs are recorded in research log step 102.
- The owner's 12 answers (items 68-79) are not votes, because the owner is not a reviewer. By their saved times,
  4 were saved before the cutoff and 8 after. The store snapshot was read at 00:33 Dhaka on 9 November, 34 minutes after
  the cutoff. All 12 are excluded.
- No answer contradicted itself, so no clarification was requested. The 3 no-majority items were not adjudicated
  after the cutoff.

## Consent and oversight

To be stated by the owner (auditor note P180): how the reviewers agreed to take part, and whether any institutional
review or approval applied. Nothing is assumed here.

## How the paper may use this result

- Allowed: 100 sampled questions were reviewed under a fixed rule; 97 accepted, 0 rejected, 3 without a majority;
  results after removing the uncertain items are unchanged (L2 +5.4, +2.8..+8.0); the limits above.
- Not allowed: "97% of labels verified", "0% label error", "independent human validation of the benchmark", or any
  claim that the human review checked the selected frames.
