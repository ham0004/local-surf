# Claim map for the CVPR 2027 draft (for the auditor)

Every sentence in the draft that states a number, a comparison or a contribution, with the ledger ID from auditor note
P167 or, where no ledger row covers it, the source file. "new" marks numbers not in the ledger that need a fresh check.
Draft files: `paper/cvpr2027/sec/*.tex`.

## Abstract and introduction (`0_abstract.tex`, `1_intro.tex`)

| Claim in the draft | Ledger ID or source |
|---|---|
| 894 held-out LongVideoBench questions | C09 |
| S1t − S0: +8.9 (4B), +8.3 (120B) | C12, C13 |
| Two trained heads add no confirmed accuracy | C10, C11 |
| 853 questions generated from slides, 112 lectures; derived, not official | C25 |
| S2m 79.4% vs S1 74.1%; +5.3 | C26, C28 |
| R5 79.8%; L1 −0.5 (−3.8, +2.9) | C26, C27 |
| Slide lost about as often before as at the final choice | C39 (11.0 vs 9.6) |
| Answers right 93.9% when the slide is shown; 56.1% to 66.0% when not | C40 |
| Question-only 51.2%, transcript-only 62.1% | C32 |
| Tuned pipeline shows no advantage over dense top-4 | C14 |
| Contributions: empirical only; no new state-of-the-art selector | P173; C56 avoided |

## Related work (`2_related.tex`)

| Claim | Source |
|---|---|
| Descriptions of VSI, GCR (32 frames, text rendered on frames), Frame-Voyager, BOLT, KFS-Bench, EG-VQA, Evidence-Backed VQA, VideoRouteAudit, LEMON, Video-MMLU, MaViLS, Cite or Decline, ReQuest, MSJoE, HORNet, RL sampler | P172 and the cited papers |
| Our heads are not new as mechanisms | P172, P173, `docs/v3/heads_design.md` |

## Study design (`3_study.tex`)

| Claim | Source |
|---|---|
| K at most 4, same model, prompt; frames at most 640 px | C30; `docs/v3/m3av_systems.md` (640 px: new) |
| System descriptions S0-S4, R2-R5; R5 adaptation and its differences from VSI; best of 10 dev settings | C55; `docs/v3/m3av_systems.md` |
| R3-R5 rank 96 frames; S1-S3 decode about 54 | P170; `docs/v3/m3av_systems.md` notes |
| Head A: residual MLP over 21 features + bge scores, zero-initialized, density voting, 6 moments, 10 s positive window, 3 seeds, 1,889 parameters | `docs/v3/research_log.md` step 39; README (new) |
| S2m trained on 1,147 questions / 297 train lectures | C25 |
| Head B: λ 2, μ 1, internal validation, CG-Bench train pools | `docs/v3/m3av_systems.md`; `docs/v3/heads_design.md` (new) |
| Earlier locked test: 459 questions (55/6, 202 + 202 on 202 lectures); reused for labelled follow-ups | C01, C05 |
| Fresh set: 894/500; 3 questions on 1 unavailable video excluded; not an official test-server result | C09, P174 |
| Generation and filter rules (4 s, 8 words, title slides, 4 options, hash order, 80% OCR match, 2 s visibility, no answer-model filter) | `docs/v3/m3av_protocol.md` (new) |
| Splits 1,147/297, 514/96, 853/112; official speaker-disjoint split | C25; `docs/v3/m3av_protocol.md` |
| hit@4 with 1 s tolerance; S-src sealed before test | C26, C31 |
| Controls: question only; transcript only with 600 BM25 words | C32; `docs/v3/m3av_protocol.md` |
| Human check: 100 questions, one per lecture, drawn after freeze; packet; rule fixed before verdicts | C41; research log steps 98-99 |
| Declared H1-H4, L1-L2, Holm; L1 changed from R3 to R5 by a dev-only amendment before test generation | C10-C13, C27, C28, P171 |
| Bootstrap over videos or lectures, 10,000; smallest p about 0.0002; failures count as wrong | P164 |

## Results (`4_results.tex`)

| Claim | Ledger ID or source |
|---|---|
| Table of fresh accuracies (27 values) | C15 |
| H1-H4 with Holm p | C10-C13 |
| One failed S1t answer with 4B counted wrong | C12 |
| Post hoc: S1 frames at fixed S0 text +8.2; text change on S1 frames +0.8; S1t − S1 +1.3 | C21 |
| Head A alone +1.8, +1.2, +2.8 (0.0, 5.6), p 0.052; Head B within one point | C16 |
| Residual ablation on fresh set +0.3 (−2.1, +2.8); learned part shows no measurable contribution | C18 |
| S1 − R2: +7.5, +8.6, +5.4; S1 − R3: −0.8, −0.1, −1.1 | C17, C14 |
| Lecture table: hit@4 and frames for all systems | C26, C30 |
| Lecture table: S-src for S1, S2m, R5 | C31 |
| Lecture table: S-src for S0, S2c, S3, R2, R3, R4 | `reports/m3av/hits_test.md` (new) |
| Lecture table: accuracy for all systems and controls | C32 |
| L2 +5.3; L1 −0.5 with Holm p | C27, C28 |
| S2m package changes transcript scope, rescoring, density voting and residual | P170 |
| S-src L1 −1.5, L2 +5.0 | C31 |
| S2m − R3 +15.2; − R4 +2.5; − S2c −0.4 | C29 |
| R3, R4, R5 show fewer than 4 frames for 52, 48, 27 questions | C30 |
| Accuracy: S2m − S1 +4.7; − R5 +0.7; S3 − S1 +0.2; S1t − S1 +0.9 | C33, C34 |
| Transcript-only-wrong subset, 322 questions: S2m − R5 +5.9 (1.6, 10.3); − S1 +9.6 (5.4, 13.8) | C35, C36 (deviation appendix) |
| Loss table for S1 and S2m | C39 |
| Loss table for S0 (75.3, 24.7, 7.9, 67.4) | `reports/m3av/evidence_loss_test.md` (new) |
| S2m accuracy by outcome: 677/93.9, 82/56.1, 94/66.0; 41 wrong answers with slide shown, unadjudicated | C40, P170 |
| Human check: 67 double, 33 single, 167 verdicts; 97/0/3; rejections were about the key | C41; `reports/m3av/human_review_test.md` |
| Rejected-or-uncertain 3.0 (0, 7.0) and 2.7 (0, 6.4) | C42 |
| Masks: L1 −0.4 (−3.7, +3.0), L2 +5.4 (+2.8, +8.0) | C45 |
| All-accept reviewer, structural impossibility of a reject, degenerate interval, kappa values, independence unverified, frames not reviewed | C43 (avoided), C44, P168; `reports/m3av/human_review_limits.md` |
| Pool time S1 0.67 s, S2m 0.97 s, R5 1.42 s, exclusions | C50 |
| Answer call about 2.4-2.5 s; 4.55 s with text | C51 (no comparative claim made) |
| Dense index 955 s for 112 lectures, mean 19.3 min | C52 |
| Frozen run reserved 16.8 of 15.93 GiB; one OOM and restart | C38 (stated as a deviation, not a blanket claim) |
| Capped dev replay: 600 identical replies, 14.19 GiB reserved, cap 14.28 GiB | C48 |

## Discussion and limitations (`5_discussion.tex`)

| Claim | Source |
|---|---|
| Candidate share 81.2 to 89.0 | C39 |
| Learned residual no measurable contribution; lecture training no gain over CG-Bench training | C18, C29, C33 |
| "A plausible reading ... does not isolate" (hypothesis, labelled as such) | interpretation; P170 |
| S2m − S1: 5.3 hit, 4.7 accuracy; S2m vs R5 no confirmed difference on either | C28, C33, C27 |
| At most 11.0 and 9.6 points recoverable per stage for S2m | C39 (arithmetic) |
| Limitations: proxy labels, generated questions, human-check limits, packages, unmatched search, R5 adaptation, one 4B model, shipped transcripts, post hoc analyses | P170, P171, P174, P178 |
| Deviations: 25,373 vs 20,000 calls; retried control (322 vs 323); memory | C53, C37, C38 |

## Statements (`6_statements.tex`)

| Item | Status |
|---|---|
| LLM Usage Statement | names Nemotron (question generation, answer model), Claude via Claude Code (code, experiments, reports, drafting), an OpenAI GPT model (audit, related-work search, model review); owner to confirm and add versions |
| Human-study consent and oversight | TODO: owner to supply (P180) |

## Not used anywhere in the draft

C38 as a blanket claim, C43, C54, C56; "first", "state of the art", "equivalent", "fully local", "97% of labels
verified", "<0.0001".
