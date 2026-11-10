# Claim map for the CVPR 2027 draft (for the auditor)

Every sentence in the draft that states a number, a comparison, a mechanism or a contribution, with the ledger ID
(auditor notes P167 and P184) or the source file. The "Wording used" column gives the bounded reading that the
draft now uses. A ledger ID validates the quantity, not every paraphrase. Draft files: `paper/cvpr2027/sec/*.tex`.

Recurring rules applied throughout (P186):
1. An interval containing zero is "no confirmed" / "no demonstrated" difference, never equality or "nothing measurable".
2. hit@4 and S-src are timestamp intersections with automatic intervals (known false matches), never verified
   visibility of the slide or the fact.
3. Package contrasts are not attributed to a single component or to speech as such.
4. Mechanisms follow the code: Head B rewards option decisiveness; 1,889 parameters per Head A member; R2/R3 differ
   between the two studies; the residual contrast is full minus residual-off; printed intervals are unadjusted and Holm
   adjusts p-values.

## Abstract and introduction (`0_abstract.tex`, `1_intro.tex`)

| Claim | Wording used | Ledger ID or source |
|---|---|---|
| Four frames | A chosen budget, not a hardware limit | C64 |
| Fixed before testing; declared comparisons | Methods fixed on dev; primary comparisons declared and Holm-corrected; others secondary or post hoc | C10-C13, C27-C28, P188 |
| 894 held-out LongVideoBench questions | as stated | C09 |
| S1t − S0: +8.9 (4B), +8.3 (120B) | package gain | C12, C13 |
| Heads | "no confirmed gain" | C10, C11 |
| S1 vs dense top-4 | "no demonstrated advantage" | C14 |
| 853 generated questions, 112 lectures | as stated | C25, C65 |
| S2m 79.4% vs S1 74.1% (+5.3) | "places a frame inside the automatically labelled source-slide interval" | C26, C28 |
| R5 79.8%; L1 −0.5 (−3.8, +2.9) | "no demonstrated difference"; "not shown to be equal" | C26, C27 |
| Missed about as often before as at the final choice | post hoc, proxy | C39 (11.0 vs 9.6) |
| 93.9% when a chosen frame falls in the interval; 56.1% to 66.0% otherwise | association | C40 |
| Question-only 51.2%, transcript-only 62.1% | observed control accuracies, not logical answerability | C32 |
| Showing the moment "should make such a question easier", but models often answer without it | motivation, consistent with controls | P187 |
| Contributions | empirical; "gains on fresh data are not confirmed"; post hoc breakdown, not causal attribution | P173, P187 |

## Related work (`2_related.tex`)

| Claim | Source |
|---|---|
| Descriptions of BOLT, VSI, GCR (32 frames, text rendered on frames), Frame-Voyager, M-LLM selection, flexible selection, RL sampler, MSJoE, HORNet (workshop), ReQuest, KFS-Bench, EG-VQA, Evidence-Backed VQA, VideoRouteAudit, Video-MMMU, Video-MMLU, LEMON, M3AV, MaViLS, Cite or Decline | P172, P187 and the cited papers |
| Uniform sampling "can miss" moments under a small budget | reworded per P187 |
| Our heads are not new as mechanisms; speech + visual retrieval has precedents in VSI and GCR | P172, P187 |
| One answer stage per question; at most four frames vs 32 in GCR's main setting; not a matched comparison | P187 |
| Difference from VideoRouteAudit: lecture test, speech as main signal, automatic slide labels (no claim about their freezing) | P187 |

## Study design (`3_study.tex`)

| Claim | Ledger ID or source |
|---|---|
| At most 4 frames; same model, prompt and preprocessing within a comparison; 640 px decode cap before the processor | C30, C64 |
| 120B answers only LongVideoBench in the two main studies and also generated the lecture questions | C25, P188 |
| Figure: speech path = BM25 windows + MiniLM, or Head A on the full transcript; ~9 candidates on lectures; ≤4 frames (boxes given fixed text widths so the figure fits the column, P223) | C61-C62, P188 |
| Systems table and caption: S0 ≤120 words; R2/R3 per study; lecture R3-R5 up to 96 times; S1-S3 at most about 54 (table reflowed to the text width with `tabularx`; "Trained on" merged into "Trained part (data)", P223) | C67, C30; `docs/v3/m3av_systems.md` |
| S0 design | `docs/v3/m3av_systems.md`, P188 |
| S1/S1t rules; ~9 candidates on the lecture test | P188; `docs/v3/m3av_systems.md` |
| Head A: 21 + 3 bge-derived = 24 inputs; zero-init reproduces frozen line scores; top 64 lines, σ 4 s, 1 s grid, 8 s suppression, 6 moments; 1,889 per member, 5,667 for three; positive if line end − 0.3 s within 10 s | C61, C62 |
| S2c, S2m training sources; S2m internal validation; no dev/test lecture | C25, C65, P188 |
| Head B: 18 features incl. similarity to each "question + option" text; greedy gain = score + 2 × Δ decisiveness − 1 × max visual similarity; decisiveness on the mean softmax option profile | C63; `scripts/v3_head_b.py` (P216) |
| Baselines: fresh R2/R3 four-time versions with merging; lecture R2 four times; lecture R3-R5 96-time distinct fill (times at least 4 s apart); R5 adaptation (Gaussian around each line; exact rule in supplement D), best of 10 dev settings | C55, C67; `scripts/m3av_vsi.py` (P216) |
| Rules and families chosen on dev; head weights fitted on training data | P188 |
| Earlier locked test: 459 (55/6, 202 + 202 on 202 lectures) | C01, C05 (supplement pending) |
| Fresh set 894/500; 3 excluded; no method chosen on it before the run; not a test-server result | C09, P174 |
| Generation and filter rules: ≥4 s annotated, ≥8 OCR words, first segment excluded, image + OCR only; 4 distinct options; 80% token match; repeats need answer-critical tokens; dev/test 1 Hz edge correlation with the saved image of each valid segment (source or repeat), 0.55, neighbouring segments, ≥2 s; train not filtered by F4; hash order; no answer-model filter | C66; `scripts/m3av_visibility.py:segment_curve` (P218) |
| Official split built to keep speakers apart (dataset authors); lecture separation checked | C65 |
| hit@4 definition with false-match caveat; S-src sealed before test | C26, C31, P188 |
| Transcript control: at most 600 BM25-selected words | C32, P188 |
| Human check sample, packet, rule fixed before verdicts | C41 |
| Model reviews exposed false repeats and motivated S-src; agreement statistics only in the human check | research log step 89; P188 |
| Freeze timing: primary before each run; QA analysis after hits and before answers; human plan before verdicts | P188 |
| H1-H4, L1-L2, Holm per family; L1 R3→R5 dev-only amendment | C10-C13, C27, C28, P171 |
| Question-weighted differences; whole videos/lectures resampled and their questions pooled; unadjusted paired percentile intervals; Holm on p-values; one-sample bootstrap for human rates; smallest p about 0.0002 | P164, P188 |
| Failures count as wrong; one retried answer as exception | C37, P188 |

## Results (`4_results.tex`)

| Claim | Ledger ID or source |
|---|---|
| Fresh accuracies (27 values) | C15 |
| H1-H4 with Holm p (0.92 and 0.34 are roundings of 0.9185 and 0.3424) | C10-C13 |
| One failed answer each for S1t and S4t with 4B | C12, P189 |
| H3/H4: Holm p 0.0008; unadjusted intervals above zero | C12, C13, P189 |
| Post hoc decomposition +8.2 / +0.8 / +1.3; allowances, not exact lengths; specific to this swap | C21 |
| Head A alone +1.8, +1.2, +2.8 (0.0, 5.6), p 0.052 uncorrected; Head B within one point | C16 |
| Full minus residual-off +0.3 (−2.1, +2.8); "no confirmed contribution" | C18 |
| S1 − R2 (secondary, uncorrected): +7.5, +8.6, +5.4; S1 − R3: −0.8, −0.1, −1.1, "no demonstrated advantage", not a transcript-only contrast | C17, C14 |
| Lecture table: hit@4 and frames | C26, C30 |
| Lecture table: S-src for all systems | C31, C59 |
| Lecture table: accuracy for all systems and controls | C32 |
| Table caption: hit@4 includes valid repeats; S-src source segment only; both proxies | C26, C59 |
| L2 +5.3; L1 −0.5 with Holm p | C27, C28 |
| S2m package changes transcript scope, rescoring, density voting and residual | P170 |
| S-src L1 −1.5, L2 +5.0, descriptive, no Holm | C31 |
| S2m − R3 +15.2; − R4 +2.5 (interval includes zero); − S2c −0.4 ("no confirmed change") | C29 |
| R3, R4, R5 fewer than 4 frames for 52, 48, 27 questions (lecture 96-time rule) | C30, C67 |
| Accuracy: S2m − S1 +4.7; − R5 +0.7; S3 − S1 +0.2; S1t − S1 +0.9 | C33, C34 |
| Accuracy and hit@4 lead to the same inferential reading for the two declared comparisons; counter-examples: R5 above S2m on hit but below on accuracy; S2c − S2m paired +0.4 hit, +0.9 accuracy | C26, C32; `reports/m3av/hits_test.md`, `reports/m3av/qa_test.md` (−0.4, −0.9 as S2m − S2c); P201 |
| Transcript-only-wrong subset: 322 (failed control is unknown and excluded from the control-defined subset; failed system answers count wrong) +5.9 (1.6, 10.3), +9.6 (5.4, 13.8); as-run 323 +5.9, +9.3 | C35, C36; `docs/v3/m3av_qa_test_deviation.md` |
| Loss table for S1 and S2m, rows labelled as interval hits | C39 |
| Loss table for S0 (75.3, 24.7, 7.9, 67.4) | C60 |
| +7.8 candidate difference vs +5.3 final; bookkeeping, not causal | C39, P189 |
| S2m accuracy by outcome 677/93.9, 82/56.1, 94/66.0; association; hit may be false; 41 wrong unadjudicated | C40 |
| Human check: 67 double, 33 single, 167 verdicts; 97/0/3; rejections about the key | C41 |
| Rejected-or-uncertain 3.0 (0, 7.0) lecture-weighted and 2.7 (0, 6.4) with each sampled lecture weighted by its number of test questions; sample estimates, not full-cohort observed rates | C42; `scripts/m3av_human_review.py:rates` (P217) |
| Masks: L1 −0.4 (−3.7, +3.0), L2 +5.4 (+2.8, +8.0); estimates move, interpretation unchanged | C45, P189 |
| All-accept reviewer, impossible combined reject, degenerate interval, kappa, owner's account of independence, device use unrecorded, frames not reviewed | C44, P168, P180; `reports/m3av/human_review_limits.md` |
| Pool time S1 0.67 s, S2m 0.97 s, R5 1.42 s; exclusions listed | C50; `reports/m3av/profile_saved.md` |
| New answer calls S1 2.47, S2m 2.41, R5 2.44, S0 3.09, S1t 4.55 s; cache reuse excluded; not matched | C51; `reports/m3av/profile_saved.md` |
| Dense index 955 s for 112 lectures, mean 19.3 min | C52 |
| Observed reserved peak 16.8 GiB on 15.93 GiB; one recorded OOM and restart; first-session peak not captured | C38 (qualified), C37 |
| Capped dev replay: 600 identical replies, 14.19 GiB reserved, cap 14.28 GiB | C48; `reports/m3av/profile_gpu.md` |

## Discussion and limitations (`5_discussion.tex`)

| Claim | Source |
|---|---|
| S1 vs R3 on fresh LongVideoBench: no demonstrated advantage; not a transcript-only contrast | C14, P190 |
| Lecture results look different; no formal cross-dataset test | P190 |
| Candidate share 81.2 to 89.0 (interval hits) | C39 |
| R5 vs S2m: 79.8 vs 79.4; neither shown to differ nor to be equal | C26, C27 |
| Residual: no confirmed change; lecture training: no confirmed change vs CG-Bench training | C18, C29 |
| "A plausible reading ... does not isolate" (hypothesis, scoped to tested rules and models) | P190 |
| Accuracy and proxy agree for the two declared lecture comparisons (numbers now only in Sec. 4.2, referenced) | C28, C33, C27, C34 |
| Same frozen pools: perfect picker at most 9.6 points of hit@4; no bound for a new candidate stage; no accuracy bound | C39, P190 |
| Reporting suggestion builds on earlier evidence-evaluation work (no first claim) | P190 |
| Limitations: proxy labels, generated train/dev/test questions, human-check limits (all-accept reviewer, 33 single votes, owner's account of independence, frames not reviewed; device use stated in Sec. 4.4), packages, lecture search work (up to 96 vs at most about 54), R5 adaptation, one 4B lecture model, shipped transcripts, post hoc analyses | P170, P171, P174, P178, P190 |
| Deviations: per-ledger caps, 25,373 charged incl. earlier-test API follow-ups; retry, 322 vs as-run 323, full-cohort accuracies unchanged (rule explained in Sec. 4.2); observed reserved memory over capacity, one recorded OOM | C53, C37, C38 (qualified) |
| Conclusion: package gain; labelled-interval placement; no demonstrated advantage over R5; heads no confirmed accuracy; measurement lesson as opinion | C12-C13, C27-C28, C10-C11, P190 |

## Statements (`6_statements.tex`)

| Item | Status |
|---|---|
| Authors' roles (idea, framework, experiment plan, direction, checking, review and editing) | owner-attested; not certifiable by the audit (P191, P195) |
| Claude via Claude Code: code, experiments, related-work search, drafting | disclosed (literature reading for the plan: research log, start) |
| OpenAI GPT through the Codex app as a separate AI auditor (numbers, statistics, wording, related-work search); not human validation | disclosed; Codex named from the auditor's own workspace (its delivery folder sits under the Codex app's documents folder); owner may still veto naming |
| Model versions not recorded for every stage; tools and families named only | disclosed (P226: no guessed historical versions) |
| Nemotron generated train, dev and test questions; also an answer model | disclosed (P191) |
| Model reviews by Claude and GPT; AI reviews, not human judgments | disclosed (P217) |
| Qwen3-VL and 120B answer models are objects of study | disclosed |
| Human-study consent and oversight | TODO in `3_study.tex` and `7_supp.tex`: owner to supply (P180, P195); not invented |

## Not used anywhere in the draft

C38 as a blanket claim, C43 as an upper bound or error bound on the labels (its degenerate 0% interval is disclosed in Sec. 4.4 and supplement E with that qualification, which is allowed), C54, C56; "first", "state of the art", "equivalent", "matches", "nothing measurable",
"fully local", "97% of labels verified", "<0.0001"; any hardware ceiling on frames; "the VLM can answer only if the
moment is shown"; "high floor"; "covering evidence for different options" for Head B; a universal "R3 decodes 96";
an 11.0-point ceiling for a perfect candidate stage; any claim that VideoRouteAudit lacks freezes.

## Supplement (`supp.tex`, `sec/7_supp.tex`)

The overview is unnumbered (`\section*`), so the substantive sections render as A-G, matching this map (P218).

| Section / claim | Ledger ID or source |
|---|---|
| Labels: declared = fixed in a protocol before the test (timing only); Holm only within H1-H4 and L1-L2; earlier test analysed with unadjusted paired intervals, no correction added; secondary, descriptive, post hoc kept distinct; unadjusted intervals | P186, P188, P214; `docs/v3/final_test_protocol.md`, `scripts/build_results_table.py` |
| A. Earlier test setup: 459 = 55/6 CG-Bench + 202 + 202 Video-MMMU on 202 lectures; CG subtitles; Video-MMMU Whisper large-v3-turbo; 2B, K ≤ 4; primary S1 − S0 per set and pooled with unadjusted intervals; secondaries uncorrected; 208 clusters | C01; `docs/v3/final_test_protocol.md` |
| A. S5: four sets hybrid/mmropt (S1), hybrid_ahead/mmropt (S2), hybrid/optset, hybrid_ahead/optset; optset = Head B's greedy gain with z-scored MobileCLIP question+options similarity as relevance, weights chosen on CG-Bench train inner validation; untrained confidence rule but trained Head A in two sets, so not training-free; no learned Head B (only in the unreported six-set extension); up to four answer requests per question, ≤4 frames each, identical requests reused; not a matched single-request budget; table caption says so | C68 (S5 accuracy validated); `scripts/v3_baseline_cycle.py:VERIFY_SETS`, `scripts/v3_head_b.py:choose` (P215) |
| A. Table of 36 accuracies (R1-S5 × CG/P/C/pooled) | `docs/report/all_results.csv` rows 284-324, 533-555 (new; canonical seed 20261019) |
| A. S1 − S0 per set +16.4 (7.0, 26.0), +1.0 (−5.0, 6.9), +2.0 (−3.5, 6.9); pooled +3.3 (−0.5, 6.9) | C02, C03; `all_results.csv` rows 289, 307, 317, 538 |
| A. Pooled secondaries S2 −1.7, S3 0.0, S4 0.0, S5 +0.9, S1 − R3 +4.6, S1 − R2 +5.0 with intervals; "no confirmed gain" (not "no improvement"); CG heads −9.1, −16.4 as secondary point estimates on six videos | C69; `all_results.csv` rows 547, 534, 543, 554, 533, 551, 294, 296 (P214) |
| A. 4B follow-ups 50.3/52.7/55.6; +5.2 (1.1, 9.4); S1t − S1 +2.8 (0.0, 5.5), p ≈ 0.050; policy chosen after test seen | C05, C06 |
| A. Head A dev 36.6 vs 28.0 on 164/17; family chosen there | C08 (reworded as dev observation) |
| B. Failed answers TMe7oXMJoSM_0 in S1t and S4t (4B); skip rule replaced | C12; `docs/v3/confirmation_deviations.md` |
| B. Secondary table: S2 − S1, S3 − S1, S4t − S1t, S1 − R3, S1 − R2 | C16, C14, C17 |
| B. Residual ablation fresh +0.3 (−2.1, 2.8), dev +3.7 (−3.7, 10.8) | C18 |
| B. λ 0 − 2: 0.0 (−0.9, 0.9), +0.6 (−0.2, 1.4); identical sets 785/792; fewer than 4 frames for 81/82 | C19; research log step 85 (81/82 new) |
| B. Text-policy table 57.3/57.0/56.7/57.7; words 425/425/425/214; differences | C20 |
| B. Decomposition table +8.9, +8.2, +0.8, +1.3, +2.8, +6.2; no interaction claim | C12, C21, C22 (reworded) |
| C. Eligibility ≥4 s annotated, ≥8 OCR words, not first segment; prompt verbatim; OCR cut to 3,000 characters; image + OCR only; hash order | C66; `scripts/m3av_questions.py` (prompt, 3,000-character cut: new) |
| C. Filters: parse, 4 distinct options, valid index, 80% evidence words in OCR; repeats need 80% + answer-critical tokens (key words on source slide, digit tokens); F4 per valid segment against that segment's own saved image (so a false repeat can pass), 1 Hz edge correlation ≥ 0.55 and not below either non-valid neighbour, accepted frames of all valid segments ≥ 2 s, threshold chosen on dev; scoring separate: hit@4 with 1 s tolerance; train not F4-filtered; no answer-model filter | C66; `scripts/m3av_questions.py`, `scripts/m3av_visibility.py:segment_curve`, `stage_apply` (P218) |
| C. 158/853 (18.5%) test questions with a repeat; 14 sampled, one model reviewer judged 7 false, the other more | research log step 89 (new) |
| C. S-src sealed before test; 3 source-empty test questions | C31 |
| C. Cohorts and lecture separation | C25, C65 |
| D. Shared inputs: 360p, shipped transcripts, 640 px cap, 2 s MobileCLIP index; decodes ≤30 / ≤54 / ≤96 | C64, C67; `docs/v3/m3av_systems.md` |
| D. Realized inputs table: frame means; median words, prompt-text tokens, visual tokens | C30; `reports/m3av/qa_test.md` (medians per `scripts/m3av_qa.py`; new) |
| D. R3/R4/R5 underfill 52/48/27 | C30 |
| D. Head A features, MLP 24→32→32→1 with GELU, zero-init last layer, top 64, σ 4 s, 1 s grid, 8 s suppression, 6 moments, line end − 0.3 s within 10 s, 3 members × 1,889; S2m 15% inner validation picks epoch | C61, C62; `src/videoqa/v3/head_a.py`; research log step 86 area (15%: new) |
| D. Head B 18 features (option statistics over "question + option i" texts) and greedy gain, λ 2, μ 1; option profile softmax(50 × cosines), averaged over selected frames, decisiveness = top minus second | C63; `scripts/v3_head_b.py:stage_features`, `decisiveness`, TAU = 50 (P216) |
| D. R5 text/visual branch, boost B = min(1, 2M) above 0.5; only B > 0.2 contributes; Gaussian centred on the line midpoint, sd = (end − start + 2W)/4, W = 2 s, scaled by B; max over lines; 0.5/0.5 z fusion; ranked times ≥ 4 s apart, up to 96, then perceptually distinct fill; VSI differences; VSI-inspired, no claim about VSI; best of 10 dev settings, 74.7% at four raw times | C55, C73; `scripts/m3av_vsi.py:text_score`, `ranked` (P216) |
| D. Four-time R3/R5 on lecture dev showed 1.4 and 2.0 frames; hash threshold 0 vs 3 chosen on dev; LVB versions unchanged | research log (new) |
| E. Human procedure; cutoff 8 Nov 2026 23:59:59 Asia/Dhaka; receipt times from retained, privately kept group-chat evidence; no self-contradiction, no clarification, 3 no-majority items not adjudicated; organizer's answers excluded; pseudonyms H1-H3. Votes H1 100 accepts / H2 items 1-34, 31 accept 3 reject (KEY) / H3 items 35-67, 33 accept 1 note; 68-100 single; impossible combined reject; 97/0/3 | C41, C44; `reports/m3av/human_review_limits.md` ("Timing and provenance"); `scripts/m3av_human_review.py:CUTOFF` (P218) |
| E. Agreement table 82/0.019, 77/0.074, 84/0.525, 31/34 κ 0, 33/33 undefined; caption: model reviews are AI reviews; near-zero/undefined κ only in rows with human verdicts; AI-AI κ 0.525 | C44 (P217) |
| E. Rates: estimator Σ n_l r_l / Σ n_l over the 100 sampled lectures (n_l = lecture's questions in the 853 cohort, r_l = reviewed verdict rejected or no majority); 2.7 (0, 6.4) cohort-question-weighted, 3.0 (0, 7.0) lecture-weighted; one-sample lecture bootstrap; sample estimates, not full-cohort observed rates; 0% degenerate | C42, C43 (disclosed, stated as not an upper bound); `scripts/m3av_human_review.py:rates`, `boot_rate` (P217) |
| E. 850-question mask L1 −0.4, L2 +5.4; S-src −1.4, +5.2; QA 82.6/87.3/86.5 | C45, C46 |
| E. Independence: organizer's account; not enforced; device use not recorded; consent TODO | P180, P205 |
| F1. Ledgers 7,587 / 7,587 / 10,199 = 25,373; per-ledger caps; API includes old-test follow-ups; 8,046 fresh API answers; cache, re-ask and retries | C53; `docs/v3/confirmation_deviations.md` |
| F2. Sealed rule; what happened; full-cohort unchanged; 322 vs 323 on 104 lectures; sensitivity table; 10,236 rows = 9,893 + 343 | C35, C36, C37; `docs/v3/m3av_qa_test_deviation.md` |
| F3. Session-2 reserved 16.8 GiB, allocated 11.49 GiB, card 15.93 GiB, session-1 peak missing; uncapped 17.26/15.23/1.56; capped 14.28 cap, 600 identical, 14.19/14.13/0.08; 1,517.6 vs 1,619.5 s, no causal claim | C38 (qualified), C48, C49; `reports/m3av/profile_saved.md` (frozen session 2), `reports/m3av/profile_gpu.md` (capped and uncapped dev replay) (P218) |
| G. Timing table: pool means, new calls, answer means; exclusions; index 955 s, 579 frames, 19.3 min, 8.4 s per lecture; model load 8.5 s | C50, C51, C52; `reports/m3av/profile_saved.md` (per-condition rows beyond S1/S2m/R5/S0/S1t: new) |
