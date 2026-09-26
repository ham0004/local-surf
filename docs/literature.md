# Literature audit (M0)

Audit date: **2026-09-26**. This is a focused, targeted search, not a systematic review.
New papers appear weekly; re-run the search before any submission.

## How to read the "Checked" column

| Level | Meaning |
|---|---|
| **method** | Method section of the HTML/PDF was read for the specific question that matters to us (training? subtitles? corruption?). |
| **abstract** | Only the arXiv abstract page was read; details may be missing. |
| **pdf-only** | Metadata taken from the research-proposal PDF (`video_qa_research_framework.pdf`) and **not re-verified** in this session. Treat as unverified. |

Venue status vocabulary: *verified proceedings*, *author-reported* (claim on arXiv only),
*preprint*, *withdrawn*.

## The question we asked of every paper

Our hypothesis (see [novelty_matrix.md](novelty_matrix.md)) is:

> A small controller, trained on the **same video + question** under three transcript
> states (CLEAN, TARGETED_DAMAGE of answer-relevant speech, MATCHED_CONTROL_DAMAGE of an
> equal amount of unrelated speech), learns to spend visual effort *because answer-relevant
> speech is missing*, and not merely because *some* speech is missing.

So for each paper we asked: (1) is anything trained? (2) are subtitles/transcripts inputs?
(3) are transcripts damaged, and if so, is damage targeted vs. a matched control?
(4) is compute cost part of the objective?

## Closest prior work (read first)

| ID | Paper | Date / venue | Checked | Trained? | Uses subtitles? | Transcript damage? | Overlap / distinction |
|---|---|---|---|---|---|---|---|
| F10 | Modality Relevance is not Modality Utility (Li & Gai) — arXiv [2607.05438](https://arxiv.org/abs/2607.05438) | 2026-07-03, preprint | abstract | Calibrated value-of-escalation router | Text+tables+images RAG, **no video** | No paired interventions | Closest *idea*: "relevance ≠ utility", escalate to vision only when worth the cost. Distinction: we operate on long video with timed ASR, and supervise with paired targeted/control transcript damage. |
| — | Caption-once, Frames-on-Demand: Visual-Need Routing (Cai et al.) — [2609.11899](https://arxiv.org/abs/2609.11899) | 2026-09-10, author-reported EMNLP 2026 Main | abstract | Not stated in abstract | Offline captions (narrative index) | None mentioned | Very close *system*: gate that triggers bounded keyframe retrieval only when a question needs perception; per-query frame cap. Distinction: routing is by question type over captions, not by transcript *reliability*; no paired damage supervision. **Must read full method before claiming anything.** |
| T2 | REVEAL (Yan et al.) — [2608.08612](https://arxiv.org/abs/2608.08612) | 2026-08-09, preprint | method | **No** ("Without any extra training") | Human subtitles as-is; no ASR | **Never** removed/corrupted | Builds contrastive decisive E+ vs. topically-relevant non-decisive E− with validity condition `A(q,E−)≠y, A(q,∅)≠y, A(q,E+)=y`. That contrast *pattern* is prior art for us; our contrast is over transcript states, and we train. |
| F6 | ReaSon (Zhou et al.) — [2511.12530](https://arxiv.org/abs/2511.12530) | AAAI 2026, verified in PDF | method | RL (composite reward) | **No** ("all evaluations without using subtitles") | No | Counterfactual reward = KL between VLM outputs on selected vs. *inverted* frame subsets. Counterfactual on *frames*, not transcripts. Required baseline for "ordinary counterfactual utility training". |
| T7 | VSI (He et al.) — [2508.06869](https://arxiv.org/abs/2508.06869) | v4 2026-04-10, author-reported CVPR 2026 Findings | method | **Training-free** | Yes (subtitle matching) | Robustness test only: "No Subtitle" and "Noisy Subtitle"; construction not specified; not targeted | Shows transcript-guided selection degrades gracefully. We study *learned* reaction to targeted vs. control damage. |
| — | Q-Gate: Where to Focus (Wang et al.) — [2604.17422](https://arxiv.org/abs/2604.17422) | 2026-04-19, preprint | abstract | **Training-free** (LLM in-context gating) | Yes ("Contextual Alignment for subtitle-driven narratives") | Not mentioned | Routes queries between visual and subtitle streams by *query intent*. We condition on *observed transcript evidence state*. |
| T3 | EviSelect (Zhang et al.) — [2608.05780](https://arxiv.org/abs/2608.05780) | 2026-08-06, preprint | abstract | GRPO, joint accuracy–efficiency reward | Not mentioned | Not mentioned | Joint timestamp/rate/resolution policy with cost-aware reward: joint budgeting is established prior art. |
| T1 | LongVideoAgent (Liu et al.) — [ACL 2026](https://aclanthology.org/2026.acl-long.1876/) | ACL 2026 | pdf-only | Master LLM RL (format + correctness) | Subtitle-conditioned grounding agent | — | Architecture-level precedent for "text brain + vision tools". |

## Other relevant work

| ID | Paper | Date / venue | Checked | Note |
|---|---|---|---|---|
| — | LongVideo-R1 (Qiu et al.) — [2602.20913](https://arxiv.org/abs/2602.20913) | CVPR 2026 | abstract | Qwen3-8B navigator over hierarchical captions; SFT on 33k GPT-5 trajectories + RL. Low-cost navigation is established. |
| — | TRACE (Liu et al.) — [2608.22516](https://arxiv.org/abs/2608.22516) | author-reported EMNLP 2026 | abstract | Training-free; stops when the answer stabilises across rounds. Stopping-by-stability baseline. |
| — | CRAFT — [2605.19075](https://arxiv.org/abs/2605.19075) | 2026-05, preprint | abstract (search snippet) | Per-video ASR + dynamic keyframe selection + critic repair loop. |
| — | Select, Compress, Reinvest — [2609.03820](https://arxiv.org/abs/2609.03820) | 2026-09, preprint | search snippet only | Controlled study of visual-token allocation; relevant to our cost-matched evaluation. |
| — | Beyond Frame Selection — [2608.05592](https://arxiv.org/abs/2608.05592) | 2026-08, preprint | search snippet only | To read. |
| — | INFACT — [2603.11481](https://arxiv.org/abs/2603.11481) | 2026-03, preprint | search snippet only | Diagnostic benchmark that includes subtitle character errors and desynchronization — useful realistic-damage reference. |
| T10 | AutoSkill: One Skill Does Not Fit All — [2609.12517](https://arxiv.org/abs/2609.12517) | 2026-09-11, preprint | search snippet | Routes among frame-selection skills by question taxonomy. |
| T6 | GCR — [2608.01660](https://arxiv.org/abs/2608.01660) | 2026-08-03, preprint | search snippet | Ground timestamped text, cover, refine omitted evidence. |
| F1–F5, F7–F9, F13, F14 | FFS, Frame-Voyager, Q-Frame, A.I.R., HORNet, ReQuest, CSES, RIDGE, FrameOracle, HFS | see PDF pp. 31–32 | pdf-only | Learned/adaptive frame selection: selection itself is **not** a novelty claim. |
| T8, X3 | EMCompress, LLMLingua-2 | ACL 2026 Findings / ACL 2024 Findings | pdf-only | Prompt compression baselines. |
| X1 | BCEA — [2606.16667](https://arxiv.org/abs/2606.16667) | preprint | pdf-only | Budgeted conformal acquire/answer/abstain. We make **no** conformal claim. |
| D7 | CourseTimeQA — [2512.00360](https://arxiv.org/abs/2512.00360) | **WITHDRAWN** 2026-06-02 (per PDF) | pdf-only | Do not use its numbers. |

Papers the user already reviewed (`video-QA/papers.md`): T*, LongVideo-R1, HAVEN-style audiovisual
entity cohesion, WorldMM, MemDreamer, LensWalk, Symphony, Think-Then-Verify, TRACE. They fall
in the two families noted there (memory-then-iterate; hypothesis-then-agentic-search). None of the
ones checked here trains on transcript damage.

## Gaps in this audit (honest list)

- Caption-once/Frames-on-Demand, F10, EviSelect and Q-Gate were read at **abstract** level only.
  Their full training objectives must be read before writing a related-work section.
- ReaSon and REVEAL method sections were read for our specific questions only.
- Search was via general web search plus arXiv pages; ACL Anthology / OpenReview / CVF were not
  crawled systematically in this session.
