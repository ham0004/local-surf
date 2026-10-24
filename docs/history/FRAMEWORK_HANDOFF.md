# Video QA research — continuity and implementation handoff

Updated: 2026-10-05. Repository: `D:\local-surf`. Branch observed: `framework-2`. Latest observed commit: `f0ca00a` (documentation report); implementation findings below were checked against the current tree and previous audits.

## Purpose and limits of this handoff

This is a portable decision record for Claude Code or a new Codex conversation/account. It summarizes the research discussion, verified implementation, important results, and the user's latest clarified goal. It is NOT a verbatim conversation export, a complete account backup, or a claim that all historical chat context is included. This file concerns ONLY the video-QA research project. Other, unrelated chats have separate handoffs and must not be merged into this project's context.

Creating this file does not implement the proposed changes. No training, downloads, paid API calls, cloud GPU provisioning, commits, or pushes were performed as part of preparing it. Existing reports describe the OLD tested architecture unless explicitly stated otherwise.

## Latest clarification: flexible research, carried out step by step

The user's governing intent is to work like a researcher, not follow a predetermined algorithm recipe. The core problem is efficient, credible video QA; the two complementary evidence paths are the proposed starting architecture. Specific encoders, embeddings, scoring functions, head architectures, losses, training/post-training techniques, temporal proposals, sampling rules, transcript coverage and frame counts remain research choices. Concrete methods and trial counts elsewhere in this file are illustrative starting points, not compulsory designs or exhaustive limits.

The work should proceed sequentially:

1. Understand the current system, relevant literature, credible datasets and evaluation requirements.
2. Establish appropriate baselines and verify that the benchmark/scorer can support the intended conclusions.
3. Identify a meaningful limitation or research opportunity, then formulate a justified method, including an original design where defensible.
4. Implement and train that method when training is required; benchmark it against the relevant baselines under a declared fair protocol.
5. Examine uncertainty, ablations, failure cases, generalization and cost. Explain what the results do and do not establish.
6. Use that evidence to choose the next experiment, which may explore another component or method. Preserve earlier results and record the reason for each change.

The research scope is not restricted to frame sampling or a particular neural head. Any relevant improvement location in the framework can be considered and prioritized with reasons. Necessary ablations, reproduction checks, robustness tests or further training should not be omitted merely because they were absent from the initial shortlist. Conversely, considering the relevant search space does not require exhaustively running every combination.

The previously suggested fixed/frozen component setup is an attribution-friendly starting control, not a permanent prohibition on justified encoder/adapter or answerer experiments. If those change, use explicit separate comparisons so gains are not incorrectly credited to a selector. Explain consequential architectural changes. Replacing the research objective itself with an unrelated task requires a separate discussion; testing relevant framework variants is already within the user's requested scope.

### Framework variants are allowed when justified

The user's latest clarification permits changing the proposed framework itself experimentally if there is a credible reason that a different organization could improve answer quality, efficiency or both. Do not treat the diagram below, the existence of exactly two paths, a particular fusion stage or a fixed module arrangement as an untouchable constraint. The original proposal remains a preserved reference architecture, not a discarded history.

Before a variant, record the limitation it addresses, the architectural change, expected mechanism and fair comparison. Preserve the original configuration/results and compare the variant against it with compatible data, scoring and declared resource budgets. Distinguish gains from more frames/text/compute from gains due to the new architecture. Develop and select variants on development data; reserve independent final evaluation. A plausible variant may be worth a bounded test without prior proof it will win.

The user permits either small coherent commits on the current branch or a separate branch/worktree when isolation would help. Branches are optional, not a ceremony required for each experiment. Keep baseline and variant configurations recoverable, respect unrelated in-progress work, and document which version produced each result. This permission for research variants does not authorize unbounded paid compute or silently replacing the research objective.

Experimental flexibility applies to method design and development decisions, while credible evaluation still requires independent held-out testing, honest reporting and an explicit compute budget. A final test set must not become an iterative method-selection set. No particular outcome, optimizer, model size or world-best/novelty claim is precommitted.

### Three distinct experimental tracks

The user distinguishes three activities, all of which belong in the research plan:

1. **Method innovation:** formulate, train and evaluate new Head A/Head B algorithms, representations or training objectives, with relevant established methods as controls.
2. **Training hyperparameters:** select suitable optimization settings for each trainable method, such as learning rate or regularization, using a declared development protocol.
3. **Framework operating choices:** test assumptions about window coverage, frame sampling interval/placement, uniform versus other sampling, candidates contributed by each path, final frame counts, and surrounding transcript amount/coverage. These choices require empirical comparison in the baseline and, where relevant, again after adding the trained methods. They are not a replacement for method innovation.

The immediate priority is to keep the proposed overall framework stable while conducting these internal experiments. Architectural variants remain a later option if a specific limitation motivates them; permission to change the framework is not an instruction to redesign it first. Record why each setting was tested and its result, then use development evidence to choose the next trial.

### Same credible benchmarks for every compared method

Terminology: a **benchmark** is the dataset plus its evaluation task/protocol; a **baseline** is an existing or established method/system used as the comparison reference. A **base model** is the pretrained model used inside a system. These are not interchangeable. The baseline should be a functioning, reasonably tuned pipeline, not a deliberately weak raw-model comparison.

The user's requirement is direct, common-benchmark comparison: every baseline and every substantive trained method/framework variant included in the study must be evaluated on the SAME chosen credible benchmark(s), with the same eligible test questions, references, scorer policy and declared resource-comparison protocol. The final comparison must not consist of testing only the winning method, or testing different methods on different datasets and comparing their absolute scores.

The distinction is between **validation/development and held-out test examples**, not a requirement to invent synthetic development questions or use a different benchmark for each method. Where a benchmark supplies suitable official splits, use them. If it lacks a usable development split, declare an appropriate video-disjoint development/test protocol before experimentation, report any resulting custom subset clearly, or use a separate development source. Respect the benchmark's official evaluation rules.

All configuration trials can be compared on the same credible benchmark's development split under consistent conditions. Once configurations are chosen, freeze EACH method's setup and run ALL declared final baselines, methods and relevant ablations on the same held-out test set. Each method may receive fair development tuning; do not tune a new method extensively while leaving an intentionally weak baseline. Preserve negative and inconclusive results instead of dropping them from the final comparison. If multiple benchmarks are in scope, evaluate the declared methods across the same benchmark suite or explicitly mark missing runs and their reasons.

Example structure: baseline, Head-A-only upgrade, Head-B-only upgrade and combined method each have development-selected configurations; all four then receive the same final test questions and scoring. A sampling-only improvement can be another explicit comparison arm. Final test scores are reported side by side with quality, uncertainty and measured cost. The exact arms depend on the justified research plan, not this example.

Repeatedly adjusting settings after seeing final test scores converts that set into development data. That restriction does not prevent testing every declared method: freeze their configurations before final evaluation. A separate additional benchmark is optional evidence of generalization, not a substitute for the common-benchmark comparison.

### Explicit per-method research and tuning cycle

This is the user's requested operational workflow, not a replacement for the literature-driven original-design requirement:

1. **Baseline cycle:** start with the current system/base models and appropriate established comparison methods. Test justified framework settings, sampling rules, evidence quantities and other relevant values on the common benchmark development protocol. Save all trials and select a strong baseline configuration according to a declared answer-quality/cost criterion.
2. **Research and design:** read relevant primary papers, including useful ideas from other fields. Select or formulate a small number of promising methods, for example two or three initially. Explain why these designs deserve implementation. Do not choose only familiar published methods; retain explicit scope for original Head A/Head B mechanisms and training formulations. Call them proposed original contributions until prior-work checks establish the degree of novelty.
3. **A full cycle for EACH candidate method:** implement it; train it where required; run development benchmarking; examine results; tune its training settings and the relevant framework operating choices; retrain when the changed configuration requires it; and retain that method's best justified configuration. Repeat this cycle for the next method. Methods need not all share one arbitrarily fixed internal setting, but they must be compared under a fair declared resource protocol.
4. **Document EACH method, not only the winner:** method rationale, borrowed/new parts, module/code changes, supervision, configuration search, all completed trial outcomes, selected configuration, costs, failure analysis and limitations. Record dataset/split/model/scorer/config/code versions and enough detail to reproduce the result. A failed trial stays a failed trial; an unrun idea stays future work.
5. **Common final comparison:** freeze the best development-selected configuration for every included method and the baseline. Evaluate all of them on the SAME held-out benchmark questions using the SAME scoring policy, report results side by side with uncertainty and cost, and identify the best observed result under the stated criterion. Do not report only the winning method or retune against these final scores.

Here, "best" means the declared answer-quality/efficiency objective, not the longest answer or a configuration selected by hindsight from final test scores. Give baselines fair tuning and account for differences in search effort. The exact number of proposed methods is flexible with available evidence and compute. The workflow is sequential, transparent and reproducible; it is not an instruction to run an exhaustive search of every paper or parameter combination.

### Final external comparison: quality versus resources

After the internal study, the user wants a meaningful comparison with a suitable strong locally runnable/open-weight model or a proprietary hosted model/system. A more expensive comparison can be a later bounded stage if early results justify it; do not treat that as permission to conceal unfavorable completed comparisons. If the framework underperforms, report the measured gap and the experiment's limitations honestly. Useful negative results are an acceptable outcome.

Choose external comparators for task suitability, strength and feasibility, not because they are easy to beat. Use the same credible benchmark questions and reference/scoring policy. Record exact model/API versions and dates, inputs, decoding settings and task modality. Distinguish two possible comparisons:

- **Controlled selection comparison:** hold the answer model, question set and evidence budgets fixed; vary how evidence is selected. This is the main way to attribute gains to the proposed selection method.
- **End-to-end system comparison:** compare the proposed pipeline with a larger local or hosted system, possibly with different supported inputs/budgets. Report those differences openly. This measures the full systems' quality/cost tradeoff; it does not by itself prove that a particular selector or training objective caused the difference.

Report answer quality with uncertainty and the resources that can actually be measured: full pipeline latency, local peak GPU memory where applicable, frames/visual tokens and transcript tokens processed, API calls and actual or documented estimated monetary cost. Include retrieval, frame decoding, feature extraction, selection and answer generation rather than counting only the final VLM call. State cold versus cached/warm conditions and hardware. Frame/token counts alone do not establish lower compute across different model architectures.

For a hosted proprietary model, internal GPU allocation, training compute and model size may be undisclosed. Do not invent these quantities or assume an API price proves a specific hardware-cost difference. Use observable client latency, documented/request usage and billed costs; mark unavailable resources as unknown. Local hardware running cost and API fees are different accounting bases and must be labeled accordingly.

Account for this project's training and label-generation costs separately from per-question deployed inference cost. If claiming that training pays for itself at sufficient query volume, show the assumptions and break-even calculation. If using a model-based judge, validate its scoring and consider possible preference toward its own model family rather than treating it as unquestionable ground truth.

The final claim must match the result: for example, higher quality at comparable cost, comparable quality at lower measured cost, or lower quality with a quantified resource tradeoff. A claim of comparable quality needs an explicit tolerance and uncertainty; failure to detect a difference alone is not proof of equivalence. If worse on both measured quality and cost, state that plainly. No favorable outcome is presumed.

## Continuity: what we built, what the review found, what comes next

This handoff builds on **Framework 2 final evidence review**, completed on 5 October 2026. The complete historical review is included alongside this file as [FRAMEWORK_2_EVIDENCE_REVIEW_2026-10-05.md](FRAMEWORK_2_EVIDENCE_REVIEW_2026-10-05.md), so continuing the work does not depend on access to the old Codex conversation or its external workspace.

The sequence is:

1. **Completed engineering:** a working transcript-guided video-QA pipeline, two candidate paths, cached features, a frozen final answer VLM, and reproducible experiment infrastructure.
2. **Completed experiments:** small learned utility heads and several cheap alternatives were tested. Visual evidence helped compared with speech alone, but the trained selectors did not demonstrate an advantage over simple selectors. The MIT synthetic question set had serious answer-option bias.
3. **What the review established:** the results and limitations documented in the historical review remain evidence. They do not establish that a genuinely learned time proposal and richer multimodal set selector were implemented or tested.
4. **The user's later clarification:** keep the two-path framework; make Head A actually learn useful temporal positions, make Head B combine visual and speech evidence to choose a useful frame set, and use credible published OPEN-ENDED QA for the primary objective. Four frames was an example, not a mandatory architecture limit.
5. **Now proposed, not implemented:** reuse the existing pipeline while replacing the missing learned temporal proposal, improving the multimodal set-selection representation/supervision, and adding valid open-ended evaluation. Implementation details below are starting designs to test, not proven improvements or finalized novelty claims.

| Area | Implemented and tested so far | Intended change | Reuse/status |
|---|---|---|---|
| Broad retrieval | BM25 transcript windows | Retain initially; measure missed-evidence windows | Reuse |
| Path A proposal | Frozen text relevance; frame fixed near utterance end | Train a question-conditioned time/subinterval predictor | New proposal training and integration needed |
| Path B scan | Fixed-step scan with cap; balanced variant already tested | Duration-proportional allocation under an explicit fixed scan budget | Controlled engineering baseline, not automatic novelty |
| Pool and deduplication | A/B merge, timestamps, hashes, cached features | Retain with safeguards against removing useful small changes | Reuse and validate |
| Head B | Small aggregate-feature utility scorers; no demonstrated gain | Context-conditioned multimodal set selection, with informative visual inputs | New representation/supervision pilot needed |
| Final answerer | Frozen Qwen VLM; experiments score MCQ correctness | Keep frozen initially; produce and score open-ended answers | Generation interface reusable; v2 quality scorer needs work |
| Transcript context | Fixed short retrieved excerpt | Budgeted local verbatim speech around selected evidence | Explicit policy change; isolate its effect in comparisons |
| Primary evaluation | Published LVB MCQs and small synthetic MIT MCQs | Accessible published open-ended QA with trustworthy reference answers | Dataset/scorer readiness still pending |
| Old labels | Single-frame or fixed-top-three fourth-frame utility, tied to old inputs | Temporal and contextual-set supervision matching actual new inputs | No automatic reuse of correctness labels |

### How to read the historical review without reviving an outdated plan

The historical review's section 2 explains **what the old system currently does**; its result tables and limitations explain **what has been measured**. Its section 9 recommended an answer-option-rotation label diagnostic while keeping the old MCQ architecture unchanged. That recommendation preceded the user's explicit open-ended and learned-timestamp clarification. It remains a possible LEGACY measurement audit, but it is **not the main next implementation task** in this handoff.

The review's warning against simply enlarging an unsuccessful scorer still applies. The proposed next design changes the temporal action space, inputs and supervision; its pilot must establish useful targets and baseline headroom before spending on larger models. Neither the review nor this handoff establishes a publication guarantee.

For continuity, the historical evidence and the current intended specification are complementary: the former records what happened; this handoff records where the user now wants to take the existing framework.

## Read this first: the user's actual objective

Build an efficient video question-answering system that selects useful visual evidence and relevant verbatim transcript, then produces an open-ended, sufficiently detailed answer. The central research direction is TWO complementary candidate paths and TWO meaningful learned components:

1. Head A learns question-conditioned times or subintervals where useful visual evidence is likely, using timed speech representations.
2. Head B learns to select a useful frame SET from the two paths using question, visual evidence, local speech, and the frames already selected. The final count K is configurable; four was the user's example and a historical experiment setting, not a fixed requirement.

The user does not want the project reduced to detecting corrupted transcripts, selecting answer letters, or training a tiny linear scorer over a few coarse signals while claiming the complete architecture has been tested. The older speech-gap controller idea is historical context, not the current primary objective.

Feasibility, better performance, and research novelty are separate questions. The architecture is implementable on the existing repository. Improvement and novelty have not been established. Failed small scorers do not establish that a genuinely learned temporal proposal and multimodal set selector cannot work.

## Proposed reference architecture (starting point; justified variants permitted)

```text
Video + timed transcript + user question
                  |
         broad transcript retrieval
                  |
        relevant temporal windows
          /                     \
Path A: text-guided              Path B: visual scan
frozen text encoder             sample across the windows
+ trainable temporal Head A     with allocation by duration
proposes useful moments         frozen MobileCLIP ranks frames
          \                     /
         pooled candidate frames
          + conservative deduplication
          + image features, timestamps,
            question and nearby verbatim speech
                       |
        trainable multimodal Head B
        selects K useful frames jointly
                       |
        frozen answer VLM initially
        selected actual images + question
        + budgeted verbatim transcript
                       |
         open-ended grounded answer
```

### Head A: learn WHERE to look

Input: question, ordered timed transcript representations and broad retrieved windows. Output: multiple proposed moments or subintervals. It should be able to learn that relevant imagery can occur before, during, or after an utterance, rather than always extracting the frame near the utterance end.

Example: a retrieved window runs from 100 to 200 seconds. A predicted 23–41% subinterval maps to 123–141 seconds. This is a representation example, not a requirement to always choose that interval length.

A workable first implementation is a trainable temporal attention head over frozen token/segment embeddings, predicting time-bin distributions or interval boundaries. It needs question conditioning, time information, local speech context and allowance for multiple moments. Head size should follow the representation and data, not an arbitrary requirement to use six scalar features. LoRA on the encoder is a later controlled option if a frozen encoder is limiting; it is not automatically necessary or novel.

Text alone cannot guarantee when a specific equation or object is visible. Head A proposes likely evidence; the visual path and final selector provide the complementary image information. There is no promise of exact visual localization from speech alone.

### Path B: provide independently useful visual candidates

Distribute a fixed scan budget across retrieved windows in proportion to their durations, with explicit rounding and minimum-coverage rules. Uniform sampling within each window is a reproducible starting point. Longer windows normally receive more samples. MobileCLIP scores these images against the question, retaining a configured candidate budget.

Keep three budgets distinct: frames decoded/scanned; candidates retained from A and B; final K frames sent to the answer VLM. K final images does not mean the pipeline only processed K images. Report full decoding, encoding, selection and generation costs.

Choose K using a bounded, declared development-set comparison of answer quality versus measured total cost and model limits. Lock the chosen configuration before final testing. Compare methods at the same K and report quality-cost curves across tested budgets; do not select K by repeatedly inspecting final test scores. An adaptive per-question frame count is an optional later experiment, not a newly imposed controller requirement. Historical four-frame results remain four-frame results.

Both paths initially use broad transcript-retrieval windows for continuity with the implementation. Missing the answer window limits both paths. Measure this retrieval bottleneck separately; do not silently add a full-video scan and attribute its cost or gain to the learned heads.

### Head B: learn WHICH frames work together

Input: question representation, candidate image features, candidate timestamps, surrounding verbatim speech, path provenance and the already-selected frame set. OCR can be an optional feature, but board text can be missed and OCR must not be treated as reliable ground truth.

A workable first implementation is a small cross-attention/set model that predicts the marginal usefulness of each remaining frame conditioned on the selected frames. It chooses K distinct frames, without enforcing an equal quota from each path. Complementary evidence means information useful for the answer; visual difference alone is not the objective.

Global MobileCLIP similarity is a baseline and useful feature, but cannot be assumed to capture tiny board symbols. A controlled spatial-feature variant may use frozen patch/region features. Its extraction latency must be included. Increasing parameter count without adding informative inputs and supervision is not sufficient.

The old rule that a TEXT-only controller must never receive image content belonged to the earlier architecture. In the latest design the selector explicitly receives image FEATURES. Gold answers and training utility labels remain forbidden inference inputs. The final answer VLM still receives the actual selected images and generates the answer.

### Transcript sent to the answer VLM

Use original transcript wording around selected moments, including configurable preceding/following context. Merge overlapping excerpts, keep timestamps, impose a stated text budget, and report truncation. Do not rewrite the excerpt into the gold answer. For selector comparisons keep the transcript policy and budget controlled; use an identical excerpt in a dedicated ablation if needed to isolate image selection gains.

## What exists and can be reused

- Video decoding, ASR/timed-transcript interfaces, retrieval and candidate construction.
- Two-path pool construction, MobileCLIP embeddings, OCR interfaces and caching.
- Answer VLM interface, experiment harness, split checks, reports and CPU tests.
- Raw videos/transcripts and unchanged cached image features, when their identities and preprocessing match.

Key current files:

- `src/videoqa/v2/candidates.py`: current Path A timestamp uses `s.end_s - 0.3`; timestamps are not learned. Both paths inherit nearest-transcript relevance scores.
- `src/videoqa/v2/head_a.py`: the small main configuration has 14 trainable parameters, over six features and two outputs; the full frozen-feature linear version has 772. The frozen text model is much larger, but its weights are not being trained.
- `src/videoqa/v2/head_b.py`: the previously audited main/default variants have about 8,333 / 15,257 trainable parameters. They are coarse-feature scorers, not a learned spatial-temporal attention system.
- `src/videoqa/v2/teacher.py`: current v2 teacher requires MCQ targets; open-ended data require an explicit answer-quality scorer.
- `src/videoqa/v2/experiment.py`: experiment integration and policy comparisons.

The previous Head A training did not replace the fixed end-of-utterance timestamp rule in main candidate generation. This is a specific mismatch with the user's intended learned temporal proposal.

Old answer-quality/action labels are not automatically reusable when frames, frame order, transcript excerpts, model revisions, decoding or answer prompts change. Cache identity must capture the actual model input and settings. Reusable media/features and reusable correctness labels are different things.

## What previous experiments actually established

These are historical MCQ experiments, not measurements of the intended detailed open-ended system.

| Experiment | Result | Interpretation |
|---|---|---|
| v1 LongVideoBench, 440 questions | transcript 32.50%; selected frames + transcript 41.59% | +9.09 percentage points, or 40 more correct answers. Not a learned-head gain. |
| MIT main, 197 synthetic questions / 20 lectures, original option order | relevance 38.58%; completion-trained head 37.56%; local-feature variant 38.07% | Learned variants did not beat relevance. |
| Fixed relevance top-three + best possible fourth frame | constrained oracle 45.18%, +6.60 points over relevance | Measured headroom only for this candidate table, original answer-option order and fixed first three frames. Not a guaranteed learnable gain. |
| v2 LongVideoBench, same 440 questions | legacy MobileCLIP 42.95%; relevance 40.68%; balanced MobileCLIP 42.50% | Scan benefit did not establish transfer; no learned-head transfer success was shown. |

The MIT questions had severe answer-position imbalance: 79 A, 92 B, 24 C, 2 D. Always choosing B would score 46.70%. Filtering by one model's ability to answer without evidence amplified the observed imbalance. The four cyclic-option rescore changed absolute accuracies but did not regenerate the dataset, retrain heads, or repair all old utility labels. These findings must retain that caveat.

On lectures the relevance selector practically selects Path A by construction: A frames come from the highest-scoring speech lines, while B frames inherit the score of the nearest line. This does not prove Path B is inherently useless. General-video results favored MobileCLIP over speech relevance. The existing selectors did not successfully combine the paths.

Other tested directions included crop/padding/tiling variants, local board features and balanced scan allocation. None established a better learned selector. The latest prior audit reported 205 passing CPU tests; tests have not been rerun merely to create this documentation.

## Data: what is available, what is not

The primary goal now requires published open-ended questions and credible reference answers, with accessible videos and timed transcripts or reproducible ASR. A published QA reference does NOT automatically provide an exact evidence timestamp or a preferred frame set.

### Lecture evaluation candidate: EduVidQA (EMNLP 2025)

- Paper reports 5,252 questions over 296 computer-science lecture videos.
- Its roughly 270 real-world questions have expert-completed/verified answers; the much larger roughly 4,982-question portion is synthetic. Do not call all training data human-authored.
- The previously inspected local release had 3,909 training, 1,056 synthetic-test and 269 real-test rows, totaling 5,234. Record version/count differences rather than hiding them.
- Last readiness check found no local videos and transcripts for 87 of 296 videos. Access, licensing and transcript quality still need verification before expensive runs.
- Real test questions remain evaluation-only. Some have question timestamps; timestamp-given and timestamp-hidden retrieval are different protocols.
- A locally inferred interval around a question timestamp is not a human-annotated evidence span.
- Sources: https://aclanthology.org/2025.emnlp-main.1760/ and https://github.com/sourjyadip/eduvidqa-emnlp25

### Additional open-ended evaluation candidates

- MINERVA-Cultural provides human-authored long-video questions/answers across cultures; it is not a lecture dataset and transcript availability must be checked. https://github.com/google-deepmind/neptune#minerva-cultural
- Neptune provides open-ended QA with semi-automatic generation and human verification, not an entirely human-authored training corpus. https://github.com/google-deepmind/neptune#neptune
- MMVU is expert annotated but predominantly short clips; audio is removed and lecture/slide material excluded. It is not the primary match for this speech-guided lecture objective.
- LongVideoBench remains a useful secondary established MCQ baseline, not the primary evidence for detailed answer quality.

Dataset choice is not settled by publication alone. A small real data-readiness pilot must show usable videos, transcripts, questions, references, distinct train/development/test videos, and an audited open-ended scorer before training scales.

## Exactly what would be trained, and on what

Initially train only temporal Head A and multimodal Head B. Keep the text encoder, MobileCLIP/scout and answer VLM frozen. This isolates selection quality from changes to the answer model. Encoder adapters or answer-model tuning are later separate experiments, not assumed prerequisites.

A training record needs:

```text
video_id, dataset_version, split
question, reference_answer, reference_provenance
timed_transcript, transcript_source
broad_windows, candidate_times, decoded_frame_ids
image_features_and_preprocessing_identity
local_verbatim_transcript, selected_context_frames
frozen_answerer_revision_and_generation_settings
generated_answer_for_each_tested_evidence_set
audited_quality_score, scorer_identity, label_confidence
measured_decode_encode_select_generate_cost
temporal_target_or_set_preference_and_how_it_was_derived
```

Gold answers and quality scores are training/evaluation targets only. The deployed heads receive question/evidence inputs, never the gold answer.

Head A needs temporal targets. Genuine evidence-span annotations can supervise those targets if the task matches. Otherwise, a limited search over candidate offsets on TRAIN videos can provide weak labels: which times helped the frozen answer VLM produce a better open-ended answer relative to the reference. Those labels are model- and context-dependent, not expert-certified frame labels.

Head B needs contextual set targets: compare adding or swapping a candidate while keeping the other frames and transcript context fixed. Prefer supervised ranking/listwise losses over reliable quality differences as the initial baseline. A first implementation should cover varied selected-frame contexts, not only one fixed top-three context. Uncertain quality ties should not become confident winner labels.

An open-ended quality scorer should assess factual correctness and reference coverage, with a human-audited sample and explicit treatment of alternative valid answers. Exact string match alone is insufficient. Reference-answer agreement also does not establish visual grounding, so retain evidence checks and text-only/visual-only ablations. Judge/model calls and human audit effort need a declared budget.

There is no requirement to manually label every frame. There IS a requirement to validate the automatically generated supervision before trusting it. The availability of expert test answers does not solve the need for adequate training examples and temporal labels.

## Research hypothesis and novelty discipline

Candidate hypothesis: learning where visual evidence occurs relative to speech, then selecting a budgeted frame set based on its usefulness alongside local speech, improves grounded open-ended answer quality at equal total inference cost compared with fixed transcript-end sampling and simple frame ranking.

This is a hypothesis, not an established novel algorithm. Related work already includes temporal localization, question-aware frame selection, set selection, synthetic utility supervision and frozen-answerer training. A particular combination or a new name is not proof of novelty. Before any novelty claim, compare the implemented mechanism to close primary papers, including the project's existing literature review; verify what those papers actually do.

No specific new optimizer or state-of-the-art post-training method has been finalized. A supervised temporal/ranking baseline is the first control. RL, DPO/GRPO-style optimization or LoRA must solve an observed limitation with reliable supervision, and must be compared against that baseline. They cannot be added solely to make the project sound novel.

## Research before upgrades: choose a few justified experiments

The user's latest requirement is a researcher-style process: inspect relevant work, SYNTHESIZE AND DESIGN NEW METHODS where justified, identify promising improvements for Head A, Head B and the wider pipeline, explain the choices in understandable language, and evaluate a small prioritized set. This is not limited to reproducing papers or swapping pretrained models. The temporal-attention and set-attention mechanisms suggested earlier in this document are candidate starting designs, not a requirement to ignore a better-supported or newly formulated method. The two-path design is the reference starting point; justified architectural variants are also within scope under the rules above.

### Original design is an explicit part of the assignment

The user explicitly wants original architecture/algorithm/training ideas to be formulated and tried, including transferring useful ideas from another field into this evidence-selection problem. Existing methods provide foundations and controls, not a ceiling on what may be designed. Candidate sources may include temporal alignment, structured prediction, information acquisition, uncertainty modeling, learning to rank, set optimization or other relevant areas; these are search directions, not claims that any one is new here.

Develop at least one concrete project-specific design proposal alongside the strongest feasible established baseline. When the proposal is technically sound and supervision/compute are feasible, implement it in the bounded shortlist rather than leaving all original ideas as future work. If no defensible proposal survives scrutiny, explain the specific obstacle instead of inventing a novelty claim or silently reducing the task to baseline reproduction.

Originality may lie in a temporal proposal parameterization, multimodal representation, conditional set-scoring mechanism, objective/loss, regularizer, training-data construction or coordinated training scheme. Describe the proposed mechanism precisely: inputs, outputs, trainable parameters, inference behavior, supervision source, mathematical objective or pseudocode, and expected reason for improvement. Identify what existing idea is borrowed, what is changed, and why that change matters for this task. An adaptation from another field can be a research candidate; being new to this repository or being an untried combination does not by itself establish research novelty.

For each new proposal, separate two questions: whether the mechanism differs meaningfully from the closest work, and whether it helps on the chosen evaluation. Include ablations that remove the proposed new component, and compare against the closest feasible method at matched resources. A performance gain alone does not establish novelty, and a novel formulation does not guarantee a gain. Record useful failures and inconclusive outcomes as carefully as successful ones.

The user is willing to consider larger trainable heads/adapters and multi-day experiments when justified; lightweight inference does not require a tiny number of learned parameters or a trivial training process. Choose capacity and training effort based on the data, mechanism and measured inference limits. Start with a feasibility pilot, then propose a larger run if evidence warrants it. A two-to-three-day willingness is not an unlimited API or cloud-spend authorization: specify actual resources, expected calls/time/cost and limits before launching.

The user clarified that the resource request is primarily about obtaining suitable free or modestly priced GPU compute for training, not requiring Gemini API to train the custom heads. Gemini is optional for teacher/judge calls if useful. Inventory actual GPU access, model/API access and budget before choosing tools; do not assume any particular entitlement is available. Teacher/judge calls produce supervision/evaluations, while custom Head A/Head B weight updates need an actual training environment. If hosted tuning is proposed, verify support for the required model, objective and data flow.

Google Colab is one possible training environment: its official FAQ confirms free GPU/TPU resources, but availability and usage limits are variable and not guaranteed. A Gemini account/subscription must not be treated as proof of unrestricted training GPU access. Prefer an already available suitable local GPU or legitimate free training runtime where feasible; consider a modest paid option when the expected research value justifies a stated cost ceiling. Multi-day research may span resumable sessions rather than one uninterrupted free session. Save checkpoints, configurations and progress so interruptions do not waste completed work. No GPU has been provisioned and no cloud/API charge authorized by preparing this document. Source checked 2026-10-05: https://research.google.com/colaboratory/faq.html

### 1. Read primary work and connect it to an observed problem

Use internet research to find relevant papers, official implementations and established benchmarks. Read the actual method/evaluation sections and, where relevant, PDF figures, appendices and code; search snippets or abstracts alone do not support a technical recommendation. Record source links, publication/preprint status, what was actually demonstrated, datasets, inference/training cost and limitations. Do not claim to have searched every paper worldwide or found the globally best method.

Target the research to concrete component questions:

- **Head A:** question-conditioned temporal localization; matching speech to visual events when they are offset; multi-moment proposals; frozen embeddings versus adapters; suitable temporal supervision and objectives.
- **Head B:** visual representations that preserve useful local detail; fusion with question and nearby speech; contextual frame-set scoring; redundancy and complementary evidence; suitable supervised or post-training objectives.
- **End-to-end evaluation:** credible open-ended video QA, reference provenance, grounding checks, reliable answer-quality scoring and measured quality/cost tradeoffs.

Reuse the existing literature review as a starting point, then verify closer or newer relevant work. For each proposed borrowing, distinguish a published technique from a project-specific adaptation and an untested hypothesis. A new model name, embedding, optimizer or combination is not automatically a novel contribution.

### 2. Rank candidates before spending compute

Produce a compact shortlist, initially around two or three targeted hypotheses in total rather than an unbounded model/algorithm sweep. This is an initial planning scale, not a permanent limit on research or a restriction against new designs. Prefer candidates that address a demonstrated bottleneck and are feasible with available training data and compute. Include a justified original-design candidate as described above, alongside appropriate established controls. The shortlist should cover the most credible Head A and Head B opportunities; a combined run is justified after separate effects are understood.

For every candidate record:

| Decision field | What the record must explain |
|---|---|
| Current problem | Evidence in the code, data or saved results that motivates this change |
| Mechanism and source | What embedding/scoring/learning method changes and which primary source supports considering it |
| Expected benefit | A specific reason it might improve selected evidence or answer quality, stated as a hypothesis |
| Training feasibility | Required inputs/targets, their provenance, availability, noise and trainable/frozen components |
| Cost | Feature extraction, label generation, training and deployed inference cost, including judge/API calls |
| Fair comparison | Baseline, controls, development/test splits, frame/text budget and unchanged components |
| Decision rule | What result would support, refute or leave the hypothesis inconclusive |
| Why this over alternatives | Why competing options are deferred: weaker evidence, unavailable labels, cost, reproducibility or poor task fit |

A head's capacity must be justified by its inputs and training evidence, not chosen solely to be extremely small or substantially larger. Advanced post-training methods are optional tools selected for a specific limitation; their current literature status and practical value need verification.

### Practical trial matrix: Head A, Head B and sampling

The user explicitly wants comparative trials across promising implementations, not one predetermined tiny head and not an indiscriminate sweep of every possible combination. The experiment runner should support the following controlled comparisons, prioritized after literature/data review:

| Trial family | Examples worth considering when justified | What the comparison should isolate |
|---|---|---|
| Established baselines | Frozen transcript relevance, visual similarity, simple fusion and uniform sampling | A reproducible reference before new training |
| Head A upgrades | Fixed start/middle/end versus learned temporal proposals; selected embedding/backbone or adapter alternatives | Value of learning useful moments, with matched retrieval scope and candidate/final budgets; candidate times may legitimately change |
| Head B upgrades | Existing scorer versus a selected established neural scorer and a defensible original formulation | Value of representation, scoring and training, initially on the same candidate pools and transcript policy |
| Sampling upgrades | A small justified set of interval allocations, uniform/temporally informed proposals, A/B candidate quotas and final frame counts | Where/how many frames are sampled and retained, with actual total cost recorded |
| Combined configuration | Promising A, B and sampling variants chosen on development data | Whether separately useful changes complement each other or interfere |
| Optional transfer | Another suitable frozen backbone or answer model, if the hypothesis and budget justify it | Whether an observed effect depends on one particular model; report this as a distinct comparison |

Learned variants need appropriate training on TRAIN videos and selection on DEVELOPMENT videos. Compare the same new neural architecture before/after training when useful to isolate learning, but random initialization is not a substitute for the strong established baseline. Check sensitivity to seed/optimization when affordable before treating a small difference as an improvement. Hold the final answerer fixed for the core selector comparison; changing it at the same time would confound attribution.

Sampling includes both candidate generation and final evidence budgets. For example, changing candidate-pool size, temporal placement, path allocation and K simultaneously can improve a score without identifying the cause. Use staged comparisons and then test a combined configuration. Explore frame counts on development data and report quality/cost tradeoffs rather than choosing the largest or best test-set score by default.

Each completed trial should explain what improved or failed and the evidence supporting that interpretation. Proposed reasons are hypotheses unless controlled comparisons or case analysis substantiate them. Record visual examples of useful/missed moments where practical, training-target quality, accuracy/answer-quality uncertainty and full inference cost. Unrun model/sampling combinations stay explicitly listed as future work.

### 3. Make experiment decisions visible before costly execution

Save a short, readable experiment plan before collecting expensive labels or training. It should state what will change, why it was chosen, expected outcome, dataset/metric, controls, compute caps and the next decision. Give the user a concise explanation of that plan so the process is understandable; normal reversible implementation need not stop for repeated confirmations when already authorized. Paid or large compute still requires a stated authorized budget.

Start with data/scorer checks and a bounded pilot. Evaluate one meaningful change at a time; record combined effects separately. Choose model variants and frame counts on development data, then lock the configuration before final testing. Reuse valid caches and previously measured baselines when the inputs/protocol match. Do not rerun every alternative merely because it exists.

If there is no credible learning signal or the primary dataset is unusable, report that specific finding before scaling. If the pilot is too small to decide, call it inconclusive rather than declaring the method successful or impossible. Do not continue changing hypotheses after inspecting the final test set.

### 4. Leave an understandable research record

Maintain one concise decision log and per-experiment report with this narrative:

**What the old system did → what limitation we observed → what we changed → why these few changes were selected → exactly what was tested → results and uncertainty → interpretation → next decision.**

Include implementation/config/data versions, relevant sources, an ordinary-language explanation and at least one concrete example when it helps. Distinguish expected behavior, actual completed runs, observed results and speculative explanations. Keep failed and inconclusive results visible. Use small tables or diagrams where they clarify the mechanism; do not substitute a polished story for evidence.

List untested alternatives in a separate future-work section, with reasons for deferring them and what evidence would make them worth trying later. They must not appear in the results as completed experiments. The desired outcome is a credible, interpretable study and a maintainable system, not a promise of a world-best score or publication.

## Concrete next sequence

1. Preserve existing reports and reproduce one cheap baseline. Write an explicit old-implementation versus new-spec map. Read targeted primary literature, save the justified small shortlist and the experiment plan described above before expensive implementation/training decisions.
2. Complete dataset/video/transcript readiness and demonstrate a few authentic open-ended rows. Freeze video-level development/test partitions and scorer policy.
3. Add the free-text answer-quality interface with a small human audit. Validate targets before creating large supervision tables.
4. Add duration-proportional Path B allocation and explicit scan/candidate/final budgets as a controlled baseline.
5. Implement a temporal proposal Head A with real trainable time outputs and an actual connection to candidate extraction. Compare against start/middle/end/uniform and frozen relevance under matched budgets.
6. Implement a context-conditioned multimodal Head B. Compare with relevance, MobileCLIP and simple fusion under identical pools, transcript policy, final K-frame budget and answerer. Select K on development data as described above.
7. Run separate A-only-improved, B-only-improved and combined experiments. Report end-to-end latency, memory, quality, confidence intervals grouped by video, and failure cases.
8. Scale only after informative inputs/labels and measurable pilot headroom are established. Keep an untouched evaluation set; repeated development on a benchmark is not untouched testing.

Large GPU/API runs need an explicit total call and time/cost budget before execution; exceeding either requires a revised authorization. Earlier work reported 6,433 calls against a declared 3,000-call limit. Do not repeat this budget mismatch. No new compute budget is provided by this handoff.

## Implementation organization, debugging and commits

The user wants an understandable, maintainable implementation with small coherent commits. This is a software-module design requirement, not a request for a UI design system.

- **First establish the current state:** read this handoff, the linked full historical review and the repository's architecture and relevant source/tests. Inspect Git status and existing history. Preserve completed work before refactoring. If the current baseline is already committed, identify that commit rather than manufacturing an empty duplicate; commit intended outstanding baseline/documentation changes separately after reviewing them. Do not bulk-add caches, downloaded datasets, generated large artifacts, credentials or unrelated user work.
- **Keep responsibilities separate:** dataset/ASR adapters; shared typed records; window retrieval; temporal Head A; Path B scan; candidate merge/dedup; multimodal Head B; transcript packing; answerer; quality evaluator; training/label generation; experiment runner. Reuse existing modules and interfaces where appropriate rather than replacing the whole repository or adding abstraction without a concrete need.
- **Document interfaces:** each important module/function should explain its purpose, inputs/outputs, tensor shapes and units, frozen versus trainable state, and meaningful failure cases. Comments should explain non-obvious decisions. A short module map and one end-to-end example should make data flow inspectable.
- **Make decisions traceable:** optional per-question debug output should record retrieved windows, proposed timestamps, candidate provenance and scores, dedup decisions, selected frame IDs, exact transcript excerpts, model/config versions, and per-stage latency/cache use. Ground-truth answers and utility labels must remain outside inference inputs. Local debugging must not silently trigger external telemetry or extra model calls.
- **Use stage-focused commits:** baseline/documentation preservation; data and evaluation interfaces; temporal proposals; scan/pool integration; multimodal selection; training; controlled evaluation/reporting. Adjust boundaries to the actual diff so every commit has one coherent purpose and useful validation. Commit messages and reports must distinguish implemented, CPU-tested, GPU-tested and unverified work.
- **Verify meaningful behavior:** tests should cover time conversion/bounds, duration allocation and budget caps, split separation, inference-target leakage, cache identities and selector/evaluator integration. A small reproducible smoke run should show the complete pipeline. Do not claim that comments or passing unit tests guarantee bug-free behavior or model quality.

Suggested implementation-start request for Claude Code (the flexible research clarification above governs all method examples):

> Read this handoff and its linked historical review, then inspect the codebase and current Git state. Preserve the working baseline. Research relevant primary papers and implementations for Head A's temporal proposals and Head B's multimodal frame-set selection. Use that research to formulate and test new project-specific designs as well as established methods; do not limit the work to copying published methods or swapping models. Consider transferable ideas from other fields when their mechanism fits. Compare candidate embeddings, scoring methods, representations and training objectives, and explain why a small prioritized shortlist, including a defensible original-design candidate, is worth testing and why other ideas are deferred. Specify what is new, what is borrowed, how it learns, and which ablation tests its contribution. Start from the two-path reference architecture, but test justified framework variants when they address a credible limitation; preserve and compare against the original. Treat frame counts, sampling allocation and transcript coverage as configurable development-selected choices. Compare strong baselines, justified trained Head A/Head B alternatives, sampling variants and then promising combinations; keep other components controlled so the source of any gain can be understood. Validate data and open-ended scoring, publish a readable experiment plan, then implement and run bounded, controlled pilots within the authorized compute budget. Larger heads/adapters and multi-day runs are possible when justified by data and pilot evidence. Prefer available suitable local or free GPU resources, consider modest paid compute within an explicit cost ceiling, and make runs resumable. Gemini is an optional teacher/judge, not a substitute for the custom-head training runtime. Organize responsibilities into clear modules with documented interfaces and useful debug traces. Make small coherent commits with appropriate checks; use a separate branch/worktree when helpful, otherwise coherent commits on the existing branch are acceptable. Record what was changed, why, what was actually tested, results and uncertainty, and future work. Do not claim novelty or superior performance before the evidence supports it, launch unbudgeted large runs, or silently replace the research objective.

## Documents worth reading in the repository

- `docs/v2/README.md`
- `docs/v2/main_report.md`
- `docs/v2/completion_report.md`
- `docs/v2/option_bias_report.md`
- `docs/v2/lvb_transfer.md`
- `docs/v2/benchmark_readiness.md`
- `docs/v2/novelty_check.md`
- `docs/v2/review_20261004.md`
- `docs/report/framework_report.html` and its PDF counterpart (historical results, not this latest proposal)

Additional audit artifacts are in the original Codex workspace at `C:\Users\USERAS\Documents\Codex\2026-09-26\so-search-all-of-the-papers\outputs`, especially `Framework 2 final evidence review.md`. These paths are local evidence references; the code must not depend on that external workspace.

## Continuing across accounts or with Claude

The repository and this handoff can be opened by another authorized local assistant. That does not recreate the original conversation sidebar or automatically transfer memory. This file captures decisions and evidence; it cannot guarantee every historical utterance is preserved.

For simultaneous Claude/Codex work, use separate branches/worktrees or explicitly disjoint files, with one integrating/reviewing changes. There is no verified automatic shared-memory or direct Claude messaging connection in this session. Both can use this same written specification; neither should silently change the architecture.

Account switching does not merge accounts or their histories. Official guidance checked on 2026-10-05:

- https://help.openai.com/en/articles/20001068-use-multiple-accounts-with-account-switching
- https://help.openai.com/en/articles/9106926-transfer-exported-conversations-between-chatgpt-accounts

The second page describes ChatGPT export-as-reference, not a verified full Codex history migration. This handoff is deliberately specific to the video-QA project and must not be represented as a complete account or multi-chat archive.
