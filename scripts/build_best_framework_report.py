"""Build the summary report "best working framework + the whole research journey" (HTML -> PDF).

The final-test table is read from reports/v3_final/test_results.json; every other number is transcribed from
the committed reports and docs it cites (docs/final_report_controller_study.md, reports/benchmark_v1*,
docs/v2/*.md, docs/v3/*.md, docs/v3/research_log.md steps 1-53).

    python scripts/build_best_framework_report.py --pdf    # docs/report/best_framework_report.{html,pdf}
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

OUT = Path("docs/report")

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
:root { --ink:#1d2430; --muted:#5b6575; --line:#d5dbe3; --accent:#0f6b6b; --head:#8a4b0f; --headbg:#fbf1e4;
        --box:#f3f6f9; --good:#1f7a3a; --bad:#a3322a; --soft:#eef5f5; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); font-size: 10.2pt;
       line-height: 1.45; margin: 0; background: #fff; }
h1 { font-family: Georgia, "Times New Roman", serif; font-size: 22pt; line-height: 1.15; margin: 0 0 4px;
     text-wrap: balance; }
h2 { font-family: Georgia, "Times New Roman", serif; font-size: 15pt; margin: 22px 0 6px; color: var(--accent);
     border-bottom: 1.5px solid var(--accent); padding-bottom: 3px; break-after: avoid; }
h3 { font-size: 11.2pt; margin: 14px 0 4px; break-after: avoid; }
p { margin: 5px 0 7px; }
.sub { color: var(--muted); font-size: 10pt; margin-bottom: 10px; }
.flow { display: flex; flex-direction: column; align-items: stretch; gap: 0; margin: 8px 0 4px; }
.step { border: 1.3px solid var(--line); background: var(--box); border-radius: 6px; padding: 6px 10px; }
.step b.t { display: block; font-size: 10.4pt; }
.step .d { color: var(--muted); font-size: 9pt; }
.step.head { background: var(--headbg); border: 2px solid var(--head); }
.step.head b.t { color: var(--head); }
.step.io { background: var(--soft); border-color: var(--accent); }
.tag { display: inline-block; font-size: 7.8pt; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
       padding: 1px 6px; border-radius: 3px; margin-left: 6px; vertical-align: 1px; }
.tag.h { background: var(--head); color: #fff; }
.tag.r { background: var(--accent); color: #fff; }
.arrow { text-align: center; color: var(--accent); font-size: 13pt; line-height: 1; margin: 1px 0; }
.arrow small { font-size: 7.8pt; color: var(--muted); }
.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.callout { border-left: 4px solid var(--accent); background: var(--soft); padding: 7px 10px; margin: 8px 0;
           break-inside: avoid; }
.callout.warn { border-color: var(--head); background: var(--headbg); }
table { border-collapse: collapse; width: 100%; margin: 6px 0 9px; font-size: 8.9pt; break-inside: avoid;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #e9eef3; font-weight: 600; }
td.n, th.n { text-align: right; }
tr.best td { font-weight: 700; background: #eef7ef; }
.good { color: var(--good); font-weight: 600; } .bad { color: var(--bad); font-weight: 600; }
.journey { border-left: 3px solid var(--accent); padding-left: 12px; margin-left: 4px; }
.stage { break-inside: avoid-page; margin-bottom: 4px; }
.stage .when { font-size: 8.4pt; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.miniflow { font-size: 8.8pt; background: var(--box); border-radius: 5px; padding: 4px 8px; margin: 4px 0;
            border: 1px dashed var(--line); }
.data { font-size: 9pt; color: #3a4350; background: #fafbfc; border: 1px solid var(--line); border-radius: 5px;
        padding: 5px 8px; margin: 5px 0 8px; break-inside: avoid; }
.data b { color: var(--ink); }
.pagebreak { break-before: page; }
.small { font-size: 8.6pt; color: var(--muted); }
"""


def table(head, rows, best=None, num_from=1):
    h = "".join(f"<th{' class=n' if i >= num_from else ''}>{c}</th>" for i, c in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        cls = " class=best" if best is not None and k == best else ""
        body.append(f"<tr{cls}>" + "".join(f"<td{' class=n' if i >= num_from else ''}>{c}</td>"
                                           for i, c in enumerate(r)) + "</tr>")
    return f"<table><tr>{h}</tr>{''.join(body)}</table>"


def arrow(label=""):
    return f'<div class="arrow">&#9660; <small>{label}</small></div>'


def step(title, desc, kind="", tag=""):
    t = f'<span class="tag {"h" if kind == "head" else "r"}">{tag}</span>' if tag else ""
    return f'<div class="step {kind}"><b class="t">{title}{t}</b><span class="d">{desc}</span></div>'


def final_tables():
    res = json.loads(Path("reports/v3_final/test_results.json").read_text(encoding="utf-8"))
    sets = ("cgbench_test", "videommmu_test", "videommmu_comp_test")
    names = {"R1": "No frames (question only)", "R2": "4 evenly spaced frames (standard rule)",
             "R3": "Dense MobileCLIP top-4 (standard rule)", "S0": "v2 framework (starting point)",
             "S1": "Tuned framework (frozen baseline)", "S2": "Tuned + Head A", "S3": "Tuned + Head B (learned)",
             "S4": "Tuned + Head A + Head B", "S5": "Tuned + answer-confidence selection"}
    rows, best = [], None
    for k, (sid, s) in enumerate(res["systems"].items()):
        rows.append([names[sid]] + [f"{100 * s[d]['accuracy']:.1f}" for d in sets] + [f"{100 * s['pooled']['accuracy']:.1f}"])
        if sid == "S1":
            best = k
    t1 = table(["System (same 2B answer model, 4 frames)", "CG-Bench (55)", "VMMMU Perception (202)",
                "VMMMU Comprehension (202)", "All 459"], rows, best)
    lab = {("S1", "S0"): "Tuned − v2 (main question)", ("S2", "S1"): "Head A − tuned", ("S3", "S1"): "Head B − tuned",
           ("S4", "S1"): "Head A+B − tuned", ("S5", "S1"): "Confidence selection − tuned",
           ("S1", "R3"): "Tuned − dense MobileCLIP", ("S1", "R2"): "Tuned − evenly spaced"}

    def cell(e):
        sig = e["ci"][0] > 0 or e["ci"][1] < 0
        txt = f"{100 * e['diff']:+.1f} ({100 * e['ci'][0]:+.1f} to {100 * e['ci'][1]:+.1f})"
        cls = "good" if sig and e["diff"] > 0 else ("bad" if sig else "")
        return f'<span class="{cls}">{txt}</span>' if cls else txt
    rows2 = [[lab[(c["a"], c["b"])]] + [cell(c[d]) for d in sets] + [cell(c["pooled"])] for c in res["comparisons"]]
    t2 = table(["Difference in points (95% interval)", "CG-Bench", "Perception", "Comprehension", "All 459"], rows2)
    return t1, t2


def build() -> str:
    t_final, t_diff = final_tables()
    parts = []
    # ------------------------------------------------------------------ page 1: best framework
    parts.append("""
<h1>Finding the Right Moments: a Local Framework for Question Answering on Long Videos</h1>
<div class="sub">Sayed Ilham Azhar Harun &middot; project <i>local-surf</i>, branch <i>framework-2</i> &middot; October 2026
&middot; everything runs on one 16 GB consumer GPU with small, frozen open models</div>
<p><b>The problem.</b> A person asks a question about a long video (a lecture, a vlog, a film; 5 to 60 minutes).
A small vision-language model can only look at a few pictures. The framework must find the few moments
that contain the answer, using the video's speech and its pictures, and give the model the right text.
Below is the best-working framework found by the experiments in Part 2; arrows show the order of work.</p>
<h2>Part 1 &middot; The best-working framework</h2>
<div class="flow">""")
    parts.append(step("Input: the question (with its answer options), the video, and its timed transcript",
                      "Transcript from the video's subtitles, or from our Whisper speech recognition when there are none.", "io"))
    parts.append(arrow("find where the speech talks about the question"))
    parts.append(step("1 &middot; Transcript search (BM25)",
                      "The transcript is cut into 20-second pieces; word matching with the question and options picks the 4 best pieces (+ neighbours)."))
    parts.append(arrow("two paths propose candidate pictures in parallel"))
    parts.append('<div class="row2">')
    parts.append(step("2a &middot; Path A: speech path &mdash; HEAD A, the speech-to-moment localiser",
                      "Scores <b>every</b> transcript line (not only the BM25 pieces) with two frozen text models "
                      "(MiniLM, BGE) plus a small learned layer that reads the context of nearby lines and the type "
                      "of question; several relevant lines then <i>vote</i> for a moment. Finds the evidence moment "
                      "far better than word matching: <b>36.6% vs 28.0%</b> of questions hit by its 6 moments "
                      "(confirmed, +8.5 points). <i>In the best-accuracy setting the simpler zero-shot speech rule "
                      "(MiniLM inside the BM25 pieces) sits here; Head A is the drop-in that localises better.</i>",
                      "head", "Head A"))
    parts.append(step("2b &middot; Path B: picture path (hybrid visual search)",
                      "Half of its frames come from a scan inside the BM25 pieces, half from a dense scan of the "
                      "<b>whole video</b> (a picture every 2 s, ranked by MobileCLIP similarity to the question). "
                      "This is the change that matters most on long videos: the right picture is now among the "
                      "candidates for <b>60%</b> of questions instead of 33%."))
    parts.append("</div>")
    parts.append(arrow("merge, remove near-duplicates"))
    parts.append(step("3 &middot; Candidate pool: about 10&ndash;12 frames",
                      "Each frame keeps its time, which path found it, and its picture embedding."))
    parts.append(arrow("choose the final 4 frames as a set"))
    parts.append(step("4 &middot; HEAD B, the frame-set selector",
                      "Picks the 4 frames that best match the question <i>and its answer options</i>, while skipping "
                      "near-identical frames (MMR rule). Learned versions were also built and tested: an evidence "
                      "scorer, a scorer trained on the answer model's own feedback, an option-evidence set rule, and "
                      "<b>answer-confidence selection</b> (the answer model answers with 4 candidate sets and the most "
                      "confident answer is kept; best of the learned variants, +0.9 points on test, not confirmed).",
                      "head", "Head B"))
    parts.append(arrow("give the model the speech that matters"))
    parts.append(step("5 &middot; Speech for the answer model (up to 600 words)",
                      "The exact speech around the 4 chosen frames, plus the best transcript lines found by "
                      "combining word matching with meaning-based matching. Keeps <b>68%</b> of the speech near the "
                      "evidence; the v2 excerpt kept only 28%."))
    parts.append(arrow(""))
    parts.append(step("Output: the frozen answer model (Qwen3-VL-2B or 4B) reads the 4 frames + the speech and answers",
                      "The answer model is never trained; only the steps above decide what it sees.", "io"))
    parts.append("</div>")
    parts.append("""
<div class="callout"><b>Headline results (locked test, 459 questions never used before).</b>
With the 4B answer model the full framework scores <b>55.6%</b> versus <b>50.3%</b> for the starting v2 framework
(<span class="good">+5.2 points, confirmed</span>), and keeping the right speech adds
<span class="good">+2.8, confirmed</span>. With the 2B model, on long videos (CG-Bench) the tuned framework scores
<b>50.9%</b> versus 34.5% (<span class="good">+16.4, confirmed</span>), and overall it beats the two standard
frame-selection rules (<span class="good">+4.6 and +5.0, confirmed</span>).</div>
<div class="callout warn"><b>What the heads add, honestly.</b> Head A clearly improves <i>finding</i> the evidence
moment and puts the right picture into the pool more often (+4.9 points, confirmed), but the small answer model did
not turn this into more correct answers; the frozen test kept the simpler speech rule in the best-accuracy setting.
Head B's learned versions found evidence frames more often on long-video test questions (51&ndash;56% vs 47%) but did
not raise accuracy; the tuned MMR rule remains the selector of the best configuration.</div>
""")
    # ------------------------------------------------------------------ Part 2: journey
    parts.append('<h2 class="pagebreak">Part 2 &middot; The research journey, step by step</h2>')
    parts.append("<p>Each step below says what was built, how it was tested, the result, and what was decided next. "
                 "Percentages are the share of questions answered correctly unless stated otherwise. "
                 "&ldquo;Confirmed&rdquo; means the 95% interval of the difference does not include zero.</p>")
    parts.append('<div class="journey">')

    parts.append("""<div class="stage"><div class="when">Step 1 &middot; Version 1 (branch QAframework), September 2026</div>
<h3>Version 1: a cost-aware controller that decides when to look at the video</h3>
<div class="miniflow">transcript &rarr; BM25 windows &rarr; candidate moments &rarr; decode frames on demand &rarr; MobileCLIP
&ldquo;scout&rdquo; &rarr; controller (look / expand / stop, gain minus cost) &rarr; frozen Qwen3-VL-2B answers</div>
<p><b>Research idea tested:</b> train the controller to look at the video exactly when important speech is damaged
or missing. <b>Result: not supported.</b> On LongVideoBench, damaging the important speech changed the answer in
only 13% (train) / 4% (dev) of cases, so there was almost nothing to learn; no trained variant looked more often
under targeted damage than under control damage.</p>
<p><b>Benchmark of the working v1 system</b> (440 LongVideoBench questions, 252 videos, 2B model, up to 8 frames):</p>""")
    parts.append(table(["Frame strategy", "Multiple choice", "Open-ended (local judge)"],
                       [["Transcript only (no frames)", "32.5%", "44.1%"],
                        ["Evenly spaced frames + transcript", "39.3%", "33.4%"],
                        ["Transcript-retrieved moments", "36.4%", "32.0%"],
                        ["MobileCLIP top frames", "41.6%", "32.3%"],
                        ["Cost-aware controller", "37.5%", "29.8%"]], best=3))
    parts.append("""<p><b>Decision:</b> the visual similarity rule was the best v1 strategy (+9.1 over transcript only,
confirmed). The controller idea was closed; work moved to a new design with two proposal paths (Version 2).</p>
<div class="data"><b>Data &mdash; LongVideoBench (public benchmark, NeurIPS 2024).</b> 3,763 videos (8 s to 60 min) with
6,678 human-written multiple-choice questions and aligned subtitles. We used 440 questions of its public validation
set. <b>Limits:</b> a subset, not the full benchmark; it was also used while developing v1, so it is not an untouched
test. The open-ended column uses a local judge model (Phi-4-mini) whose agreement with humans was never measured,
and LongVideoBench is a multiple-choice benchmark, so its open-ended numbers compare strategies only.</div></div>""")

    parts.append("""<div class="stage"><div class="when">Step 2 &middot; Version 2 (branch framework-2), early October 2026</div>
<h3>Version 2: two proposal paths, Head A (speech credit) and Head B (frame-set selector)</h3>
<div class="miniflow">transcript &rarr; BM25 windows &rarr; <b>Path A</b> (speech lines scored by MiniLM, frame at line end) +
<b>Path B</b> (visual scan inside the windows) &rarr; pool (~11 frames) &rarr; <b>Head A / Head B</b> learned selectors
&rarr; 4 frames + 120-word BM25 transcript excerpt &rarr; frozen 2B answer model</div>
<p><b>Main experiment</b> (197 lecture questions, 20 MIT lectures, each lecture tested by heads that never saw it):</p>""")
    parts.append(table(["Selector (4 frames)", "Accuracy"],
                       [["Transcript only", "15.2%"], ["MobileCLIP top-4", "34.5%"], ["MobileCLIP + diversity", "36.0%"],
                        ["Head B, learned (with screen text)", "34.7%"], ["Head A, learned frame credit", "30.5% (significantly worse)"],
                        ["Zero-shot transcript relevance (no training)", "38.6%"]], best=5))
    parts.append("""<p><b>Follow-ups in v2:</b> a pilot (132 questions) and a &ldquo;fourth frame&rdquo; completion study
(&minus;1.0, refuted) found no learned gain; spreading the visual scan evenly across windows gave +4.6 (not confirmed)
and did not transfer to LongVideoBench (43.0% vs 42.5%); option-rotation scoring was added after an answer-position
bias was found. <b>Decision:</b> the learned heads did not beat simple rules on this data; v3 had to start from
credible public benchmarks where pictures provably matter.</p>
<div class="data"><b>Data &mdash; self-made MIT lecture questions.</b> Multiple-choice questions were generated from 20 MIT
OpenCourseWare lectures (Gilbert Strang) by a larger model (Qwen3-VL-4B), checked automatically, and questions the
2B model could answer without any evidence were dropped (349 verified &rarr; 197 kept). <b>Possible flaws:</b> the
questions are synthetic and machine-checked, not human-written; the generator put the correct answer at A or B in
171 of 197 kept questions, and the dropping rule interacted with the answer model's letter preference, which pushed
&ldquo;transcript only&rdquo; below chance (15.2%); we corrected this with option-rotation scoring, but the set remains
development data, not an independent benchmark.</div></div>""")

    parts.append("""<div class="stage"><div class="when">Step 3 &middot; v3 gates: which benchmarks can show an effect?</div>
<h3>Checking three benchmarks before building anything</h3>
<p>A frame-selection method can only be judged where pictures change the answer, so each benchmark was first checked
with simple rules (same 2B model):</p>""")
    parts.append(table(["Benchmark", "No frames", "Evenly spaced frames", "Well-chosen frames", "Verdict"],
                       [["EduVidQA (lectures, open-ended)", "FactQA 0.587", "0.493 (worse)", "0.498 (worse)", "pictures do not help; not used for frame claims"],
                        ["Video-MMMU Perception (lectures)", "40.2%", "52.6%", "56.7% (+16.5, confirmed)", "passes"],
                        ["CG-Bench (long videos)", "15.9%", "17.1%", "40.2% (+24.4); human-marked frames 63.4%", "passes, large headroom"]],
                       num_from=9))
    parts.append("""<div class="data"><b>Data &mdash; EduVidQA (EMNLP 2025).</b> 296 lecture videos with 5,234 open-ended questions,
scored by FactQA (an AI judge counts supported facts; we used the paper's prompt word for word, with a free NVIDIA-hosted
model). <b>Limits found:</b> the transcript alone already supports about half of the reference answers' facts, frames
made the small model's answers shorter and worse, and 10 of 87 caption tracks labelled English were actually Tamil
(filtered out). The 269 expert-written real questions were kept locked and never used.</div>
<div class="data"><b>Data &mdash; Video-MMMU (2025).</b> 300 lecture videos from university subjects, 3 questions per video
(Perception, Comprehension, Adaptation), mostly 10-option multiple choice. Perception questions often require reading
text on the screen. We used Perception and Comprehension, with our own Whisper transcripts. <b>Limits:</b> the
benchmark has no official training split, so we split its videos once by a fixed rule (one third dev, two thirds test).</div>
<div class="data"><b>Data &mdash; CG-Bench (2024).</b> 1,219 long videos (10&ndash;80 minutes), 12,129 questions; for every
question, human annotators marked the seconds where the answer is visible (&ldquo;clue intervals&rdquo;), which lets
us measure directly whether a method found the right moment. We used the 51 English-subtitled videos available to us
(510 questions), split by video into train 37 / dev 8 / test 6. <b>Limits:</b> the test part is small (55 questions on
6 videos, so its intervals are wide), the split is our own (the benchmark has none), and three open-ended reference
answers are broken in the dataset (counted as wrong for every method).</div></div>""")

    parts.append("""<div class="stage"><div class="when">Step 4 &middot; v3 baseline cycle (dev only)</div>
<h3>Tuning the rule-based framework before training anything</h3>
<p>About 40 settings were tried one at a time on dev (CG-Bench 82 questions, Video-MMMU 98): where Path B looks,
which frame rule picks the final frames, how many frames, how much transcript. The most important findings:</p>""")
    parts.append(table(["Setting (2B model, 4 frames)", "CG-Bench", "VMMMU Perception", "Evidence frame in pool (CG-Bench)"],
                       [["v2 default: Path B only inside BM25 windows, 120-word excerpt", "28.0%", "56.1%", "33%"],
                        ["Path B scans the whole video evenly", "34.1%", "57.1%", "50%"],
                        ["Path B dense whole-video scan", "37.8%", "52.0%", "62%"],
                        ["Hybrid Path B (half windows, half dense), MMR on question + options", "35.4%", "58.2%", "60%"],
                        ["&hellip; and no transcript excerpt (frozen baseline)", "40.2%", "63.3%", "60%"]], best=4))
    parts.append("""<p><b>Decision:</b> frozen baseline = hybrid Path B, MMR on question + options, 4 frames, no excerpt
(+12.2 points over v2 on CG-Bench dev, confirmed). The BM25 excerpt was dropped because it distracted the small model;
Step 9 later replaced it with a better one.</p></div>""")

    parts.append("""<div class="stage"><div class="when">Step 5 &middot; Head A cycle</div>
<h3>Head A: learning where the evidence is from the speech</h3>
<p>Training data: 877 CG-Bench questions with human-marked evidence times (subtitles only, 83 videos) plus 546 EduVidQA
questions with their time reference removed. Metric: share of dev questions (164) where at least one of 6 proposed
moments lands inside the human evidence.</p>""")
    parts.append(table(["Head A method", "Hit with 6 moments", "Note"],
                       [["Zero-shot MiniLM (the v2 rule)", "28.0%", "starting point"],
                        ["Fine-tuned MiniLM", "~8%", "training destroyed its ranking; dropped"],
                        ["Larger zero-shot rerankers (BGE up to 568M)", "up to 31.1%", "size barely helps"],
                        ["Light learned layer on frozen scores", "32.7%", ""],
                        ["&hellip; with learned speech-to-picture offsets (original idea)", "31.7%", "offsets did not help"],
                        ["&hellip; with several lines voting + BGE scores (chosen)", "36.6%", "+8.5, confirmed"]], best=5, num_from=1))
    parts.append("""<p>Inside the full framework Head A put the right picture into the pool more often (64.6% vs 59.8%, confirmed),
but accuracy did not rise (39.0% vs 40.2% on CG-Bench dev) because the frame selector did not pick its frames, and it
transferred poorly to lectures.</p></div>""")

    parts.append("""<div class="stage"><div class="when">Step 6 &middot; Head B cycle</div>
<h3>Head B: choosing the final frames as a set</h3>
<p>Training data: pools of the 373 CG-Bench train questions; for the answer-feedback version, every candidate frame
was shown alone to the answer model (4,203 answers) to measure how much it helps. Pooled over 278 dev questions
(CG-Bench + both Video-MMMU tracks), difference to the tuned rule:</p>""")
    parts.append(table(["Head B method", "Difference to the tuned rule"],
                       [["Option-evidence set rule (no training, original idea)", "&minus;1.4"],
                        ["Learned evidence scorer", "&minus;1.4"], ["Learned evidence scorer + option term", "&minus;0.4"],
                        ["Answer-feedback scorer", "&minus;3.6"], ["Answer-confidence selection over 4 sets", "+1.1 (not confirmed)"]],
                       best=4, num_from=1))
    parts.append("""<p><b>Decision:</b> no learned selector beat the simple rule clearly; training data existed for only one
benchmark and the small answer model caps what better frames can achieve.</p></div>""")

    parts.append(f"""<div class="stage"><div class="when">Step 7 &middot; Final frozen comparison (locked test, run once)</div>
<h3>The declared test: 9 systems, 459 never-used questions</h3>
<p>The systems, test sets and statistics were written down and published before any test question was answered.</p>
{t_final}
{t_diff}
<p class="small">Green = confirmed gain, red = confirmed loss. CG-Bench test has only 6 videos, so its intervals are
unreliable.</p></div>""")

    parts.append("""<div class="stage"><div class="when">Step 8 &middot; Not losing the speech</div>
<h3>How much of the important speech reaches the answer model?</h3>
<p>Measured on CG-Bench dev: does the text given to the model contain the speech within 10 seconds of the
human-marked evidence?</p>""")
    parts.append(table(["Text given to the answer model", "Important speech kept", "Open-ended correct (dev)"],
                       [["v2 BM25 excerpt, 120 words", "28%", "17.1%"], ["No text", "0%", "19.5%"],
                        ["Hybrid search (word + meaning), 300 words", "62%", "20.7%"],
                        ["Full transcript, capped at 1,500 words", "66%", "19.2%"],
                        ["Speech around the chosen frames + hybrid, 600 words (chosen)", "68%", "22.0%"]], best=4))
    parts.append("""<p>The chosen policy beats v2 on open-ended answers by +8.5 points (confirmed) and costs no
multiple-choice accuracy. Open-ended answers were graded by a free NVIDIA-hosted model with our one-line
correct/incorrect prompt; on 40 randomly chosen answers it agreed with hand grading 40 times.</p></div>""")

    parts.append("""<div class="stage"><div class="when">Step 9 &middot; A stronger answer model</div>
<h3>Does the framework still help with Qwen3-VL-4B?</h3>""")
    parts.append(table(["Answer model (all 459 test questions)", "v2 framework", "Tuned, no text", "Tuned + kept speech"],
                       [["Qwen3-VL-2B", "38.1%", "41.4%", "41.4%"], ["Qwen3-VL-4B", "50.3%", "52.7%", "55.6%"]], best=1))
    parts.append("""<p>With 4B, the full framework beats v2 by <span class="good">+5.2 (confirmed)</span> and the kept speech adds
<span class="good">+2.8 (confirmed)</span>: a stronger reader uses the speech the framework keeps. These two results
were run after the declared test and are reported as follow-ups.</p></div>""")
    parts.append("</div>")

    parts.append("""<h2>Part 3 &middot; What can and cannot be claimed</h2>
<p><b>Supported by the locked test.</b> On long videos, searching only where the transcript matches the question
misses most of the visual evidence; adding a dense whole-video visual search, matching frames to the question and its
answer options, and giving the model the speech around the chosen frames improves a local 2B/4B system substantially
over the starting design and over standard frame-selection rules. Keeping the right speech helps a stronger reader.</p>
<p><b>Not supported.</b> That the learned heads raise final accuracy: Head A finds the evidence moment significantly
better and Head B's learned versions pick evidence frames more often on long videos, but neither turned this into more
correct answers with these small answer models. These are reported as negative results, together with the original
offset idea that did not help.</p>
<p><b>Why the absolute numbers are modest.</b> The answer models are deliberately small (2B and 4B, frozen, one 16 GB
GPU). Even when given the frames human annotators marked as the answer, the 2B model reaches only 63% on CG-Bench dev;
the framework's gains are measured against other ways of choosing frames for the same model.</p>
<p class="small">Sources: docs/final_report_controller_study.md, reports/benchmark_v1, docs/v2, docs/v3/research_log.md
(steps 1&ndash;53), docs/v3/final_test_protocol.md, docs/v3/final_test_results.md, reports/v3_final.</p>""")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    html_path = OUT / "best_framework_report.html"
    html_path.write_text("<!doctype html><html lang=en><head><meta charset=utf-8>"
                         "<title>Best Framework Report</title>"
                         f"<style>{CSS}</style></head><body>{build()}</body></html>", encoding="utf-8")
    print(f"wrote {html_path}")
    if a.pdf:
        pdf = (OUT / "best_framework_report.pdf").resolve()
        for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
            if Path(exe).exists():
                subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                                f"--print-to-pdf={pdf}", html_path.resolve().as_uri()], check=True, timeout=120)
                print(f"wrote {pdf}")
                break


if __name__ == "__main__":
    main()
