"""Build "Experiments from start to end": for every experiment, the framework structure (flow diagram) and then
its full performance table (HTML -> PDF with headless Edge).

The final-test table (reports/v3_final/test_results.json) and the combination grid (runs/v3_baseline/rows.jsonl)
are read from the result files; cells not yet run are marked "running". Other numbers are transcribed from the
committed reports they cite (docs/final_report_controller_study.md, reports/benchmark_v1*, docs/v2/*.md,
docs/v3/*.md, docs/v3/research_log.md).

    python scripts/build_experiments_report.py --pdf     # docs/report/experiments_report.{html,pdf}
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

OUT = Path("docs/report")
ROWS = Path("runs/v3_baseline/rows.jsonl")

CSS = """
@page { size: A4; margin: 14mm 13mm 14mm 13mm; }
:root { --ink:#1c2330; --muted:#5a6474; --line:#d4dae2; --accent:#155e75; --soft:#ecf4f6; --head:#9a4d10;
        --headbg:#fcf1e3; --box:#f4f6f9; --good:#1d7a3a; --bad:#a3312a; --run:#7b6f2a; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; color: var(--ink); font-size: 9.8pt; line-height: 1.42;
       margin: 0; background: #fff; }
h1 { font-family: Georgia, serif; font-size: 21pt; margin: 0 0 4px; line-height: 1.15; }
h2 { font-family: Georgia, serif; font-size: 14pt; color: var(--accent); margin: 18px 0 6px;
     border-bottom: 1.5px solid var(--accent); padding-bottom: 2px; break-after: avoid; }
h3 { font-size: 10.6pt; margin: 10px 0 4px; break-after: avoid; }
p { margin: 4px 0 6px; }
.sub { color: var(--muted); margin-bottom: 8px; }
.flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 4px 0; margin: 6px 0 8px; break-inside: avoid; }
.fb { border: 1.3px solid var(--line); background: var(--box); border-radius: 5px; padding: 4px 7px;
      font-size: 8.5pt; max-width: 150px; display: flex; flex-direction: column; justify-content: center; }
.fb b { font-size: 8.8pt; }
.fb.h { background: var(--headbg); border: 2px solid var(--head); }
.fb.h b { color: var(--head); }
.fb.io { background: var(--soft); border-color: var(--accent); }
.ar { color: var(--accent); font-size: 13pt; padding: 0 4px; display: flex; align-items: center; }
.split { display: flex; flex-direction: column; gap: 4px; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 8px; font-size: 8.5pt; break-inside: avoid;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 2.5px 5px; text-align: left; vertical-align: top; }
th { background: #e8edf2; font-weight: 600; }
td.n, th.n { text-align: right; }
tr.best td { font-weight: 700; background: #edf7ef; }
.good { color: var(--good); font-weight: 600; } .bad { color: var(--bad); font-weight: 600; }
.run { color: var(--run); font-style: italic; }
.note { font-size: 8.4pt; color: var(--muted); margin: 2px 0 8px; }
.box { border-left: 4px solid var(--accent); background: var(--soft); padding: 6px 9px; margin: 6px 0;
       break-inside: avoid; }
.box.w { border-color: var(--head); background: var(--headbg); }
.exp { break-inside: avoid-page; }
.pb { break-before: page; }
.terms td:first-child { font-weight: 600; white-space: nowrap; }
"""


# --------------------------------------------------------------------------------------------- helpers
def table(head, rows, best=None, num_from=1):
    h = "".join(f"<th{' class=n' if i >= num_from else ''}>{c}</th>" for i, c in enumerate(head))
    out = []
    for k, r in enumerate(rows):
        cls = " class=best" if best is not None and k in (best if isinstance(best, (list, tuple, set)) else [best]) else ""
        out.append(f"<tr{cls}>" + "".join(f"<td{' class=n' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    return f"<table><tr>{h}</tr>{''.join(out)}</table>"


def flow(*items):
    """items: (title, text, kind) boxes or lists of boxes (shown stacked = parallel steps)."""
    parts = []
    for i, it in enumerate(items):
        if i:
            parts.append('<div class="ar">&#10140;</div>')
        if isinstance(it, list):
            parts.append('<div class="split">' + "".join(box(*b) for b in it) + "</div>")
        else:
            parts.append(box(*it))
    return '<div class="flow">' + "".join(parts) + "</div>"


def box(title, text="", kind=""):
    return f'<div class="fb {kind}"><b>{title}</b>{text}</div>'


def load_rows():
    last = {}
    if ROWS.exists():
        for line in ROWS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                last[(r["dataset"], r["pool"], r["selector"], r["k"], r["words"], r["qa_id"])] = r
    return last


def acc(last, ds, pool, sel, words, expected):
    sel = sel.format(ds=ds)
    v = [r["correct"] for (d, p, s, k, w, q), r in last.items() if (d, p, s, k, w) == (ds, pool, sel, 4, words)]
    if len(v) < expected:
        return '<span class="run">running</span>' if not v else f'<span class="run">{len(v)}/{expected} done</span>'
    return f"{100 * np.mean(v):.1f}"


# --------------------------------------------------------------------------------------------- content
def build() -> str:
    last = load_rows()
    P = []
    P.append("""<h1>Experiments from Start to End</h1>
<div class="sub">Local question answering on long and educational videos &middot; Sayed Ilham Azhar Harun &middot;
project <i>local-surf</i> &middot; October 2026 &middot; one 16 GB GPU, small frozen open models</div>
<p>For every experiment: first the <b>structure</b> of the framework that was tested (a flow diagram, read left to
right), then the <b>performance table</b>. Section 10 is the full grid of every framework version, with every
answer model and with or without speech text.</p>
<h2>How to read this document</h2>""")
    P.append('<table class="terms">' + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in [
        ("Accuracy", "share of questions answered correctly (the final score). Every percentage is accuracy unless the table says otherwise."),
        ("Evidence hit", "share of questions where the chosen pictures include the moment human annotators marked as showing the answer (CG-Bench only). It measures <i>looking in the right place</i>, not answering."),
        ("Dev questions", "used during the research to choose settings. Numbers on dev guided decisions."),
        ("Test questions", "locked away until the end and used once. Test numbers are the trustworthy results."),
        ("Confirmed", "the 95% confidence interval of a difference does not include zero (bootstrap over videos)."),
        ("Answer model", "the frozen vision-language model that reads the pictures and answers: Qwen3-VL-2B (small) or Qwen3-VL-4B (larger). Never trained by us."),
        ("Speech text", "up to 600 transcript words given to the answer model next to the pictures. &ldquo;No speech text&rdquo;: the transcript is still used to <i>find</i> pictures, but no words are shown to the answer model."),
        ("Head A", "our trained speech-path model: decides <i>where to look</i> from the spoken words."),
        ("Head B", "our trained picker: decides <i>which 4 candidate pictures</i> to show the answer model."),
    ]) + "</table>")
    P.append(table(["Dataset", "What it is", "Used for"], [
        ["LongVideoBench", "public benchmark (NeurIPS 2024): 3,763 videos, multiple choice, subtitles; we used 440 validation questions", "v1"],
        ["MIT lecture questions (self-made)", "197 multiple-choice questions generated by a larger model from 20 MIT OpenCourseWare lectures, machine-checked; answer-position bias found and corrected", "v2"],
        ["EduVidQA", "public lecture benchmark (EMNLP 2025), open-ended answers", "v3 check only"],
        ["Video-MMMU", "public lecture benchmark (2025): 300 university lecture videos, 10-option multiple choice. <b>Perception</b> = read or see what is on the slide; <b>Comprehension</b> = understand what was taught", "v3 (lectures)"],
        ["CG-Bench", "public long-video benchmark (2024): videos of 10&ndash;80 min, human-marked answer seconds; we used 51 English-subtitled videos (train 37 / dev 8 / test 6)", "v3 (long everyday videos), head training"],
    ], num_from=9))

    # ---------------- 1. v1
    P.append('<h2 class="pb">1 &middot; Version 1: cost-aware controller (LongVideoBench)</h2><div class="exp">')
    P.append(flow(("Inputs", "question + options, video, subtitles", "io"), ("BM25 search", "transcript windows that match the question"),
                  ("Candidate moments", "times inside the windows"), ("MobileCLIP scout", "scores pictures on demand"),
                  ("Controller", "look / expand / stop, gain vs time cost"), ("Answer model 2B", "&le; 8 pictures + transcript", "io")))
    P.append(table(["Frame strategy (440 questions)", "Multiple choice", "Open-ended (local judge, not human-checked)", "Time per question"], [
        ["Transcript only (no pictures)", "32.5", "44.1", "1.14 s"], ["Evenly spaced pictures + transcript", "39.3", "33.4", "1.26 s"],
        ["Transcript-retrieved moments", "36.4", "32.0", "1.11 s"], ["MobileCLIP top pictures", "41.6", "32.3", "2.25 s"],
        ["Cost-aware controller", "37.5", "29.8", "1.47 s"]], best=3))
    P.append('<p class="note">Research idea (train the controller to look when important speech is damaged): not supported; '
             'damage changed the answer in only 13% (train) / 4% (dev) of cases. Chance level 21.4%.</p></div>')

    # ---------------- 2. v2
    P.append('<h2>2 &middot; Version 2: two candidate paths and first learned heads (MIT lectures)</h2><div class="exp">')
    P.append(flow(("Inputs", "question + options, lecture, transcript", "io"), ("BM25 search", "4 transcript windows"),
                  [("Path A: speech", "MiniLM scores lines in the windows; picture at line end"), ("Path B: pictures", "scan inside the windows, MobileCLIP")],
                  ("Pool", "~11 candidates"), ("Picker", "rule or learned Head A / Head B", "h"),
                  ("Answer model 2B", "4 pictures + 120 BM25 words", "io")))
    P.append(table(["Picker (197 questions, 20 lectures, each tested by models that never saw it)", "Accuracy"], [
        ["Transcript only", "15.2"], ["MobileCLIP top-4", "34.5"], ["MobileCLIP + diversity", "36.0"],
        ["Learned Head B (with screen text)", "34.7"], ["Learned Head A frame credit", "30.5 (confirmed worse)"],
        ["Zero-shot transcript relevance (no training)", "38.6"]], best=5))
    P.append(table(["Follow-up", "Result"], [
        ["Learned &ldquo;4th picture&rdquo; completion", "&minus;1.0 (refuted)"],
        ["Spread the picture scan over all windows", "34.5 &rarr; 39.1 (+4.6, not confirmed)"],
        ["Same on LongVideoBench (transfer check)", "43.0 &rarr; 42.5 (no gain)"],
        ["Option-rotation scoring (fixes answer-position bias)", "MobileCLIP top-4: 49.7 rotated vs 34.3 transcript only"]], num_from=9))
    P.append("</div>")

    # ---------------- 3. gates
    P.append('<h2>3 &middot; v3 benchmark check: can pictures change the answer?</h2><div class="exp">')
    P.append(flow(("Question", "", "io"), [("No pictures", "transcript only"), ("4 evenly spaced pictures", ""),
                                           ("4 chosen pictures", "MobileCLIP best match"), ("4 human-marked pictures", "upper reference")],
                  ("Answer model 2B", "", "io")))
    P.append(table(["Benchmark (dev)", "No pictures", "Evenly spaced", "Chosen pictures", "Human-marked pictures", "Decision"], [
        ["EduVidQA (open-ended, FactQA score)", "0.587", "0.493", "0.498", "&mdash;", "pictures do not help: not used for picture claims"],
        ["Video-MMMU Perception", "40.2", "52.6", "56.7", "&mdash;", "pictures help (+16.5, confirmed)"],
        ["CG-Bench", "15.9", "17.1", "40.2", "63.4", "pictures help (+24.4); large room"]], num_from=9))
    P.append("</div>")

    # ---------------- 4. baseline cycle
    P.append('<h2 class="pb">4 &middot; v3 tuning of the rule-based framework (dev only)</h2><div class="exp">')
    P.append(flow(("Inputs", "", "io"), ("BM25 search", "4 windows"),
                  [("Path A: speech rule", "MiniLM on lines inside the windows &rarr; 6 pictures"),
                   ("Path B: <i>where to look?</i>", "inside windows / whole video evenly / dense whole video / hybrid")],
                  ("Pool", "~10&ndash;12"), ("Picker: <i>which rule?</i>", "clip / clip+options / MMR / MMR+options / speech relevance"),
                  ("Answer model 2B", "4 pictures + <i>0 / 120 / 300 words</i>", "io")))
    P.append(table(["Setting (2B, 4 pictures)", "Long videos (CG-Bench, 82)", "Lectures screen-reading (98)", "Right picture among candidates (CG-Bench)"], [
        ["Starting framework v2: Path B inside windows, 120 BM25 words", "28.0", "56.1", "33%"],
        ["Path B scans the whole video evenly", "34.1", "57.1", "50%"],
        ["Path B dense whole-video scan", "37.8", "52.0", "62%"],
        ["Path B hybrid (half windows, half dense) + MMR on question+options", "35.4", "58.2", "60%"],
        ["&hellip; + no transcript words (frozen <b>best framework</b>)", "40.2", "63.3", "60%"],
        ["&hellip; with 2 / 8 pictures", "37.8 / 39.0", "52.0 / 60.2", ""]], best=4))
    P.append('<p class="note">Best framework vs starting framework on dev: +12.2 on long videos (confirmed).</p></div>')

    # ---------------- 5. Head A
    P.append('<h2>5 &middot; Head A: trained speech-to-moment locator</h2><div class="exp">')
    P.append(flow(("Whole transcript", "every line (not only the windows)", "io"),
                  ("Two frozen text models", "MiniLM (question) + BGE (question + options)"),
                  ("24 facts per line", "neighbour relevance, word overlap, length, position, question type"),
                  ("Small trained network", "1,889 numbers; corrects the score", "h"),
                  ("Lines vote", "nearby relevant lines stack into peaks"), ("6 moments", "&rarr; 6 pictures for the pool", "io")))
    P.append('<p class="note">Trained on 877 CG-Bench questions (83 videos) whose answer seconds were marked by humans; '
             'measured on 164 new dev questions. Number = evidence hit: at least one of 6 moments inside the marked seconds.</p>')
    P.append(table(["Head A method", "Evidence hit (6 moments)"], [
        ["Zero-shot MiniLM, lines inside windows (speech rule of the best framework)", "25.0"],
        ["Zero-shot MiniLM, whole transcript", "28.0"], ["Fine-tuned MiniLM", "~8 (training destroyed its ranking)"],
        ["Larger zero-shot rerankers (BGE up to 568M)", "up to 31.1"], ["Light trained network on frozen scores", "32.7"],
        ["&hellip; with learned speech-to-picture offsets (original idea)", "31.7"],
        ["&hellip; lines vote + BGE facts, 3 copies averaged (<b>chosen Head A</b>)", "36.6 (+8.5, confirmed)"]], best=6))
    P.append(table(["Inside the best framework (2B, no speech text, dev)", "Right picture among candidates", "Accuracy long videos", "Accuracy lectures screen-reading"], [
        ["Speech rule (best framework)", "59.8", "40.2", "63.3"], ["Head A", "64.6 (+4.9, confirmed)", "39.0", "62.2"]]))
    P.append("</div>")

    # ---------------- 6. Head B
    P.append('<h2 class="pb">6 &middot; Head B: trained picker of the final 4 pictures</h2><div class="exp">')
    P.append(flow(("Pool", "~10&ndash;12 candidates", "io"), ("18 facts per picture", "match to question / options, path, speech, time, look-alikes"),
                  ("Small trained scorer", "trained on human-marked seconds or on the answer model's feedback", "h"),
                  ("Set choice", "separate the answer options, avoid look-alikes"), ("4 pictures", "&rarr; answer model", "io")))
    P.append('<p class="note">Trained on 373 CG-Bench questions; the feedback version used 4,203 answers given with one picture at a time. '
             'Numbers: accuracy difference to the MMR rule, pooled over 278 dev questions (long videos + both lecture tracks), 2B, no speech text.</p>')
    P.append(table(["Head B method", "Difference to the MMR rule"], [
        ["Option-evidence set rule (no training, original idea)", "&minus;1.4"], ["Trained on human-marked seconds", "&minus;1.4"],
        ["&hellip; + option-evidence term (<b>chosen Head B</b>, &ldquo;ev_set&rdquo;)", "&minus;0.4"],
        ["Trained on answer-model feedback", "&minus;3.6"], ["Confidence choice: answer with 4 picture sets, keep the most confident", "+1.1 (not confirmed)"]], best=2))
    P.append("</div>")

    # ---------------- 7. final test
    res = json.loads(Path("reports/v3_final/test_results.json").read_text(encoding="utf-8"))
    sets = ("cgbench_test", "videommmu_test", "videommmu_comp_test")
    names = {"R1": "No pictures", "R2": "4 evenly spaced pictures (standard rule)", "R3": "Dense MobileCLIP top-4 (standard rule)",
             "S0": "Starting framework (v2)", "S1": "<b>Best framework</b>", "S2": "Best + Head A", "S3": "Best + Head B",
             "S4": "Best + Head A + Head B", "S5": "Best + confidence choice"}
    rows = [[names[k]] + [f"{100 * s[d]['accuracy']:.1f}" for d in sets] + [f"{100 * s['pooled']['accuracy']:.1f}"]
            for k, s in res["systems"].items()]
    P.append('<h2>7 &middot; Final locked test (declared in advance, run once; 2B, no speech text except v2)</h2><div class="exp">')
    P.append(flow(("Same 459 test questions", "never used before", "io"), ("9 systems", "same answer model, 4 pictures"),
                  ("Accuracy", "per benchmark and overall"), ("Paired comparison", "95% intervals over videos", "io")))
    P.append(table(["System", "Long videos (55)", "Lectures screen-reading (202)", "Lectures understanding (202)", "All 459"], rows, best=4))
    lab = {("S1", "S0"): "Best &minus; starting framework", ("S2", "S1"): "Head A &minus; best", ("S3", "S1"): "Head B &minus; best",
           ("S4", "S1"): "Head A+B &minus; best", ("S5", "S1"): "Confidence choice &minus; best",
           ("S1", "R3"): "Best &minus; dense MobileCLIP", ("S1", "R2"): "Best &minus; evenly spaced"}

    def cell(e):
        sig = e["ci"][0] > 0 or e["ci"][1] < 0
        t = f"{100 * e['diff']:+.1f} ({100 * e['ci'][0]:+.1f} to {100 * e['ci'][1]:+.1f})"
        return f'<span class="{"good" if e["diff"] > 0 else "bad"}">{t}</span>' if sig else t
    P.append(table(["Difference in points (95% interval)", "Long videos", "Screen-reading", "Understanding", "All 459"],
                   [[lab[(c["a"], c["b"])]] + [cell(c[d]) for d in sets] + [cell(c["pooled"])] for c in res["comparisons"]]))
    P.append('<p class="note">Green / red = confirmed gain / loss. The long-video test has only 6 videos, so its intervals are unreliable.</p></div>')

    # ---------------- 8. speech text
    P.append('<h2 class="pb">8 &middot; Speech text for the answer model: keeping the important words</h2><div class="exp">')
    P.append(flow(("4 chosen pictures", "", "io"), [("Speech around the pictures", "lines within 15 s, nearest first (300 words)"),
                                                     ("Best lines of the whole video", "word match + meaning match combined, with neighbours")],
                  ("Up to 600 words", "in time order with timestamps"), ("Answer model", "pictures + words", "io")))
    P.append(table(["Text given to the answer model (long videos, dev)", "Important speech kept", "Open-ended correct", "Multiple choice"], [
        ["Starting framework v2 (BM25, 120 words)", "28%", "13.4", "28.0"], ["Best pictures + BM25 120 words", "28%", "17.1", "35.4"],
        ["Best pictures, no words", "0%", "19.5", "40.2"], ["Best pictures + whole transcript (capped 1,500 words)", "66%", "19.2", "40.2"],
        ["Best pictures + meaning-based search, 300 words", "62%", "20.7", "37.8"],
        ["Best pictures + speech around pictures + search, 600 words (<b>chosen</b>)", "68%", "22.0", "37.8"]], best=5))
    P.append('<p class="note">Important speech kept = the words include the speech within 10 s of the human-marked answer seconds. '
             'Open-ended answers graded by a free NVIDIA-hosted judge; it agreed with hand grading on 40 of 40 checked answers. '
             'Chosen policy vs v2 on open-ended: +8.5 (confirmed).</p></div>')

    # ---------------- 9. 4B
    P.append('<h2>9 &middot; Larger answer model (4B) on the test questions</h2><div class="exp">')
    P.append(table(["System (4B unless stated)", "Long videos (55)", "Screen-reading (202)", "Understanding (202)", "All 459"], [
        ["Starting framework v2", "38.2", "60.9", "43.1", "50.3"], ["Best framework, no speech text", "41.8", "62.9", "45.5", "52.7"],
        ["<b>Best framework + speech text</b>", "49.1", "67.8", "45.0", "55.6"],
        ["Dense MobileCLIP top-4 (standard rule)", "40.0", "54.8*", "&mdash;", "&mdash;"],
        ["Best framework + speech text with the 2B model", "49.1", "49.0", "31.7", "41.4"]], best=2))
    P.append('<p class="note">Best + speech text vs starting framework (4B): +5.2 overall (confirmed); +6.9 on screen-reading lectures '
             '(confirmed); the speech text adds +2.8 over no speech text (confirmed). *188 of 202 questions (time budget ended).</p></div>')

    # ---------------- 10. grid
    P.append('<h2 class="pb">10 &middot; Full grid: every framework version &times; answer model &times; speech text</h2>')
    P.append(flow(("Inputs", "", "io"), ("BM25 search", ""),
                  [("Path A", "speech rule <i>or</i> Head A", "h"), ("Path B", "hybrid picture search")],
                  ("Pool", ""), ("Picker", "MMR rule <i>or</i> Head B", "h"), ("Answer model", "2B <i>or</i> 4B; speech text 0 <i>or</i> 600 words", "io")))
    versions = [("Best framework", "hybrid", "mmropt"), ("Best + Head A", "hybrid_ahead", "mmropt"),
                ("Best + Head B", "hybrid", "file:{ds}_hybrid_ev_set"), ("Best + Head A + Head B", "hybrid_ahead", "file:{ds}_hybrid_ahead_ev_set")]
    cols = [("cgbench", 82), ("cgbench_test", 55), ("videommmu", 98), ("videommmu_test", 202), ("videommmu_comp", 98), ("videommmu_comp_test", 202)]
    for title, suffix, words in (("2B answer model, no speech text", "", 0), ("2B answer model, with speech text (600 words)", "|tp=frames+hybrid", 600),
                                 ("4B answer model, no speech text", "|ans=4b", 0), ("4B answer model, with speech text (600 words)", "|tp=frames+hybrid|ans=4b", 600)):
        P.append(f"<h3>{title}</h3>")
        rows = [[v] + [acc(last, ds, pool, sel + suffix, words, n) for ds, n in cols] for v, pool, sel in versions]
        P.append(table(["Version", "Long videos dev", "Long videos TEST", "Screen-reading dev", "Screen-reading TEST",
                        "Understanding dev", "Understanding TEST"], rows))
    P.append('<p class="note">All combinations completed (research log steps 54&ndash;55). Test numbers in this grid beyond the '
             'declared comparison of section 7 are follow-up results.</p>')
    P.append('<div class="box">With the <b>4B</b> model and no speech text, all three head versions score above the best framework on '
             'dev and test; Head A + Head B: dev +4.3, test +2.2, and <b>+3.0 over dev and test together (737 questions, confirmed)</b>. '
             'With the 600-word speech text the gain mostly disappears on test (+1.2 over dev and test, not confirmed). '
             'With the 2B model no head version differs reliably from the best framework.</div>')

    # ---------------- 11. summary
    P.append("""<h2>11 &middot; Summary: what gave what</h2>""")
    P.append(table(["Framework / component", "What it achieved", "On which data"], [
        ["v1 MobileCLIP top pictures", "41.6% (vs 32.5% transcript only)", "LongVideoBench"],
        ["v3 best framework (rules, tuned)", "50.9% vs 34.5% for the starting framework (+16.4, confirmed)", "long everyday videos, test"],
        ["Best framework + speech text + 4B model", "<b>55.6% vs 50.3%</b> (+5.2, confirmed); 67.8% on screen-reading lectures (+6.9, confirmed)", "all test questions; lectures"],
        ["Speech-text selection", "keeps 68% of the important speech (v2: 28%); open-ended +8.5 (confirmed)", "long videos, dev"],
        ["Head A (trained)", "finds the answer moment 36.6% vs 28.0% (+8.5, confirmed); right picture among candidates 64.6% vs 59.8% (confirmed); accuracy not improved", "long everyday videos"],
        ["Head B (trained)", "chosen pictures include the answer moment 50.9% vs 47.3% (with Head A: 56.4%); accuracy not improved", "long everyday videos, test"],
        ["Head A + Head B with the 4B model, no speech text", "+3.0 over the best framework (dev and test together, 737 questions, confirmed); test alone 54.9% vs 52.7%", "all benchmarks; lectures test 57.7% vs 54.2%"],
        ["Confidence choice", "best 2B overall row, 42.3% vs 41.4% (not confirmed)", "all test questions"]], num_from=9))
    P.append('<div class="box">Best system for educational videos so far: <b>best framework + speech text + 4B answer model</b> '
             '(67.8% screen-reading, 45.0% understanding, test; 55.6% on all 459). The heads help the 4B model when it gets no speech text, but do not add to the best system once the speech text is given.</div>')
    return "\n".join(P)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    html_path = OUT / "experiments_report.html"
    html_path.write_text("<!doctype html><html lang=en><head><meta charset=utf-8><title>Experiments Report</title>"
                         f"<style>{CSS}</style></head><body>{build()}</body></html>", encoding="utf-8")
    print(f"wrote {html_path}")
    if a.pdf:
        pdf = (OUT / "experiments_report.pdf").resolve()
        for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
            if Path(exe).exists():
                subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                                f"--print-to-pdf={pdf}", html_path.resolve().as_uri()], check=True, timeout=120)
                print(f"wrote {pdf}")
                break


if __name__ == "__main__":
    main()
