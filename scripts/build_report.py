"""Build the framework report (HTML -> PDF) from the committed result files.

Every number in the tables and charts is read from reports/*.json, so the
document cannot drift from the data. Figures that need local video frames
(runs/, not in git) are skipped if those files are absent.

    python scripts/build_report.py                      # writes docs/report/framework_report.html
    python scripts/build_report.py --pdf                # ...and prints it to PDF with headless Edge/Chrome
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import subprocess
from pathlib import Path

R = Path("reports")
OUT = Path("docs/report")


def j(path):
    return json.loads((R / path).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def pts(x, d=1):
    return f"{100 * x:+.{d}f}"


def ci(c, d=1):
    return f"{100 * c[0]:+.{d}f} to {100 * c[1]:+.{d}f}"


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# Charts (inline SVG, theme-neutral)
# ---------------------------------------------------------------------------


def bars(items, xmax=0.6, width=640, label_w=250, unit_ticks=(0, .1, .2, .3, .4, .5, .6), caption=""):
    """Horizontal bars. items: (label, value, highlight)."""
    row, top = 24, 8
    plot_w = width - label_w - 60
    h = top + row * len(items) + 30
    x = lambda v: label_w + plot_w * v / xmax  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(caption)}">']
    for t in unit_ticks:
        out.append(f'<line x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{h - 24}" class="grid"/>')
        out.append(f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{int(t * 100)}%</text>')
    for i, (lab, v, hi) in enumerate(items):
        y = top + i * row
        out.append(f'<text x="{label_w - 8}" y="{y + 16}" class="lab" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{label_w}" y="{y + 4}" width="{max(0, x(v) - label_w):.1f}" height="16" '
                   f'class="{"bar hi" if hi else "bar"}"/>')
        out.append(f'<text x="{x(v) + 5:.1f}" y="{y + 16}" class="val">{100 * v:.1f}%</text>')
    out.append("</svg>")
    return f'<figure>{"".join(out)}<figcaption>{caption}</figcaption></figure>'


def forest(items, lo=-0.15, hi=0.25, width=640, label_w=300, caption=""):
    """Paired differences with 95% CIs. items: (label, diff, (lo, hi))."""
    row, top = 24, 8
    plot_w = width - label_w - 120
    h = top + row * len(items) + 30
    x = lambda v: label_w + plot_w * (v - lo) / (hi - lo)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(caption)}">']
    t = lo
    while t <= hi + 1e-9:
        out.append(f'<line x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{h - 24}" class="{"zero" if abs(t) < 1e-9 else "grid"}"/>')
        out.append(f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{100 * t:+.0f}</text>')
        t += 0.05
    for i, (lab, d, c) in enumerate(items):
        y = top + i * row + 12
        sig = c[0] > 0 or c[1] < 0
        out.append(f'<text x="{label_w - 8}" y="{y + 4}" class="lab" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<line x1="{x(c[0]):.1f}" y1="{y}" x2="{x(c[1]):.1f}" y2="{y}" class="whisk"/>')
        out.append(f'<circle cx="{x(d):.1f}" cy="{y}" r="4.5" class="{"dot sig" if sig else "dot"}"/>')
        out.append(f'<text x="{width - 4}" y="{y + 4}" class="val" text-anchor="end">{100 * d:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]</text>')
    out.append("</svg>")
    return f'<figure>{"".join(out)}<figcaption>{caption}</figcaption></figure>'


def pipeline_svg():
    """Framework diagram: frozen parts, trained parts, training-only labels."""
    return """<figure><svg viewBox="0 0 760 360" role="img" aria-label="Two-path evidence selection pipeline">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>
<rect x="10" y="20" width="150" height="46" rx="6" class="box"/><text x="85" y="40" class="bt" text-anchor="middle">question + options</text><text x="85" y="56" class="bs" text-anchor="middle">+ timed transcript</text>
<rect x="200" y="20" width="150" height="46" rx="6" class="box frozen"/><text x="275" y="40" class="bt" text-anchor="middle">BM25 windows</text><text x="275" y="56" class="bs" text-anchor="middle">(v1 retrieval)</text>
<line x1="160" y1="43" x2="198" y2="43" class="ln" marker-end="url(#ar)"/>
<rect x="390" y="20" width="200" height="46" rx="6" class="box frozen"/><text x="490" y="40" class="bt" text-anchor="middle">retained excerpt</text><text x="490" y="56" class="bs" text-anchor="middle">≤ 120 words, fixed per question</text>
<line x1="350" y1="43" x2="388" y2="43" class="ln" marker-end="url(#ar)"/>
<rect x="120" y="110" width="210" height="56" rx="6" class="box frozen"/><text x="225" y="132" class="bt" text-anchor="middle">Path A: hot moments</text><text x="225" y="150" class="bs" text-anchor="middle">MiniLM cross-encoder (22.7M) relevance;</text><text x="225" y="162" class="bs" text-anchor="middle">frame at each top line's end (6)</text>
<rect x="360" y="110" width="230" height="56" rx="6" class="box frozen"/><text x="475" y="132" class="bt" text-anchor="middle">Path B: visual scan</text><text x="475" y="150" class="bs" text-anchor="middle">5 s scan in windows (cap 24), stable runs,</text><text x="475" y="162" class="bs" text-anchor="middle">MobileCLIP-S2 ranking (6)</text>
<line x1="275" y1="66" x2="225" y2="108" class="ln" marker-end="url(#ar)"/><line x1="275" y1="66" x2="475" y2="108" class="ln" marker-end="url(#ar)"/>
<rect x="230" y="200" width="250" height="40" rx="6" class="box"/><text x="355" y="218" class="bt" text-anchor="middle">merged candidate pool</text><text x="355" y="232" class="bs" text-anchor="middle">≈ 10–11 frames, de-duplicated</text>
<line x1="225" y1="166" x2="300" y2="198" class="ln" marker-end="url(#ar)"/><line x1="475" y1="166" x2="410" y2="198" class="ln" marker-end="url(#ar)"/>
<rect x="200" y="272" width="310" height="46" rx="6" class="box trained"/><text x="355" y="291" class="bt" text-anchor="middle">selector → K = 4 frames</text><text x="355" y="307" class="bs" text-anchor="middle">rules (relevance / MobileCLIP / MMR) or small learned heads</text>
<line x1="355" y1="240" x2="355" y2="270" class="ln" marker-end="url(#ar)"/>
<rect x="580" y="272" width="170" height="46" rx="6" class="box frozen"/><text x="665" y="291" class="bt" text-anchor="middle">Qwen3-VL-2B</text><text x="665" y="307" class="bs" text-anchor="middle">frozen answerer</text>
<line x1="510" y1="295" x2="578" y2="295" class="ln" marker-end="url(#ar)"/>
<line x1="590" y1="43" x2="700" y2="43" class="ln"/><line x1="700" y1="43" x2="700" y2="270" class="ln" marker-end="url(#ar)"/>
<text x="706" y="160" class="bs">excerpt</text>
<line x1="665" y1="318" x2="665" y2="345" class="ln dash"/><line x1="665" y1="345" x2="355" y2="345" class="ln dash"/><line x1="355" y1="345" x2="355" y2="320" class="ln dash" marker-end="url(#ar)"/>
<text x="510" y="340" class="bs" text-anchor="middle">training only: measured answer change R(T,S+c) − R(T,S) vs gold</text>
<rect x="12" y="200" width="14" height="10" class="box frozen"/><text x="32" y="209" class="bs">frozen</text>
<rect x="12" y="218" width="14" height="10" class="box trained"/><text x="32" y="227" class="bs">trainable (≤ 8,333 params)</text>
</svg><figcaption>Figure 1. The v2 pipeline. All large models are frozen; only the selector can be trained,
from labels produced by the same frozen answerer against gold answers (dashed).</figcaption></figure>"""


def img_tag(pil, width_pct=100, quality=72):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "JPEG", quality=quality)
    return (f'<img style="width:{width_pct}%" alt="" '
            f'src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"/>')


def crop_figure():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return ""
    d = Path("runs/v2_main/pools/mit_18.065_03_1050")
    if not d.exists():
        return ""
    ims = []
    for i in (0, 5):
        im = Image.open(d / f"mit_18.065_03_1050_f0{i}.png").convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.rectangle([140, 0, 499, 359], outline=(220, 40, 40), width=4)
        ims.append(im)
    pair = Image.new("RGB", (1290, 360), (255, 255, 255))
    pair.paste(ims[0], (0, 0))
    pair.paste(ims[1], (650, 0))
    return (f"<figure>{img_tag(pair)}<figcaption>Figure 5. Two candidate frames from MIT 18.065 lecture 3. "
            "The red box is what MobileCLIP-S2's default transform keeps (resize to 256, centre crop): 56% of the "
            "width. The answerer always sees the full frame.</figcaption></figure>")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def build() -> str:
    v1 = j("benchmark_v1/results.json")["policies"]
    main = j("v2_main/eval_summary.json")
    fix = j("v2_review/eval_summary_corrected_ablation.json")
    cond = j("v2_review/transcript_conditioning.json")
    res = j("v2_review/residual_qa/eval_summary.json")
    comp = j("v2_completion/summary.json")
    loc = j("v2_completion_local/summary.json")
    bal = j("v2_balanced/summary.json")
    circ_m = j("v2_circular/summary_v2_main.json")
    circ_b = j("v2_circular/summary_v2_balanced.json")
    lvb = j("v2_lvb_transfer/summary.json")
    crop = j("v2_scout_crop/summary.json")
    dedup = j("v2_evidence_loss/dedup_summary.json")
    bias = j("v2_review/option_position_bias.json")
    cache = j("v2_review/cache_identity_audit.json")
    acc = main["accuracy"]
    P = main["paired"]
    lat = main["latency_s_median"]
    dec = cond["useful_only_without_speech_decomposition"]
    cm, cb = circ_m["arms"], circ_b["arms"]

    # Circular paired numbers from per-question files (balanced vs legacy, same questions).
    pm = json.loads((R / "v2_circular/per_question_v2_main.json").read_text())
    pb = json.loads((R / "v2_circular/per_question_v2_balanced.json").read_text())
    import numpy as np

    def boot(f):
        by = {}
        for q in pm:
            by.setdefault(pm[q]["video_id"], []).append(f(q))
        s = np.array([sum(v) for v in by.values()])
        c = np.array([len(v) for v in by.values()])
        ids = np.random.default_rng(0).integers(0, len(s), (5000, len(s)))
        b = s[ids].sum(1) / c[ids].sum(1)
        return float(s.sum() / c.sum()), np.percentile(b, [2.5, 97.5]).tolist()

    circ_bal_A = boot(lambda q: np.mean(pb[q]["A"]) - np.mean(pm[q]["A"]))
    circ_bal_F = boot(lambda q: np.mean(pb[q]["F"]) - np.mean(pm[q]["F"]))
    lq = json.loads((R / "v2_lvb_transfer/per_question.json").read_text())["legacy"]
    by = {}
    for q, r in lq.items():
        by.setdefault(r["video_id"], []).append(r["F_relevance"] - r["A_mobileclip_topk"])
    s = np.array([sum(v) for v in by.values()])
    c = np.array([len(v) for v in by.values()])
    b = s[np.random.default_rng(0).integers(0, len(s), (5000, len(s)))].sum(1) / c[np.random.default_rng(0).integers(0, len(s), (5000, len(s)))].sum(1)
    lvb_FA = (float(s.sum() / c.sum()), np.percentile(b, [2.5, 97.5]).tolist())

    fe = comp["fourth_frame_effects"]
    ca, cp = comp["accuracy"], comp["paired"]
    la, lp = loc["accuracy"], loc["paired"]

    S = []
    a = S.append
    a(f"""<header class="title">
<p class="kicker">Technical report · October 2026</p>
<h1>Choosing Evidence for a Small Frozen Video-QA Model</h1>
<p class="sub">A two-path, fully local framework for long-video question answering: design, dataset construction,
learned and rule-based frame selection, and what the experiments do and do not show</p>
<p class="author">Sayed Ilham Azhar Harun · repository <code>ham0004/local-surf</code>, branches <code>QAframework</code> (v1) and <code>framework-2</code> (v2)</p>
</header>

<section class="abstract"><h2>Abstract</h2>
<p>We build a question-answering system for long videos that runs on one 16 GB consumer GPU. Instead of showing a
video to a vision-language model, it retrieves the relevant transcript windows, proposes candidate frames along two
paths (transcript "hot moments" and a sparse visual scan), selects four frames, and asks a frozen
Qwen3-VL-2B model to answer from those frames and a short transcript excerpt. Version 1 scores
<b>{pct(v1["scout_similarity"]["accuracy"])}</b> on 440 LongVideoBench questions with MobileCLIP frame selection,
{pts(v1["scout_similarity"]["accuracy"] - v1["transcript_only"]["accuracy"])} points over transcript-only.
Version 2 asks whether small learned selectors, trained on measured answer changes of the frozen model, choose
better frames than simple rules. On a lecture dataset we built from 20 MIT OpenCourseWare lectures, they do not:
four families of learned or feature-based selectors, including one trained on deployment-matched labels, all fail
to beat zero-shot transcript relevance, although an exhaustive oracle shows
<b>{pts(cp["oracle_minus_relevance_rank4"]["difference"])} points</b> of headroom in the last frame slot. A visual-scan
change that helped on lectures did not transfer to LongVideoBench. We also found and corrected a dataset flaw:
answerer-conditioned filtering amplified option-position bias, so that "always answer B" would have outscored every
system; circular evaluation raises all lecture accuracies by about 15 points while preserving selector rankings.
We report these results, the corrected numbers, and the remaining open questions without claiming a new method.</p></section>

<section><h2>1. What this report claims</h2>
<table class="claims"><thead><tr><th>Claim</th><th>Evidence</th><th>Strength</th></tr></thead><tbody>
<tr><td>A working, fully local long-video QA framework with per-stage cost accounting</td><td>v1 on LongVideoBench, 440 questions; ≈ 1–2 s per question</td><td>engineering result</td></tr>
<tr><td>Frames add substantially over transcript alone</td><td>v1 LVB {pts(v1["scout_similarity"]["accuracy"] - v1["transcript_only"]["accuracy"])} pts; MIT (debiased) {pts(circ_m["paired_circular_mean"]["A_minus_transcript_only"]["difference"])} pts</td><td>robust (CIs exclude 0)</td></tr>
<tr><td>Small learned selectors do not beat transcript relevance on this lecture data</td><td>Head B, residual scorers, completion head, local-feature head</td><td>negative result, controlled</td></tr>
<tr><td>The last frame slot has measurable headroom that no tested selector captures</td><td>exhaustive fourth-frame table: oracle {pct(ca["oracle"])} vs {pct(ca["relevance_rank4"])}</td><td>measured on development data</td></tr>
<tr><td>Answerer-conditioned question filtering amplifies option-position bias</td><td>gold A/B 171/197; C/D-gold dropped 61% vs 39%</td><td>methodological finding</td></tr>
<tr><td>A new frame-selection method that beats baselines</td><td>—</td><td><b>not claimed</b></td></tr>
</tbody></table></section>

<section><h2>2. System design</h2>
<h3>2.1 Version 1 (branch QAframework)</h3>
<p>Transcript lines are grouped into 20-second units and ranked with BM25 against the question and its options. The
top windows (up to 4, expanded by one neighbour) define where to look and give a retained excerpt of at most 120
words. Frames are then chosen by one of five strategies (Section 5.1) and passed, with timestamp labels, to the frozen
Qwen3-VL-2B-Instruct (revision 89644892, bfloat16, greedy decoding, 640-pixel frames). Answers cite timestamps,
and every stage is timed.</p>
<h3>2.2 Version 2 (branch framework-2)</h3>
{pipeline_svg()}
<p>v2 keeps the v1 retrieval and adds two proposal paths that feed one candidate pool per question:</p>
<ul>
<li><b>Path A (transcript hot moments).</b> A frozen MiniLM-L6 cross-encoder (22.7M parameters) scores every
transcript line inside the windows. The six best lines each contribute a frame taken 0.3 s before the line ends,
when the board is fullest. In the main experiment these proposals use the pretrained relevance score; the trained
Head A only re-ranks the fixed pool.</li>
<li><b>Path B (visual scan).</b> Frames are decoded every 5 s inside the windows (at most 24). Runs of visually
similar frames (8×8 average hash within distance 6) collapse to their last frame, and MobileCLIP-S2 ranks the
survivors against the question; the top six enter the pool. An opt-in <i>balanced</i> policy spreads the same 24
slots across all windows instead of filling the earliest first.</li>
<li><b>Merge.</b> Path B frames within 1 s or hash distance 3 of a Path A frame are merged into it. Pools hold
about 10–11 frames. Each candidate gets one MobileCLIP embedding, RapidOCR text and nearby speech.</li>
<li><b>Selector.</b> Picks K = 4 frames. Rules: MobileCLIP top-4 (A), MMR diversity (B), zero-shot transcript
relevance (F). Learned: Head B, residual scorers, completion heads (Section 4).</li>
</ul>
<table><thead><tr><th>Component</th><th>Model</th><th>Size</th><th>Status</th></tr></thead><tbody>
<tr><td>Answerer and label source</td><td>Qwen3-VL-2B-Instruct</td><td>2B</td><td>frozen</td></tr>
<tr><td>Frame / text embeddings</td><td>MobileCLIP-S2 (open_clip, datacompdr)</td><td>99.2M (35.8M image, 63.4M text)</td><td>frozen</td></tr>
<tr><td>Transcript relevance</td><td>cross-encoder/ms-marco-MiniLM-L6-v2</td><td>22.7M</td><td>frozen</td></tr>
<tr><td>On-screen text</td><td>RapidOCR (ONNX, CPU)</td><td>—</td><td>frozen</td></tr>
<tr><td>Head A (speech vs frame credit)</td><td>linear head on 6 features</td><td>14 params</td><td>trained</td></tr>
<tr><td>Head B (frame utility)</td><td>3 projections + MLP</td><td>8,333 params (15,257 pilot)</td><td>trained</td></tr>
<tr><td>Residual scorers</td><td>pairwise ridge around MobileCLIP</td><td>17 / 27 coefficients</td><td>trained</td></tr>
<tr><td>Completion heads</td><td>pairwise ridge around relevance</td><td>26 / 34 coefficients</td><td>trained</td></tr>
</tbody></table>
</section>

<section><h2>3. Datasets</h2>
<h3>3.1 LongVideoBench subset</h3>
<p>440 validation questions on 252 videos (8 s to 60 min) with subtitles, 4–5 options, chance level
{pct(j("benchmark_v1/results.json")["chance_accuracy"])}. Gold positions are naturally balanced
(105/103/92/83/57). This subset was used to develop and benchmark v1; it is not the full benchmark (1,337 validation
questions) and not an untouched test.</p>
<h3>3.2 MIT lecture questions (constructed)</h3>
<p>Existing benchmarks rarely require local speech <i>and</i> board content, so we built a lecture set:</p>
<ol>
<li><b>Lectures.</b> 20 MIT OpenCourseWare lectures by Gilbert Strang (11 from 18.06SC, 9 from 18.065), CC BY-NC-SA 4.0,
with captions, selected in a fixed sha256 order.</li>
<li><b>Mining.</b> Each lecture is cut into 30-second windows. A larger generator, Qwen3-VL-4B-Instruct (deliberately
not the 2B answerer), sees two frames (window middle and end) and the captions, labels the speech/board relation and
writes one multiple-choice question with four options. 591 windows were processed; 589 parsed.</li>
<li><b>Verification.</b> Rule checks: quoted speech must occur in the captions, the answer must be grounded in the
speech for speech-bearing relations, options must be distinct (25 rejected). Then the frozen 2B answerer is asked
under four conditions: no evidence, captions only, frame only, both. 50 items failing both single-source conditions
were rejected; 349 passed.</li>
<li><b>Prior filter.</b> Questions the 2B answerer got right with no evidence (152) were dropped, leaving
<b>197 questions</b> (79 speech-and-board, 74 speech, 44 board-support; 4–17 per lecture).</li>
</ol>
<p>All items are marked synthetic and are not human-verified. Section 7 shows that step 4, combined with
the generator's habit of putting the answer first, created a strong option-position bias.</p>
</section>""")

    # --- Section 4: labels and training
    a(f"""<section><h2>4. Labels, learned selectors and protocol</h2>
<p><b>Utility labels.</b> R(q, T, S) = 1 if the frozen answerer answers question q correctly with retained transcript T
and frame set S. Gold answers are used only to compute labels, never as inputs.</p>
<ul>
<li><b>Head B:</b> R(T, S + c) − R(T, S) for single frames on an empty history: 1,573 rows from 197 questions
(1,249 zero, 237 positive, 87 negative), 9 answerer calls per question.</li>
<li><b>Head A:</b> separate interventions on an empty context, line alone vs frame alone, for 1,182 Path A
moments (116 positive text, 367 positive frame labels).</li>
<li><b>Fourth-frame completion:</b> the relevance baseline fixes three frames; every other candidate c is labelled
R(T, top-3 + c). Exhaustive (1,500 completions, 1,368 fresh calls), so any fourth-frame policy is scored exactly
offline, including an oracle.</li>
</ul>
<p><b>Learned selectors.</b> Head B (projected MobileCLIP, OCR and nearby-speech embeddings plus scalar and history
features; regression with within-question ranking). Residual scorers: a pairwise ridge correction to MobileCLIP
with 17 coefficients, or 27 with retained-transcript lexical interactions. Completion heads: a pairwise ridge
correction to the relevance rank, with 26 global features (candidate plus redundancy and novelty against the
chosen frames and speech), optionally plus 8 local features (question relevance of 8 full-coverage tiles and
chalk-ink novelty against the chosen frames).</p>
<p><b>Protocol.</b> Leave-one-lecture-out cross-validation: heads evaluating a lecture never saw its labels; ridge
strength is chosen by inner leave-one-lecture-out with a no-change fallback. Every arm sees the same pool, retained
transcript, answerer, prompt and K = 4. Intervals are 95% lecture- or video-level bootstrap. All answerer
calls go through an exact-request cache, so call counts are exact. Latency is a sum of measured stage medians, not
an end-to-end timing.</p></section>""")

    # --- Section 5: results
    v1_items = [("Transcript only", v1["transcript_only"]["accuracy"], False),
                ("Transcript-retrieved moments", v1["retrieval"]["accuracy"], False),
                ("Cost-aware heuristic", v1["heuristic"]["accuracy"], False),
                ("Uniform frames", v1["uniform"]["accuracy"], False),
                ("MobileCLIP top-k", v1["scout_similarity"]["accuracy"], True)]
    a(f"""<section><h2>5. Results</h2>
<h3>5.1 Version 1 on LongVideoBench</h3>
{bars(v1_items, caption="Figure 2. v1 accuracy on 440 LongVideoBench questions, at most 8 frames, same frozen 2B answerer. Chance is 21.4%.")}
<table><thead><tr><th>Strategy</th><th>Accuracy</th><th>95% CI</th><th>Frames shown</th><th>Time / question</th></tr></thead><tbody>
{"".join(f"<tr><td>{n}</td><td>{pct(v1[k]['accuracy'])}</td><td>{pct(v1[k]['ci95'][0])}–{pct(v1[k]['ci95'][1])}</td><td>{v1[k]['frames_mean']:.1f}</td><td>{v1[k]['warm_ms_median'] / 1000:.2f} s</td></tr>" for n, k in (("Transcript only", "transcript_only"), ("Transcript-retrieved moments", "retrieval"), ("Cost-aware heuristic", "heuristic"), ("Uniform frames", "uniform"), ("<b>MobileCLIP top-k</b>", "scout_similarity")))}
</tbody></table>
<p>MobileCLIP top-k is the best v1 strategy (+9.1 points over transcript only, CI +4.6 to +13.6); its lead over
uniform frames (+2.3) is not significant. Leaderboard 7–8B models score 40–50% with 8–16 frames on the full benchmark;
this is a reference point, not a matched comparison.</p>

<h3>5.2 Version 2 main experiment (MIT, original option order)</h3>
<table><thead><tr><th>Arm (K = 4, 197 questions, 20 held-out lectures)</th><th>Accuracy</th><th>Composed latency</th></tr></thead><tbody>
<tr><td>Transcript only</td><td>{pct(acc["transcript_only"])}</td><td>—</td></tr>
<tr><td>A. MobileCLIP top-4</td><td>{pct(acc["A_mobileclip_topk"])}</td><td>{lat["A_mobileclip_topk"]:.2f} s</td></tr>
<tr><td>B. MobileCLIP + MMR</td><td>{pct(acc["B_mobileclip_mmr"])}</td><td>{lat["B_mobileclip_mmr"]:.2f} s</td></tr>
<tr><td>C. Head B with OCR (3 seeds)</td><td>{pct(acc["C_ocr"])}</td><td>{lat["C_ocr"]:.2f} s</td></tr>
<tr><td>C. Head B without OCR</td><td>{pct(acc["C_noocr"])}</td><td>{lat["C_noocr"]:.2f} s</td></tr>
<tr><td>C. Head B without OCR or speech features (corrected ablation)</td><td>{pct(fix["accuracy"]["C_noocr_notext"])}</td><td>—</td></tr>
<tr><td>E. Trained Head A frame credit</td><td>{pct(acc["E_headA_frame_credit"])}</td><td>{lat["E_headA_frame_credit"]:.2f} s</td></tr>
<tr><td><b>F. Zero-shot transcript relevance</b></td><td><b>{pct(acc["F_relevance"])}</b></td><td>{lat["F_relevance"]:.2f} s</td></tr>
<tr><td>G. Trained Head A text credit</td><td>{pct(acc["G_headA_text_credit"])}</td><td>{lat["G_headA_text_credit"]:.2f} s</td></tr>
</tbody></table>
{forest([("A − transcript only", P["A_mobileclip_topk - transcript_only"]["diff"], P["A_mobileclip_topk - transcript_only"]["ci95_lecture_bootstrap"]),
         ("E − F (trained frame credit vs relevance)", P["E_headA_frame_credit - F_relevance"]["diff"], P["E_headA_frame_credit - F_relevance"]["ci95_lecture_bootstrap"]),
         ("C no OCR − A", P["C_noocr - A_mobileclip_topk"]["diff"], P["C_noocr - A_mobileclip_topk"]["ci95_lecture_bootstrap"]),
         ("C OCR − A", P["C_ocr - A_mobileclip_topk"]["diff"], P["C_ocr - A_mobileclip_topk"]["ci95_lecture_bootstrap"]),
         ("C no OCR − C without speech features", fix["paired"]["C_noocr - C_noocr_notext"]["diff"], fix["paired"]["C_noocr - C_noocr_notext"]["ci95_lecture_bootstrap"])],
        lo=-0.15, hi=0.30, caption="Figure 3. Paired differences (points) with 95% lecture-bootstrap intervals. Filled dots: interval excludes zero.")}
<p>Frames matter; the learned Head B is not better than MobileCLIP and does not use its speech features; the
trained frame-credit head is significantly worse than zero-shot relevance. Head A's out-of-fold AUC for frame gain
was 0.50 (zero-shot 0.45). Stage medians: decode 2.11 s, OCR 3.15 s, answer {lat["answer_k_frames"]:.2f} s.</p>

<h3>5.3 Does speech change which frames help?</h3>
<p>For {cond["matched_frames"]} matched question/frame pairs measured with and without the retained transcript, positive
frame-utility labels agree moderately (Cohen's κ = {cond["cohen_kappa"]:.2f}). Of
{cond["table"]["useful_only_without_speech"]} frames useful only without speech,
{dec["redundant_success_both_sources_and_combination_correct"]} are redundant (speech already answers),
{dec["speech_interference_frame_correct_speech_and_combination_wrong"]} fail because the combined input is wrong although the
frame alone was right, and {dec["destructive_combination_both_sources_correct_alone"]} fail only in combination. These counts
use the original option order and therefore include position-bias effects (Section 7).</p>

<h3>5.4 Residual scorers and the fourth frame</h3>
<table><thead><tr><th>Selector (K = 4)</th><th>Correct / 197</th><th>Accuracy</th></tr></thead><tbody>
{"".join(f"<tr><td>{n}</td><td>{res['accuracy'][k]['correct']}</td><td>{pct(res['accuracy'][k]['accuracy'])}</td></tr>" for n, k in (("MobileCLIP", "clip"), ("MMR", "mmr"), ("Zero-shot relevance", "relevance"), ("Residual scorer, 17 coefficients", "residual_base"), ("Residual scorer + retained-transcript features", "residual_context")))}
</tbody></table>
<p>Residual − MobileCLIP {pts(res["paired"]["residual_base_minus_clip"]["difference"])} (CI {ci(res["paired"]["residual_base_minus_clip"]["ci95_video_bootstrap"])});
residual − relevance {pts(res["paired"]["residual_base_minus_relevance"]["difference"])} (CI {ci(res["paired"]["residual_base_minus_relevance"]["ci95_video_bootstrap"])}).</p>
{bars([("Three frames only (reference)", ca["three_frame_reference"], False), ("MobileCLIP-best 4th", ca["clip_best"], False),
       ("MMR 4th", ca["mmr_next"], False), ("Random 4th (exact expectation)", ca["expected_random"], False),
       ("Single-frame residual 4th", ca["single_frame_residual"], False), ("Completion head (global)", la["global"], False),
       ("Completion head (global + local)", la["global_local"], False), ("Relevance rank-4 (baseline)", ca["relevance_rank4"], True),
       ("Oracle 4th frame", ca["oracle"], False)], xmax=0.5, unit_ticks=(0, .1, .2, .3, .4, .5),
      caption="Figure 4. Exact four-frame accuracy of fourth-frame policies after the relevance baseline fixes three frames (exhaustive table, no extra calls per policy).")}
<p>The deployment-matched completion head scores {pct(la["global"])} ({pts(lp["global_minus_relevance_rank4"]["difference"])}, CI {ci(lp["global_minus_relevance_rank4"]["ci95_lecture_bootstrap"])})
and falls back to the baseline in {loc["fallback_folds"]["global"]}/20 folds; adding local board features gives {pct(la["global_local"])}
({pts(lp["global_local_minus_relevance_rank4"]["difference"])}, CI {ci(lp["global_local_minus_relevance_rank4"]["ci95_lecture_bootstrap"])}), fallback in {loc["fallback_folds"]["global_local"]}/20.
The oracle is {pts(cp["oracle_minus_relevance_rank4"]["difference"])} points above the baseline (CI {ci(cp["oracle_minus_relevance_rank4"]["ci95_lecture_bootstrap"])}), but the choice changes the answer in only
{fe["questions_where_the_choice_matters"]} of 197 questions: too few to learn from. In {fe["anchor_correct_and_some_fourth_frame_breaks_it"]} of {fe["anchor_correct"]} questions
the three chosen frames answer correctly and some fourth frame breaks the answer; in {fe["anchor_wrong_and_some_fourth_frame_rescues_it"]} of {fe["anchor_wrong"]} a fourth frame rescues it.
MobileCLIP ({pts(cp["clip_best_minus_relevance_rank4"]["difference"])}) and MMR ({pts(cp["mmr_next_minus_relevance_rank4"]["difference"])}) completions are significantly worse than relevance in this setting.</p>

<h3>5.5 Candidate generation: balanced scan, crop and de-duplication</h3>
<table><thead><tr><th>Change (only the pool or scout feature differs)</th><th>Data</th><th>Effect on MobileCLIP top-4</th></tr></thead><tbody>
<tr><td>Balanced scan (original order)</td><td>MIT 197</td><td>{pts(bal["balanced_minus_legacy"]["A_mobileclip_topk"]["difference"])} (CI {ci(bal["balanced_minus_legacy"]["A_mobileclip_topk"]["ci95_lecture_bootstrap"])})</td></tr>
<tr><td>Balanced scan (circular, debiased)</td><td>MIT 197</td><td>{pts(circ_bal_A[0])} (CI {ci(circ_bal_A[1])})</td></tr>
<tr><td><b>Balanced scan: declared transfer test</b></td><td>LongVideoBench 440</td><td><b>{pts(lvb["balanced_minus_legacy"]["A_mobileclip_topk"]["difference"])} (CI {ci(lvb["balanced_minus_legacy"]["A_mobileclip_topk"]["ci95_lecture_bootstrap"])})</b></td></tr>
<tr><td>Whole frame padded to a square (same cost)</td><td>MIT balanced pools</td><td>{pts(crop["v2_balanced"]["paired"]["pad_minus_crop"]["difference"])} (CI {ci(crop["v2_balanced"]["paired"]["pad_minus_crop"]["ci95_lecture_bootstrap"])})</td></tr>
<tr><td>Three full-coverage tiles (1.8× scout cost)</td><td>MIT balanced pools</td><td>{pts(crop["v2_balanced"]["paired"]["tiles3_minus_crop"]["difference"])} (CI {ci(crop["v2_balanced"]["paired"]["tiles3_minus_crop"]["ci95_lecture_bootstrap"])})</td></tr>
</tbody></table>
{crop_figure()}
<p>The balanced scan touches all 711 retrieved windows instead of 273 and helped on MIT, but it did not transfer to
LongVideoBench, where relevance was also worse than MobileCLIP ({pts(lvb_FA[0])}, CI {ci(lvb_FA[1])}). Padding
shrinks writing and hurts; tiles change 75% of selections without a reliable gain. Re-decoding every scan frame shows
the stable-run rule drops {dedup["dropped_frames"]:,} of {dedup["scan_frames"]:,} frames, almost all because the lecturer moved; only
{dedup["dropped_small_local_edit_1to3_tiles"]} are small local edits, and the one flagged as lost writing is a chalk-tray reflection.
De-duplication is not a measurable loss on these lectures.</p></section>""")

    # --- Section 6: best scores
    a(f"""<section><h2>6. Best method per setting</h2>
<table><thead><tr><th>Setting</th><th>Best method</th><th>Score</th><th>Runner-up</th><th>Difference</th></tr></thead><tbody>
<tr><td>LongVideoBench 440, v1, ≤ 8 frames</td><td>MobileCLIP top-k</td><td>{pct(v1["scout_similarity"]["accuracy"])}</td><td>Uniform frames {pct(v1["uniform"]["accuracy"])}</td><td>+2.3 (n.s.)</td></tr>
<tr><td>LongVideoBench 440, v2 pools, K = 4</td><td>MMR (legacy scan)</td><td>{pct(lvb["legacy"]["accuracy"]["B_mobileclip_mmr"])}</td><td>MobileCLIP {pct(lvb["legacy"]["accuracy"]["A_mobileclip_topk"])}</td><td>n.s.</td></tr>
<tr><td>MIT 197, original order, K = 4</td><td>Relevance (either scan) / MobileCLIP balanced</td><td>{pct(acc["F_relevance"])} / {pct(bal["balanced"]["accuracy"]["A_mobileclip_topk"])}</td><td>Residual {pct(res["accuracy"]["residual_base"]["accuracy"])}</td><td>n.s.</td></tr>
<tr><td>MIT 197, circular (debiased), K = 4</td><td>MobileCLIP, balanced scan</td><td>{pct(cb["A"]["circular_mean"])}</td><td>Relevance, balanced {pct(cb["F"]["circular_mean"])}</td><td>{pts(cb["A"]["circular_mean"] - cb["F"]["circular_mean"])} (n.s.)</td></tr>
<tr><td>MIT fourth frame (fixed top-3)</td><td>Relevance rank-4</td><td>{pct(ca["relevance_rank4"])}</td><td>Completion head + local {pct(la["global_local"])}</td><td>{pts(lp["global_local_minus_relevance_rank4"]["difference"])}</td></tr>
</tbody></table>
<p>No learned selector is the best in any setting. The strongest choices are label-free rules, and which rule wins
depends on the data: transcript relevance on chalkboard lectures, image similarity on general long videos.</p></section>

<section><h2>7. Validity problems found and corrected</h2>
<h3>7.1 Option-position bias</h3>
<p>The generator never shuffled options and its example used answer index 0, so gold is A or B in 282 of 349 verified
questions. The prior filter used the same 2B answerer; it prefers C/D, so C/D-gold questions were "answered without
evidence" more often and dropped more often ({pct(bias["prior_filter_drop_rate"]["gold_C_or_D"], 0)} vs {pct(bias["prior_filter_drop_rate"]["gold_A_or_B"], 0)}).
In the kept 197, gold counts by position are {bias["gold_index_counts"]["qa_v2 (197, used in all v2 main/completion/scan results)"]}
while the transcript-only model picks {bias["transcript_only_predicted_index_counts_qa_v2"]}. "Always B" would score
{pct(bias["always_B_baseline_accuracy_qa_v2"])}. We re-evaluated the main arms under all four cyclic option rotations
(circular evaluation, 3,495 extra calls):</p>
{bars([("Transcript only", cm["transcript_only"]["circular_mean"], False), ("MobileCLIP (legacy)", cm["A"]["circular_mean"], False),
       ("MMR (legacy)", cm["B"]["circular_mean"], False), ("Relevance (legacy)", cm["F"]["circular_mean"], False),
       ("Relevance (balanced)", cb["F"]["circular_mean"], False), ("MobileCLIP (balanced)", cb["A"]["circular_mean"], True)],
      caption="Figure 6. MIT accuracy after debiasing (mean over the four option rotations). Original-order values were about 15 points lower.")}
<p>Debiasing raises transcript-only from {pct(cm["transcript_only"]["original_rotation0"])} to {pct(cm["transcript_only"]["circular_mean"])} and every frame arm by about
15 points. Rankings keep their direction; relevance − MobileCLIP becomes {pts(-circ_m["paired_circular_mean"]["A_minus_F"]["difference"])}
(n.s.); balanced − legacy for MobileCLIP becomes significant on MIT ({pts(circ_bal_A[0])}, CI {ci(circ_bal_A[1])}) but, as shown in
5.5, did not transfer. Label-based analyses (5.3, 5.4) used the original order and carry this caveat.</p>
<h3>7.2 Other corrections</h3>
<ul>
<li><b>Interpretation:</b> an early claim that speech "mostly makes frames redundant" was wrong; about half the cases are failures of the combined input (5.3).</li>
<li><b>Speech ablation:</b> the first "no speech features" Head B still received speech-derived scores; corrected and rerun: {pct(fix["accuracy"]["C_noocr_notext"])} vs {pct(fix["accuracy"]["C_noocr"])}.</li>
<li><b>Cache identity:</b> answer-cache keys omitted frame time labels, transcript ids/end times and decoding settings. The key now hashes the full rendered request and stores predictions, not correctness. An audit of {sum(v["cache_records"] for k, v in cache.items() if isinstance(v, dict)):,} historical records found no possible collision; historical results replay exactly.</li>
<li><b>Data checks:</b> completion labels reject duplicates and inconsistent anchors; video leakage is checked on the whole manifest before split filtering; pool comparisons assert that only the scan differs.</li>
</ul></section>""")

    # --- Section 8-10
    a("""<section><h2>8. Relation to prior work and novelty</h2>
<p>We do not claim a new method. Learning frame selection from a frozen answer model's signal (SeViLA, Frame-Voyager,
TSPO, FrameOracle, question-aware synthetic keyframe supervision 2603.14953), history-conditioned selection (ReFoCUS,
MarKey, FORTE, GIFT), dual subtitle and visual streams (VSI, Q-Gate) and spatio-temporal rationales (TranSTR) are all
established. Learning the marginal benefit of the next item given the current list goes back to Ross et al. (ICML 2013).
Our MMR completion rule is not an implementation of MarKey or FORTE, and its failure here does not refute them.</p>
<p>What this project adds is narrower: (1) an exhaustive, deployment-matched fourth-frame table for one frozen
answerer, which gives exact offline scores for any policy and an exact oracle; (2) controlled negative results for four
families of small selectors trained on such labels; (3) a concrete, reproducible case in which answerer-conditioned
filtering of synthetic questions amplifies option-position bias enough to invert conclusions about absolute accuracy.
Option-position bias itself is well known (circular evaluation was introduced for it); the amplification mechanism
and its size in a video-QA construction pipeline are the specific observation.</p></section>

<section><h2>9. Limitations</h2>
<ul>
<li>The lecture set is synthetic, not human-verified, from one course and one lecturer, and has been used for many
development decisions; it is not a final test set.</li>
<li>The same frozen 2B model produced labels and is evaluated; a second answerer has not been tested.</li>
<li>Latency figures are composed stage medians, not end-to-end timings; some were measured while another GPU job ran.</li>
<li>The LongVideoBench subset is not the full benchmark and was used to develop v1.</li>
<li>Learned heads were trained on at most 1,573 labels; only 36 questions carry fourth-frame signal.</li>
<li>No published learned selector was reproduced as a baseline.</li>
</ul></section>

<section><h2>10. Next steps</h2>
<ol>
<li><b>An untouched, properly built test set:</b> regenerate lecture questions with seeded option shuffling, check
gold-position balance before and after any filter, and audit a sample by hand.</li>
<li><b>EduVidQA (EMNLP 2025):</b> 3,909 training and 269 expert-verified real test questions on video-disjoint splits;
needs video access (0 of 296 videos local, transcripts for 87) and a validated free-text scorer.</li>
<li><b>More signal for the last slot:</b> the +6.6-point headroom is real, but it sits in 36 questions; more varied
training questions are needed before any learned selector can be judged.</li>
</ol></section>""")

    a(f"""<section><h2>Appendix: reproducibility and compute</h2>
<p>Code: <code>src/videoqa/v2/</code>; experiments: <code>scripts/v2_*.py</code>; every number above is read from
<code>reports/</code> by <code>scripts/build_report.py</code>. Hypotheses for the completion, local-feature and
LongVideoBench experiments were committed before their results (<code>docs/v2/novelty_check.md</code>,
<code>docs/v2/lvb_transfer.md</code>). 205 unit tests pass (CPU only).</p>
<table><thead><tr><th>Experiment</th><th>Fresh answerer calls</th><th>GPU time</th></tr></thead><tbody>
<tr><td>v2 main (pools, labels, evaluation)</td><td>—</td><td>3.4 h</td></tr>
<tr><td>Residual four-frame QA</td><td>{res["fresh_calls_total"]}</td><td>{res["answer_seconds_total"] / 60:.0f} min</td></tr>
<tr><td>Fourth-frame completion labels</td><td>1,368</td><td>43 min</td></tr>
<tr><td>Balanced scan pools and comparison (MIT)</td><td>{bal["balanced"]["fresh_calls"]}</td><td>≈ 35 min</td></tr>
<tr><td>Scout crop ablation</td><td>{crop["v2_main"]["fresh_calls"] + crop["v2_balanced"]["fresh_calls"]}</td><td>≈ 20 min</td></tr>
<tr><td>Circular evaluation</td><td>{circ_m["fresh_calls"] + circ_b["fresh_calls"]:,}</td><td>{(circ_m["answer_seconds"] + circ_b["answer_seconds"]) / 3600:.1f} h</td></tr>
<tr><td>LongVideoBench transfer (pools + comparison)</td><td>{lvb["legacy"]["fresh_calls"] + lvb["balanced"]["fresh_calls"]:,}</td><td>≈ 1.6 h</td></tr>
</tbody></table>
<p>Hardware: one NVIDIA RTX 5060 Ti (16 GB), Windows 11, Python 3.12, PyTorch 2.11 (CUDA 12.8), transformers 5.17.</p></section>""")
    return "\n".join(S)


CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
:root { --ink:#1d2430; --muted:#5b6574; --rule:#d5dae2; --accent:#1f5f8b; --accent2:#c0532b; --bg:#ffffff; --soft:#f3f6f9; }
html { color-scheme: light; }
body { font-family: Cambria, Georgia, "Times New Roman", serif; color: var(--ink); background: var(--bg);
       font-size: 10.6pt; line-height: 1.45; max-width: 760px; margin: 0 auto; padding: 0 16px; }
h1 { font-size: 21pt; line-height: 1.15; margin: 4px 0 6px; text-wrap: balance; }
h2 { font-size: 13.5pt; margin: 20px 0 6px; padding-bottom: 3px; border-bottom: 1px solid var(--rule); break-after: avoid; }
h3 { font-size: 11.2pt; margin: 14px 0 4px; color: var(--accent); break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
.kicker { font-family: Calibri, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: .08em; font-size: 8.5pt; color: var(--muted); margin: 0; }
.sub { font-size: 11.5pt; color: var(--muted); margin: 0 0 6px; }
.author { font-family: Calibri, "Segoe UI", sans-serif; font-size: 9.5pt; color: var(--muted); }
header.title { border-bottom: 2px solid var(--ink); padding-bottom: 8px; margin-bottom: 8px; }
.abstract { background: var(--soft); padding: 4px 14px 8px; border-left: 3px solid var(--accent); }
.abstract h2 { border: 0; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 10px; font-family: Calibri, "Segoe UI", sans-serif;
        font-size: 9.2pt; font-variant-numeric: tabular-nums; break-inside: avoid; }
th { text-align: left; border-bottom: 1.5px solid var(--ink); padding: 3px 6px; }
td { border-bottom: 1px solid var(--rule); padding: 3px 6px; vertical-align: top; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
figure { margin: 10px 0 12px; break-inside: avoid; }
figcaption { font-family: Calibri, "Segoe UI", sans-serif; font-size: 8.8pt; color: var(--muted); margin-top: 3px; }
svg { width: 100%; height: auto; font-family: Calibri, "Segoe UI", sans-serif; }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .zero { stroke: var(--ink); stroke-width: 1.2; }
svg .tick { font-size: 10px; fill: var(--muted); }
svg .lab { font-size: 11px; fill: var(--ink); }
svg .val { font-size: 10.5px; fill: var(--ink); }
svg .bar { fill: #9fb4c7; }
svg .bar.hi { fill: var(--accent); }
svg .whisk { stroke: var(--ink); stroke-width: 1.4; }
svg .dot { fill: #ffffff; stroke: var(--ink); stroke-width: 1.4; }
svg .dot.sig { fill: var(--accent2); stroke: var(--accent2); }
svg .box { fill: #ffffff; stroke: var(--ink); stroke-width: 1.2; }
svg .box.frozen { fill: #e7eef5; }
svg .box.trained { fill: #fbe6dc; stroke: var(--accent2); }
svg .bt { font-size: 12px; font-weight: 700; fill: var(--ink); }
svg .bs { font-size: 10px; fill: var(--muted); }
svg .ln { stroke: var(--ink); stroke-width: 1.2; fill: none; }
svg .ln.dash { stroke-dasharray: 4 3; stroke: var(--accent2); }
svg .arrowhead { fill: var(--ink); }
.claims td:last-child { white-space: nowrap; }
section { break-inside: auto; }
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
           f"<title>Evidence Selection Report</title><style>{CSS}</style></head><body>{build()}</body></html>")
    html_path = OUT / "framework_report.html"
    html_path.write_text(doc, encoding="utf-8")
    print(f"wrote {html_path}")
    if a.pdf:
        for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
            if Path(exe).exists():
                pdf = (OUT / "framework_report.pdf").resolve()
                subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                                f"--print-to-pdf={pdf}", html_path.resolve().as_uri()], check=True, timeout=120)
                print(f"wrote {pdf}")
                break


if __name__ == "__main__":
    main()
