"""Build the framework guide (HTML -> PDF): the flow of the final framework step by step, with the file and
function that does each step, how experiments are run and evaluated, and where every part of the code lives.

    python scripts/build_framework_guide.py      # docs/report/framework_guide.{html,pdf} (generated, not tracked)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

OUT = Path("docs/report")

CSS = """
@page { size: A4; margin: 14mm 13mm; }
:root { --ink:#1b2330; --muted:#5a6575; --line:#d3d9e1; --accent:#0e5f74; --soft:#eaf3f5; --head:#94500f; --headbg:#fcf0e1; --box:#f4f6f9; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.42; margin: 0; background: #fff; }
h1 { font-family: Georgia, serif; font-size: 20pt; margin: 0 0 3px; }
h2 { font-family: Georgia, serif; font-size: 13.5pt; color: var(--accent); border-bottom: 1.5px solid var(--accent);
     padding-bottom: 2px; margin: 16px 0 6px; break-after: avoid; }
h3 { font-size: 10.3pt; margin: 9px 0 3px; break-after: avoid; }
p { margin: 3px 0 6px; }
.sub { color: var(--muted); margin-bottom: 8px; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #eef1f5; padding: 0 3px; border-radius: 3px; }
.flow { display: flex; flex-direction: column; gap: 0; margin: 6px 0; }
.st { border: 1.3px solid var(--line); background: var(--box); border-radius: 5px; padding: 5px 9px; break-inside: avoid; }
.st.h { background: var(--headbg); border: 2px solid var(--head); }
.st.io { background: var(--soft); border-color: var(--accent); }
.st b { font-size: 9.8pt; } .st .f { color: var(--accent); font-family: Consolas, monospace; font-size: 8.3pt; display: block; }
.ar { text-align: center; color: var(--accent); font-size: 12pt; line-height: 1; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4px 0 8px; break-inside: avoid; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #e7ecf1; }
td:first-child { white-space: nowrap; }
.box { border-left: 4px solid var(--accent); background: var(--soft); padding: 6px 9px; margin: 6px 0; break-inside: avoid; }
.pb { break-before: page; }
"""


def st(title, text, files, kind=""):
    return (f'<div class="st {kind}"><b>{title}</b><br>{text}'
            + "".join(f'<span class="f">{f}</span>' for f in files) + "</div>")


AR = '<div class="ar">&#9660;</div>'


def build() -> str:
    P = ["""<h1>Framework Guide: Long-Video and Lecture Question Answering</h1>
<div class="sub">local-surf, branch <code>framework-2</code> · last commit 8c439e4 (2026-10-25, research log step 66) ·
best version tagged <code>v3-best-speech-4b</code> · Sayed Ilham Azhar Harun</div>
<p><b>What it does.</b> Answers multiple-choice questions about long videos and university lectures on one 16 GB GPU.
A small vision-language model can only look at a few pictures, so the framework finds the right moments (from the
speech and from the pictures), chooses 4 frames, selects the speech that matters, and asks a frozen answer model.</p>
<h2>1 · The final framework, step by step (with the file that does each step)</h2>
<div class="flow">"""]
    P.append(st("Step 0a · Transcript (once per video)",
                "Use the video's subtitles; if there are none, Whisper large-v3-turbo transcribes the audio with word "
                "timestamps, grouped into timed lines and cleaned (times kept in order).",
                ["scripts/research/asr_transcribe.py  (load_audio, group_words, sanitize)",
                 "src/videoqa/transcript.py  (parse_srt_or_vtt, parse_json_segments, load_transcript)"], "io"))
    P.append(AR)
    P.append(st("Step 0b · Picture index (once per video)",
                "Decode one picture every 2 s with PyAV (max 640 px) and turn each into 512 numbers with MobileCLIP-S2; "
                "saved as data/&lt;dataset&gt;/features/&lt;video&gt;.npz (times, embeddings). About 30–45 s per 30-min video.",
                ["scripts/v3_features.py  (video_features)", "src/videoqa/frames.py  (probe, decode_at)",
                 "src/videoqa/v2/encoders.py  (FrozenEncoders: MobileCLIP image + text embeddings)"], "io"))
    P.append(AR)
    P.append(st("Step 1 · Transcript search (BM25)",
                "Cut the transcript into ~20 s units, score each with BM25 against question + options, keep the 4 best "
                "units widened by one neighbour each side = 4 windows of about 60 s.",
                ["src/videoqa/transcript.py  (build_units)", "src/videoqa/retrieval.py  (tokenize, bm25_scores, bm25_rank, build_windows)"]))
    P.append(AR)
    P.append('<div class="row">')
    P.append(st("Step 2a · Path A, speech path",
                "<b>Speech rule</b> (best framework): MiniLM cross-encoder scores each line inside the windows with its "
                "neighbour lines; best 6 lines → picture 0.3 s before each line ends. <b>Head A</b> (trained, optional): scores "
                "every line of the whole transcript with MiniLM + BGE + 22 simple facts and a 1,889-parameter network; nearby "
                "lines vote; 6 peaks → 6 pictures.",
                ["src/videoqa/v2/head_a.py  (HotMomentScorer: MiniLM)", "src/videoqa/v3/head_a.py  (LearnedMoments, line_features, density_moments)",
                 "src/videoqa/v2/candidates.py  (build_pool, Path A part)"], "h"))
    P.append(st("Step 2b · Path B, picture path (hybrid)",
                "3 pictures from a 5 s scan inside the windows + 3 from the dense whole-video index (question vs all saved "
                "pictures, top 24 times ≥ 4 s apart); repeats removed by perceptual hash; ranked by MobileCLIP similarity.",
                ["src/videoqa/v2/candidates.py  (scan_timestamps, _feature_scan_times, _stable_representatives)",
                 "src/videoqa/frames.py  (decode_at, average_hash)"]))
    P.append("</div>")
    P.append(AR)
    P.append(st("Step 3 · Merge into the candidate pool",
                "All needed pictures are decoded in one pass; Path A pictures first, then Path B; a picture within 1 s or "
                "almost identical to one already in is merged (marked 'both'). ~10–12 candidates, saved per question.",
                ["src/videoqa/v2/candidates.py  (build_pool, save_pool, load_pool)", "src/videoqa/v2/records.py  (FrameCandidate, QuestionPool)"]))
    P.append(AR)
    P.append(st("Step 4 · Choose the final 4 pictures",
                "<b>MMR rule</b> (best framework): pick one at a time by 0.7 × match to question + options − 0.3 × similarity "
                "to pictures already chosen. <b>Head B</b> (trained, optional): 18 facts per candidate, small scorer, set choice "
                "that separates the answer options.",
                ["scripts/v3_baseline_cycle.py  (select: rule 'mmropt')", "scripts/v3_head_b.py  (pool_features, train_scorer, greedy, select)"], "h"))
    P.append(AR)
    P.append(st("Step 5 · Speech text for the answer model (up to 600 words)",
                "Speech within 15 s of the 4 chosen pictures (nearest first, 300 words) + the best lines of the whole video "
                "by BM25 + MiniLM rank fusion, each with neighbours; in time order with timestamps.",
                ["scripts/v3_baseline_cycle.py  (transcript_for, policy 'frames+hybrid')", "src/videoqa/packing.py  (format_excerpt)"]))
    P.append(AR)
    P.append(st("Step 6 · Frozen answer model",
                "Prompt: instruction, 4 pictures each labelled 'Frame at Xs:', question, lettered options, transcript excerpt, "
                "'Reply with the option letter first'. Qwen3-VL-2B or 4B locally, or Nemotron-120B through the NVIDIA API.",
                ["src/videoqa/answerer.py  (SYSTEM_PROMPT, build_prompt_text, make_answerer)", "src/videoqa/answerer_hf.py  (HFVLMAnswerer, local)",
                 "src/videoqa/answerer_api.py  (APIVLMAnswerer, NVIDIA API)", "configs/gpu_12gb.yaml · configs/gpu_16gb_4b.yaml · configs/api_nemotron_vl.yaml"], "io"))
    P.append(AR)
    P.append(st("Step 7 · Read and score the answer",
                "First option letter is the choice (forced from the model's probabilities if no letter; 'insufficient "
                "evidence' counts as wrong); compared with the answer key.",
                ["src/videoqa/answerer.py  (parse_option_letter, finalize_answer)"], "io"))
    P.append("</div>")

    P.append('<h2 class="pb">2 · How an experiment is run</h2>')
    P.append("""<table><tr><th>Stage</th><th>Command</th><th>What it does</th></tr>
<tr><td>Data</td><td><code>scripts/research/fetch_*.py</code>, <code>prepare_cgbench.py</code>, <code>fetch_lvb_fresh.py</code></td><td>download videos, subtitles, questions; fixed video-level train / dev / test split</td></tr>
<tr><td>Loading</td><td><code>src/videoqa/datasets.py</code> (load_local_dataset)</td><td>reads qa.jsonl + videos + transcripts; refuses leakage between splits</td></tr>
<tr><td>Pools</td><td><code>scripts/v3_baseline_cycle.py pools --dataset D --pool hybrid</code></td><td>Steps 1–3 for every question, saved to runs/v3_baseline/D/hybrid/pools</td></tr>
<tr><td>Answers</td><td><code>scripts/v3_baseline_cycle.py answer --dataset D --pool hybrid --selectors mmropt --words 600 --tpolicy frames+hybrid --config ...</code></td><td>Steps 4–7; every answer cached by request key (runs/v3_baseline/answers.jsonl), each run on a call budget (src/videoqa/v2/budget.py)</td></tr>
<tr><td>Report</td><td><code>scripts/v3_baseline_cycle.py report</code>, <code>scripts/v3_final_report.py</code>, <code>scripts/build_results_table.py</code></td><td>accuracy per system and set; paired bootstrap 95% intervals over videos; the all-results table</td></tr>
<tr><td>Declared runs</td><td><code>scripts/run_final_test.sh</code>, <code>run_full_grid.sh</code>, <code>run_confirmation.sh</code></td><td>exact command sequence of each declared experiment</td></tr></table>""")
    P.append("<h2>3 · Training the heads</h2>")
    P.append("""<table><tr><th>Part</th><th>Files</th><th>How</th></tr>
<tr><td>Head A data</td><td><code>scripts/v3_head_a_data.py</code></td><td>877 CG-Bench training questions: transcript lines + human-marked answer seconds (data/head_a/*.jsonl)</td></tr>
<tr><td>Head A training</td><td><code>scripts/v3_head_a_light.py</code> (features, train), <code>src/videoqa/v3/head_a.py</code></td><td>supervised, multiple-instance listwise loss (a line near the marked seconds should rank first); 3 seeds averaged; stop on held-out training videos</td></tr>
<tr><td>Head A evaluation</td><td><code>scripts/v3_head_a_eval.py</code></td><td>share of new questions where one of 6 moments is inside the marked seconds (36.6% vs 28.0%)</td></tr>
<tr><td>Head B</td><td><code>scripts/v3_head_b.py</code>, <code>scripts/v3_head_b_labels.py</code></td><td>supervised on 373 CG-Bench training pools (human seconds; answer-model feedback version); set choice by greedy selection</td></tr></table>""")
    P.append("<h2>4 · Tests and quality checks</h2>")
    P.append("""<p><code>uv run pytest -q</code> runs 224 unit tests in <code>tests/</code>, for example: <code>test_transcript.py</code> (subtitle
parsing, units), <code>test_retrieval.py</code> (BM25, windows), <code>test_frames.py</code> (real-mp4 decoding), <code>test_answerer.py</code>
(letter parsing, citations), <code>test_acquisition.py</code> (each policy decodes only what it needs), <code>test_datasets.py</code>
(no leakage between splits), <code>test_v2_cache_identity.py</code> (cache keys, budgets), <code>test_v3_head_a_data.py</code> (time-reference
removal, dense-index scan, judge parsing). Experiments are declared before running (docs/v3/*_protocol.md) and logged in
docs/v3/research_log.md.</p>""")
    P.append("<h2>5 · Where everything is</h2>")
    P.append("""<table><tr><th>Folder / file</th><th>Contents</th></tr>
<tr><td><code>src/videoqa/</code></td><td>core: transcript.py, retrieval.py (BM25), frames.py (PyAV decoding), packing.py, answerer*.py, datasets.py, config.py, schemas.py; v1 modules (controller.py, scout.py, acquisition.py, pipeline.py, cli.py)</td></tr>
<tr><td><code>src/videoqa/v2/</code></td><td>candidate pools (candidates.py), MobileCLIP (encoders.py), MiniLM speech scorer (head_a.py), selectors, answer cache (teacher.py), budgets</td></tr>
<tr><td><code>src/videoqa/v3/</code></td><td>Head A inference (head_a.py), open-ended answers and judges (openended.py, factqa.py, judge.py), metrics, language filter</td></tr>
<tr><td><code>scripts/</code></td><td>experiment runners (v3_*.py), declared run sequences (run_*.sh), report builders (build_*.py)</td></tr>
<tr><td><code>scripts/research/</code></td><td>downloads, Whisper transcription, dataset preparation</td></tr>
<tr><td><code>configs/</code></td><td>answer-model profiles (2B, 4B, Nemotron API) and measured costs</td></tr>
<tr><td><code>docs/</code></td><td>EXPERIMENTS.md (index), v3/research_log.md (every step), protocols, results; history/ (v1, v2)</td></tr>
<tr><td><code>reports/</code></td><td>result summaries per experiment (JSON / Markdown)</td></tr>
<tr><td><code>data/ runs/ cache/</code></td><td>datasets, run outputs and model caches (local only, not in git)</td></tr></table>""")
    P.append("""<h2>6 · Results to remember</h2><div class="box">
Locked test, 459 questions (CG-Bench + Video-MMMU): best framework + speech text with Qwen3-VL-4B <b>55.6%</b> vs
50.3% for the starting design (+5.2, confirmed); lecture screen-reading 67.8% (+6.9, confirmed); long videos with 2B
50.9% vs 34.5% (+16.4, confirmed). Head A finds the answer moment 36.6% vs 28.0% (confirmed) but does not raise
accuracy reliably; on 894 fresh LongVideoBench questions with 4B, Head A + Head B −0.2 (not supported), best
framework +7.6 over the starting design (confirmed) and equal to dense MobileCLIP top-4. Latency with 2B about 4–7 s
per question (+30–45 s one-time index per 30-min video).</div>""")
    return "\n".join(P)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    html = OUT / "framework_guide.html"
    html.write_text(f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Framework Guide</title><style>{CSS}</style></head>"
                    f"<body>{build()}</body></html>", encoding="utf-8")
    exe = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if exe.exists():
        subprocess.run([str(exe), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={(OUT / 'framework_guide.pdf').resolve()}", html.resolve().as_uri()], check=True, timeout=120)
    print(f"wrote {OUT / 'framework_guide.pdf'}")


if __name__ == "__main__":
    main()
