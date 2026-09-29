"""Turn a clean-only `videoqa evaluate` run into reports/benchmark_v1/.

Reads   <run>/records.jsonl (one row per question x policy) and the dataset's
        qa.jsonl (for the number of options, i.e. chance level).
Writes  <out>/RESULTS.md     human-readable results + published reference points
        <out>/results.json   the same numbers, machine-readable
        <out>/records.jsonl  every prediction (question, policy, answer, cost)

Accuracy CIs are 95% bootstrap intervals resampling VIDEOS (questions from one
video are not independent). Video-length groups follow LongVideoBench's own
duration buckets.

    python scripts/benchmark_report.py --run runs/bench_lvb_v1 --data data/longvideobench_full --out reports/benchmark_v1
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.frames import probe  # noqa: E402

POLICY_LABEL = {
    "transcript_only": "Transcript only (no frames)",
    "uniform": "Uniform frames + transcript",
    "retrieval": "Transcript-retrieved moments",
    "scout_similarity": "MobileCLIP top-k frames",
    "heuristic": "Cost-aware heuristic controller",
}
BUCKETS = [("8–15 s", 8, 15), ("15–60 s", 15, 60), ("3–10 min", 180, 600), ("15–60 min", 900, 3600)]

# Published LongVideoBench validation-set results (official leaderboard,
# https://longvideobench.github.io/, read 2026-09-29). Full 1,337-question val
# set, each model's own frame count and protocol: reference points only.
REFERENCE = [
    ("GPT-4o (0513)", "proprietary", 256, 66.7),
    ("Gemini-1.5-Pro (0514)", "proprietary", 256, 64.0),
    ("LLaVA-Video-7B-Qwen2", "7B", 128, 61.1),
    ("GPT-4o-mini", "proprietary", 250, 56.5),
    ("Idefics2", "8B", 16, 49.7),
    ("Phi-3-Vision-Instruct", "4.2B", 16, 49.6),
    ("LLaVA-Next-Mistral-7B", "7B", 8, 49.1),
    ("LLaVA-1.5-7B", "7B", 8, 40.3),
]


def boot_ci(values_by_video: dict[str, list[float]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vids = list(values_by_video)
    sums = np.array([sum(values_by_video[v]) for v in vids])
    cnts = np.array([len(values_by_video[v]) for v in vids])
    idx = rng.integers(0, len(vids), size=(n_boot, len(vids)))
    acc = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bench_lvb_v1")
    ap.add_argument("--data", default="data/longvideobench_full")
    ap.add_argument("--out", default="reports/benchmark_v1")
    a = ap.parse_args()
    run, data, out = Path(a.run), Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(x) for x in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    recs = [r for r in recs if r["condition"] == "clean"]
    qa = {q["qa_id"]: q for q in map(json.loads, (data / "qa.jsonl").read_text(encoding="utf-8").splitlines()) if q}
    chance = float(np.mean([1 / len(qa[r["qa_id"]]["options"]) for r in recs if r["policy"] == recs[0]["policy"]]))

    dur_path = out / "video_durations.json"
    durations = json.loads(dur_path.read_text(encoding="utf-8")) if dur_path.exists() else {}
    for vid in {r["video_id"] for r in recs} - durations.keys():
        vp = next(p for p in (data / "videos").glob(f"{vid}.*"))
        durations[vid] = round(probe(vp).duration_s, 2)
    dur_path.write_text(json.dumps(durations, indent=1, sort_keys=True), encoding="utf-8")

    def bucket(vid):
        d = durations[vid]
        return next((name for name, lo, hi in BUCKETS if lo <= d <= hi), "other")

    policies = [p for p in POLICY_LABEL if any(r["policy"] == p for r in recs)]
    res = {}
    for p in policies:
        rs = [r for r in recs if r["policy"] == p]
        by_vid = collections.defaultdict(list)
        for r in rs:
            by_vid[r["video_id"]].append(r["quality"])
        lo, hi = boot_ci(by_vid)
        by_b = collections.defaultdict(list)
        for r in rs:
            by_b[bucket(r["video_id"])].append(r["quality"])
        res[p] = {
            "n": len(rs), "videos": len(by_vid), "accuracy": float(np.mean([r["quality"] for r in rs])),
            "ci95": [lo, hi], "frames_mean": float(np.mean([r["frames"] for r in rs])),
            "decoded_frames_mean": float(np.mean([r["decoded_frames"] for r in rs])),
            "scouted_frames_mean": float(np.mean([r["scouted_frames"] for r in rs])),
            "warm_ms_mean": float(np.mean([r["warm_ms"] for r in rs])),
            "warm_ms_median": float(np.median([r["warm_ms"] for r in rs])),
            "no_option_rate": float(np.mean([r["option"] is None for r in rs])),
            "by_length": {b: {"n": len(v), "accuracy": float(np.mean(v))} for b, v in by_b.items()},
        }
    clean = sum(bool(re.match(r"^\(?[A-H]\)?[.:),\s]", r["answer"].strip())) for r in recs)
    best = max(res, key=lambda p: res[p]["accuracy"])
    summary = {"dataset": "LongVideoBench (validation subset held locally)", "questions": res[policies[0]]["n"],
               "videos": res[policies[0]]["videos"], "chance_accuracy": chance, "policies": res,
               "best_policy": best, "reference_leaderboard": REFERENCE,
               "answers_with_clean_leading_letter": [clean, len(recs)]}
    shutil.copy(run / "records.jsonl", out / "records.jsonl")

    L = []
    L.append("# Benchmark results: LongVideoBench (v1 framework)\n")
    L.append(f"{summary['questions']} multiple-choice questions over {summary['videos']} videos from the "
             "LongVideoBench validation set, each answered on its original subtitles by a **2B-parameter** "
             "vision-language model (Qwen3-VL-2B-Instruct) running locally on one RTX 5060 Ti (16 GB), "
             f"with **at most 8 frames** per question. Chance level: {100 * chance:.1f}%.\n")
    L.append("## How answers are scored\n")
    L.append("Each question has 4–5 options and one correct option marked by the dataset's annotators. The model "
             "answers in free text with the option letter first; the letter is extracted and compared with the "
             "correct one (1 = match, 0 = otherwise), the leaderboard's own exact-match protocol. An "
             "\"insufficient evidence\" reply counts as wrong. Parsing check: "
             f"{clean} of {len(recs)} answers begin with a clean option letter.\n")
    L.append("## Frame-selection strategies (same model, same questions)\n")
    L.append("| strategy | accuracy | 95% CI (by video) | frames shown to the model | frames decoded | time per question (warm, median) |")
    L.append("|---|---|---|---|---|---|")
    for p in policies:
        r = res[p]
        L.append(f"| {POLICY_LABEL[p]} | **{100 * r['accuracy']:.1f}%** | {100 * r['ci95'][0]:.1f}–{100 * r['ci95'][1]:.1f}% "
                 f"| {r['frames_mean']:.1f} | {r['decoded_frames_mean']:.0f} | {r['warm_ms_median'] / 1000:.2f} s |")
    L.append("")
    L.append("## Accuracy by video length\n")
    L.append("| strategy | " + " | ".join(b for b, _, _ in BUCKETS) + " |")
    L.append("|---|" + "---|" * len(BUCKETS))
    for p in policies:
        cells = []
        for b, _, _ in BUCKETS:
            x = res[p]["by_length"].get(b)
            cells.append(f"{100 * x['accuracy']:.1f}% (n={x['n']})" if x else "–")
        L.append(f"| {POLICY_LABEL[p]} | " + " | ".join(cells) + " |")
    L.append("")
    # Paired differences on the same questions (bootstrap resampling videos),
    # written by `videoqa evaluate` into <run>/summary.json.
    comps = json.loads((run / "summary.json").read_text(encoding="utf-8")).get("comparisons", {})

    def paired(pa, pb):
        c = comps.get(f"{pa} - {pb} | clean | quality")
        if c:
            return c["diff"], c["ci95_low"], c["ci95_high"]
        c = comps.get(f"{pb} - {pa} | clean | quality")
        return (-c["diff"], -c["ci95_high"], -c["ci95_low"]) if c else None

    pairs = [(best, p) for p in policies if p != best] + [
        ("uniform", "transcript_only"), ("retrieval", "transcript_only"), ("uniform", "retrieval")]
    L.append("## Paired comparisons (same 440 questions)\n")
    L.append("Difference in accuracy between two strategies on the *same* questions, with a 95% bootstrap "
             "interval resampling videos. ✔ = the interval excludes zero.\n")
    L.append("| comparison | difference | 95% CI | |")
    L.append("|---|---|---|---|")
    summary["paired"] = {}
    for pa, pb in pairs:
        d = paired(pa, pb)
        if d is None or pa == pb:
            continue
        sig = d[1] > 0 or d[2] < 0
        summary["paired"][f"{pa} - {pb}"] = {"diff": d[0], "ci95": [d[1], d[2]], "significant": sig}
        L.append(f"| {POLICY_LABEL[pa]} vs {POLICY_LABEL[pb]} | {100 * d[0]:+.1f} pts "
                 f"| {100 * d[1]:+.1f} to {100 * d[2]:+.1f} | {'✔' if sig else '✗ not significant'} |")
    L.append("")

    t = {p: res[p]["warm_ms_median"] / 1000 for p in policies}
    acc = {p: 100 * res[p]["accuracy"] for p in policies}
    L.append("## Accuracy vs time\n")
    L.append("Measured per-action costs on this GPU (`configs/cost_model_rtx5060ti.yaml`): decoding ≈ 43 ms "
             "per frame read, MobileCLIP scoring ≈ 47 ms per frame, the answer model reading images ≈ 0.36 ms "
             "per image token (≈ 80 ms per 640-px frame). Video can only be decoded forward from a keyframe, "
             "so reaching one frame often means reading many (the *frames decoded* column).\n")
    L.append("| strategy | accuracy | time per question | vs transcript only |")
    L.append("|---|---|---|---|")
    for p in sorted(policies, key=lambda p: t[p]):
        note = ("baseline" if p == "transcript_only"
                else f"{acc[p] - acc['transcript_only']:+.1f} pts for {t[p] - t['transcript_only']:+.2f} s")
        L.append(f"| {POLICY_LABEL[p]} | {acc[p]:.1f}% | {t[p]:.2f} s | {note} |")
    L.append("")
    if "uniform" in res and "scout_similarity" in res:
        L.append(f"Uniform frames give most of the gain over text for almost no time "
                 f"({acc['uniform'] - acc['transcript_only']:+.1f} pts for "
                 f"{t['uniform'] - t['transcript_only']:+.2f} s). MobileCLIP top-k is the most accurate, but it "
                 f"decodes and scores every candidate (~{res['scout_similarity']['decoded_frames_mean']:.0f} frames "
                 f"read), which costs about {t['scout_similarity'] - t['uniform']:.1f} s more than uniform for a "
                 f"{acc['scout_similarity'] - acc['uniform']:+.1f}-point difference that is not significant. "
                 "Uniform is the speed choice and MobileCLIP the accuracy choice; the framework measures both "
                 "so the choice is explicit.\n")

    L.append("## Findings\n")
    L.append("1. **Looking at frames helps.** Uniform frames and MobileCLIP top-k both beat transcript-only "
             "(see the paired table), MobileCLIP by about 9 points. Transcript-retrieved frames did not clearly "
             "beat transcript-only.")
    L.append("2. **Choosing frames by image content beat choosing them by subtitle match.** BM25 usually finds "
             "the right ~60 s transcript window, but the retrieval strategy samples only its start, middle and "
             "end, so its frames can be tens of seconds from the moment the question refers to. MobileCLIP "
             "compares frame images with the question and can pick the exact frame. (Likely explanation, "
             "consistent with the numbers; not separately tested.)")
    L.append("3. **But image-based selection was not clearly better than simply spreading frames evenly**, and "
             "it costs about a second more per question.")
    L.append("4. **The hand-tuned cost-aware heuristic was worse than MobileCLIP top-k** and no better than "
             "uniform: a controller has to be learned or validated, not hand-set.")
    L.append("5. **The longest videos (15–60 min) are the hardest for every strategy**, most likely because 8 "
             "frames cover an hour thinly.\n")
    L.append("**Why the absolute accuracy is modest.** The answer model has 2B parameters and sees at most 8 "
             "frames at 640 px, on one 16 GB consumer GPU. LongVideoBench rewards many frames of long videos; "
             "the leaderboard leaders use 7B–72B models or GPT-4o with 128–256 frames. The ceiling most likely "
             "comes from the model size and frame budget, which are configuration settings; a larger model and "
             "more frames have not been benchmarked yet.\n")
    L.append("**Not measured here:** the answer model given the raw video with its own default frame sampling "
             "(no pipeline), open-ended answers, and the correctness of the model's timestamp citations.\n")

    L.append("## Published reference points (official leaderboard)\n")
    L.append("| model | size | frames | LongVideoBench val accuracy |")
    L.append("|---|---|---|---|")
    rows = [(n, s, f, v) for n, s, f, v in REFERENCE]
    rows.append((f"**This framework — {POLICY_LABEL[best]}**", "2B", "≤ 8", 100 * res[best]["accuracy"]))
    for n, s, f, v in sorted(rows, key=lambda x: -x[3]):
        L.append(f"| {n} | {s} | {f} | {v:.1f}% |")
    L.append("")
    L.append("**How to read this comparison.** The leaderboard numbers are the models' own published results on "
             "the full 1,337-question validation set, each with its own protocol and frame count (8–256 frames). "
             "This framework was run on the subset of validation questions whose videos are held locally, with a "
             "much smaller model and far fewer frames, so the rows are reference points, not a controlled "
             "head-to-head. The controlled comparison is the strategy table above: same model, same questions, "
             "same frame budget.\n")
    L.append("Source: https://longvideobench.github.io/ (leaderboard read 2026-09-29).\n")
    L.append("Reproduce: `bash scripts/run_lvb_benchmark.sh` then "
             "`python scripts/benchmark_report.py`. Every prediction is in `records.jsonl`.")
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
