"""Grade free-text answers with a local LLM judge, and write the report.

Input:  an `evaluate --clean-only --open-ended` run (records.jsonl: one free
        answer per question x policy) and the dataset's qa.jsonl (question and
        gold answer text).
Judge:  microsoft/Phi-4-mini-instruct (MIT licence), deliberately a different
        model family from the Qwen answerer, greedy decoding, fixed prompt.
        For each answer it returns {"correct": true/false, "score": 0-5}.
Output: <out>/judged.jsonl        one verdict per answer (resumable)
        <out>/RESULTS.md          judge accuracy per strategy (95% CI by video),
                                  mean 0-5 score, token F1
        <out>/results.json
        <out>/human_audit.csv     50 answers for a person to grade, so the
                                  judge's agreement with a human can be reported

    python scripts/judge_open_ended.py --run runs/bench_lvb_v1_open --out reports/benchmark_v1_open_ended
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

JUDGE = "microsoft/Phi-4-mini-instruct"
JUDGE_REV = "cfbefacb99257ffa30c83adab238a50856ac3083"

SYSTEM = ("You grade answers to questions about videos. Compare the CANDIDATE answer with the REFERENCE answer. "
          "The candidate is correct if its main answer means the same as the reference: paraphrases, synonyms and "
          "extra correct detail are fine. It is incorrect if it gives a different answer, contradicts the "
          "reference, or says it cannot tell. Reply with JSON only, no explanation: "
          '{"correct": true or false, "score": an integer 0-5 where 5 = fully matches and 0 = unrelated}.')
USER = "QUESTION: {q}\nREFERENCE: {ref}\nCANDIDATE: {cand}"

POLICY_LABEL = {
    "transcript_only": "Transcript only (no frames)",
    "uniform": "Uniform frames + transcript",
    "retrieval": "Transcript-retrieved moments",
    "scout_similarity": "MobileCLIP top-k frames",
    "heuristic": "Cost-aware heuristic controller",
}


class Judge:
    def __init__(self, cache_dir: str = "cache/hf/hub"):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(JUDGE, revision=JUDGE_REV, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(JUDGE, revision=JUDGE_REV, cache_dir=cache_dir,
                                                          dtype=torch.bfloat16, device_map="cuda").eval()

    def __call__(self, question: str, reference: str, candidate: str) -> tuple[dict | None, str]:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(q=question, ref=reference, cand=candidate[:600])}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=32, do_sample=False)
        text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return parse_verdict(text), text


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d.get("correct"), bool):
                return {"correct": d["correct"], "score": int(d.get("score", 5 if d["correct"] else 0))}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    low = text.lower()
    if '"correct": true' in low or "correct: true" in low:
        return {"correct": True, "score": 5}
    if '"correct": false' in low or "correct: false" in low:
        return {"correct": False, "score": 0}
    return None


def boot_ci(by_video: dict[str, list[float]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vids = list(by_video)
    sums = np.array([sum(by_video[v]) for v in vids])
    cnts = np.array([len(by_video[v]) for v in vids])
    idx = rng.integers(0, len(vids), size=(n_boot, len(vids)))
    acc = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bench_lvb_v1_open")
    ap.add_argument("--data", default="data/longvideobench_full")
    ap.add_argument("--out", default="reports/benchmark_v1_open_ended")
    ap.add_argument("--audit-n", type=int, default=50)
    a = ap.parse_args()
    run, data, out = Path(a.run), Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(x) for x in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    qa = {q["qa_id"]: q for q in map(json.loads, (data / "qa.jsonl").read_text(encoding="utf-8").splitlines()) if q}

    judged_p = out / "judged.jsonl"
    done = {}
    if judged_p.exists():
        for x in judged_p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                j = json.loads(x)
                done[(j["qa_id"], j["policy"])] = j
    todo = [r for r in recs if (r["qa_id"], r["policy"]) not in done]
    if todo:
        judge, t0 = Judge(), time.time()
        with open(judged_p, "a", encoding="utf-8") as fh:
            for n, r in enumerate(todo):
                q = qa[r["qa_id"]]
                verdict, raw = judge(q["question"], q["gold_answer"], r["answer"])
                j = {"qa_id": r["qa_id"], "video_id": r["video_id"], "policy": r["policy"],
                     "correct": bool(verdict and verdict["correct"]), "score": verdict["score"] if verdict else 0,
                     "parse_failed": verdict is None, "judge_raw": raw}
                fh.write(json.dumps(j) + "\n")
                done[(r["qa_id"], r["policy"])] = j
                if (n + 1) % 100 == 0:
                    print(f"judged {n + 1}/{len(todo)}  {(time.time() - t0) / (n + 1):.2f}s each", flush=True)

    policies = [p for p in POLICY_LABEL if any(r["policy"] == p for r in recs)]
    res = {}
    for p in policies:
        rs = [r for r in recs if r["policy"] == p]
        js = [done[(r["qa_id"], p)] for r in rs]
        by_vid = collections.defaultdict(list)
        for j in js:
            by_vid[j["video_id"]].append(float(j["correct"]))
        lo, hi = boot_ci(by_vid)
        res[p] = {"n": len(rs), "judge_accuracy": float(np.mean([j["correct"] for j in js])), "ci95": [lo, hi],
                  "mean_score_0_5": float(np.mean([j["score"] for j in js])),
                  "token_f1": float(np.mean([r["quality"] for r in rs])),
                  "frames_mean": float(np.mean([r["frames"] for r in rs])),
                  "warm_ms_median": float(np.median([r["warm_ms"] for r in rs])),
                  "judge_parse_failures": int(sum(j["parse_failed"] for j in js))}
    summary = {"judge": {"model": JUDGE, "revision": JUDGE_REV, "decoding": "greedy", "system_prompt": SYSTEM},
               "questions": res[policies[0]]["n"], "policies": res}
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rng = random.Random(0)
    sample = rng.sample(recs, min(a.audit_n, len(recs)))
    with open(out / "human_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qa_id", "policy", "question", "reference", "candidate", "judge_correct", "human_correct (fill 1/0)"])
        for r in sample:
            q = qa[r["qa_id"]]
            w.writerow([r["qa_id"], r["policy"], q["question"], q["gold_answer"], r["answer"],
                        int(done[(r["qa_id"], r["policy"])]["correct"]), ""])

    L = ["# Open-ended benchmark: LongVideoBench without options (v1 framework)\n",
         f"The same {summary['questions']} LongVideoBench questions as the multiple-choice benchmark, but the "
         "**options are hidden**: the model writes its own answer, which is graded against the dataset's "
         "correct answer text. Same model (Qwen3-VL-2B), same ≤8-frame budget, same GPU.\n",
         "## How answers are graded\n",
         f"A local judge model ({JUDGE}, a different model family from the answerer, greedy decoding) reads the "
         "question, the reference answer and the candidate answer, and returns correct/incorrect plus a 0–5 "
         "score. The exact judge prompt is in `results.json`. Token F1 (word overlap with the reference) is "
         "reported alongside as a judge-free check. `human_audit.csv` holds 50 random answers for a person to "
         "grade; until that is done, the judge's agreement with humans is **not yet measured**.\n",
         "## Results\n",
         "| strategy | judge accuracy | 95% CI (by video) | mean score (0–5) | token F1 | frames | time / question |",
         "|---|---|---|---|---|---|---|"]
    for p in policies:
        r = res[p]
        L.append(f"| {POLICY_LABEL[p]} | **{100 * r['judge_accuracy']:.1f}%** | {100 * r['ci95'][0]:.1f}–"
                 f"{100 * r['ci95'][1]:.1f}% | {r['mean_score_0_5']:.2f} | {r['token_f1']:.3f} | "
                 f"{r['frames_mean']:.1f} | {r['warm_ms_median'] / 1000:.2f} s |")
    fails = sum(r["judge_parse_failures"] for r in res.values())
    L += ["", f"Judge replies that could not be parsed (counted as incorrect): {fails}.\n",
          "**Caveats.** LongVideoBench is a multiple-choice benchmark; its reference answers are short option "
          "texts (e.g. \"black short hair\"), so this measures short free answers, not long explanations. There "
          "is no published open-ended leaderboard for it, so these numbers compare the strategies with each "
          "other, not with other models.\n",
          "Reproduce: `evaluate --clean-only --open-ended` (see `scripts/run_lvb_benchmark.sh`), then "
          "`python scripts/judge_open_ended.py`."]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
