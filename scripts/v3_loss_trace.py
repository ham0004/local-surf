"""Per-question loss trace on CG-Bench dev: where each wrong answer is lost (research_log step 75). No model calls.

For one system and answer model, every question falls in exactly one class, in pipeline order:
    pool miss     no candidate frame lies inside the human-marked seconds (the search never looked there)
    choice miss   the pool has such a frame, but none of the 4 chosen frames does
    reading miss  a chosen frame lies inside the marked seconds, the answer is wrong, but the same answer model
                  is right with at least one of the 5 oracle placements inside the marked seconds
    model limit   a chosen frame lies inside, the answer is wrong, and no oracle placement is right either
    right, hit    correct, with a chosen frame inside the marked seconds
    right, miss   correct without any chosen frame inside (other cues or a lucky guess)
Oracle: runs/v3_ceiling/answers.jsonl, H0 placements for systems without text, H600 for systems with text. The
classes are descriptive, not causal: "model limit" means the 5 tested placements failed, not that no frames can
help, and S0 (120 BM25 words) is compared with the H600 oracle, whose text is built around the gold frames.

A second table gives unconditional stage coverage over ALL questions (auditor P37): a candidate inside the marked
seconds in the pool, from Path A (speech moments), from Path B (visual search), among the 4 chosen; accuracy
overall and split by whether a chosen frame is inside. Inside = within 1 s of a human clue interval (as the rows).

    python scripts/v3_loss_trace.py          # reports/v3_loss_trace/{summary.md,questions.jsonl}
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN, OUT = Path("runs/v3_baseline"), Path("reports/v3_loss_trace")
SYSTEMS = {  # id: (pool, selector, words)
    "S0": ("v2", "clip", 120),
    "S1": ("hybrid", "mmropt", 0),
    "S1t": ("hybrid", "mmropt|tp=frames+hybrid", 600),
    "S3": ("hybrid", "file:cgbench_hybrid_ev_set", 0),
    "S4": ("hybrid_ahead", "file:cgbench_hybrid_ahead_ev_set", 0),
    "S4t": ("hybrid_ahead", "file:cgbench_hybrid_ahead_ev_set|tp=frames+hybrid", 600),
}
MODELS = {"2B": ("", "Qwen3-VL-2B-Instruct"), "4B": ("|ans=4b", "Qwen3-VL-4B-Instruct")}
CLASSES = ["pool miss", "choice miss", "reading miss", "model limit", "right, hit", "right, miss"]


def oracle() -> dict:
    """(model, condition 'H0'/'H600', qa_id) -> right with any of the 5 placements."""
    out: dict = collections.defaultdict(bool)
    for line in Path("runs/v3_ceiling/answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["set"] == "dev":
            out[(r["model"], r["condition"].split("_")[0], r["qa_id"])] |= bool(r["correct"])
    return out


def path_hits(pool: str, intervals: dict) -> dict:
    """qa_id -> (Path A candidate inside, Path B candidate inside) from the saved pool."""
    out = {}
    for q, iv in intervals.items():
        f = RUN / "cgbench" / pool / "pools" / q.replace(":", "_") / "pool.json"
        cands = json.loads(f.read_text(encoding="utf-8"))["candidates"] if f.exists() else []
        inside = [any(x - 1.0 <= c["time_s"] <= y + 1.0 for x, y in iv) for c in cands]
        out[q] = (any(h and "A" in c["paths"] for h, c in zip(inside, cands)),
                  any(h and "B" in c["paths"] for h, c in zip(inside, cands)))
    return out


def main() -> None:
    from v3_baseline_cycle import items  # noqa: PLC0415

    intervals = {it.qa.qa_id: it.qa.evidence_intervals_s for it in items("cgbench")}
    hits = {pool: path_hits(pool, intervals) for pool in {v[0] for v in SYSTEMS.values()}}
    stages: dict = {}
    last = {}
    for line in (RUN / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["dataset"] == "cgbench" and r["k"] == 4:
            last[(r["pool"], r["selector"], r["words"], r["qa_id"])] = r
    orc = oracle()
    table, per_q = {}, []
    for m, (tag, model_name) in MODELS.items():
        for sid, (pool, sel, words) in SYSTEMS.items():
            rows = [r for (p, s, w, _), r in last.items() if (p, s, w) == (pool, sel + tag, words)]
            cond = "H600" if words else "H0"
            counts = collections.Counter()
            stages[f"{sid} {m}"] = st = collections.Counter()
            for r in rows:
                a_hit, b_hit = hits[pool][r["qa_id"]]
                st.update({"pool": r["pool_hits_evidence"], "A": a_hit, "B": b_hit, "chosen": r["chosen_hits_evidence"],
                           "right": r["correct"], "right_hit": r["correct"] and r["chosen_hits_evidence"],
                           "right_miss": r["correct"] and not r["chosen_hits_evidence"]})
                if r["correct"]:
                    c = "right, hit" if r["chosen_hits_evidence"] else "right, miss"
                elif not r["pool_hits_evidence"]:
                    c = "pool miss"
                elif not r["chosen_hits_evidence"]:
                    c = "choice miss"
                else:
                    c = "reading miss" if orc[(model_name, cond, r["qa_id"])] else "model limit"
                counts[c] += 1
                per_q.append({"system": sid, "model": m, "qa_id": r["qa_id"], "class": c,
                              "pool_hit": r["pool_hits_evidence"], "path_a_hit": a_hit, "path_b_hit": b_hit,
                              "chosen_hit": r["chosen_hits_evidence"], "correct": r["correct"]})
            table[f"{sid} {m}"] = (len(rows), counts)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in per_q), encoding="utf-8")
    md = ["# Loss trace on CG-Bench dev (82 questions)", "",
          "Each question is put in one class: right answers by whether a chosen frame is inside the marked seconds, wrong "
          "answers by the first pipeline step whose output misses them. The "
          "classes are descriptive, not causal: they do not say which change would recover the answer, \"model limit\" "
          "means the 5 tested placements failed, and text systems are compared with an oracle whose text differs. "
          "Pre-filter and window losses are not separated.", "",
          "| System | n | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 2) + "|"]
    for k, (n, c) in table.items():
        md.append(f"| {k} | {n} | " + " | ".join(f"{c[x]} ({100 * c[x] / n:.0f}%)" for x in CLASSES) + " |")
    md += ["", "## Unconditional stage coverage (all questions)", "",
           "| System | pool inside | Path A inside | Path B inside | chosen inside | accuracy | accuracy, chosen inside "
           "| accuracy, chosen outside |", "|---|---|---|---|---|---|---|---|"]
    for k, st in stages.items():
        n = table[k][0]
        pc = lambda x, d=n: f"{100 * x / d:.0f}%" if d else "-"  # noqa: E731
        md.append(f"| {k} | {pc(st['pool'])} | {pc(st['A'])} | {pc(st['B'])} | {pc(st['chosen'])} | {pc(st['right'])} | "
                  f"{pc(st['right_hit'], st['chosen'])} | {pc(st['right_miss'], n - st['chosen'])} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
