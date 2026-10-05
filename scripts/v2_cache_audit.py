"""Could the legacy (v1) cache key have served a wrong answer in a historical run?

The v1 key omitted frame time labels and transcript ids/end times. Inside one
run, a collision needs two DIFFERENT prompts with the same v1 key:

  1. two candidates of the same pool with identical pixels but different times
     (a static board decoded twice, kept as separate candidates), or
  2. two transcript lines of the same question with equal start time and text
     but different id/end (Head A's single-line interventions), or
  3. a gold index that changed after labelling (records stored correctness).

This script counts each case for the given runs. Zero for all three means no
collision of THESE kinds was found in the audited caches. It does not
re-render every historical request or configuration, so it is evidence, not
proof, that no historical result was affected.

    python scripts/v2_cache_audit.py --runs runs/v2_main runs/v2_balanced runs/v2_pilot \
        --out reports/v2_review/cache_identity_audit.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def audit_run(run: Path) -> dict:
    meta = json.loads((run / "pools.json").read_text(encoding="utf-8"))
    same_pixels_diff_time, line_clashes, gold_changes, records = [], [], 0, 0
    golds = {}
    for q in meta["qa_ids"]:
        d = run / "pools" / q.replace(":", "_").replace("/", "_")
        pool = json.loads((d / "pool.json").read_text(encoding="utf-8"))
        golds[q] = pool["gold_option_index"]
        by_digest = collections.defaultdict(set)
        for c in pool["candidates"]:
            by_digest[c["digest"]].add(round(c["time_s"], 2))
        same_pixels_diff_time += [(q, dg) for dg, ts in by_digest.items() if len(ts) > 1]
        by_line = collections.defaultdict(set)
        for s in pool["transcript"]:
            by_line[(s["start_s"], s["text"])].add((s["id"], s["end_s"]))
        line_clashes += [q for v in by_line.values() if len(v) > 1]
    cache = run / "teacher_cache.jsonl"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records += 1
                if "quality" in rec and rec.get("option") is not None and rec["qa_id"] in golds:
                    gold_changes += int((rec["option"] == golds[rec["qa_id"]]) != bool(rec["quality"]))
    return {"questions": len(meta["qa_ids"]), "cache_records": records,
            "same_pixels_different_time_in_one_pool": len(same_pixels_diff_time),
            "transcript_lines_differing_only_in_id_or_end": len(line_clashes),
            "records_whose_stored_correctness_disagrees_with_current_gold": gold_changes,
            "examples": same_pixels_diff_time[:5]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["runs/v2_main", "runs/v2_balanced", "runs/v2_pilot"])
    p.add_argument("--out", default="reports/v2_review/cache_identity_audit.json")
    a = p.parse_args()
    out = {r: audit_run(Path(r)) for r in a.runs if (Path(r) / "pools.json").exists()}
    out["conclusion"] = ("no collision of the checked kinds found in these runs" if all(
        v["same_pixels_different_time_in_one_pool"] == v["transcript_lines_differing_only_in_id_or_end"]
        == v["records_whose_stored_correctness_disagrees_with_current_gold"] == 0 for v in out.values())
        else "possible v1 collisions found; see counts")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
