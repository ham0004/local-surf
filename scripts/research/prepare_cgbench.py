"""CG-Bench adapter for the English-subtitled videos we hold: QA layout + declared split.

Input: data/cgbench_raw/cgbench.json (12,129 questions) and the videos/subtitles fetched by
scripts/research/fetch_cgbench_videos.py (data/cgbench/videos, data/cgbench/transcripts/*.srt).

Output: data/cgbench/qa.jsonl (videoqa.datasets layout) with, per question: choices, gold index,
the human-annotated clue intervals (as evidence_intervals_s), domain and sub_category.

Split, declared before any CG-Bench result (2026-10-09): by video, fixed sha256 rule,
60% train / 20% dev / 20% test. CG-Bench has no official split, so this is a custom protocol and is
reported as such. Test videos are used once, for the final frozen comparison.

    python scripts/research/prepare_cgbench.py
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
from pathlib import Path

RAW = Path("data/cgbench_raw/cgbench.json")
OUT = Path("data/cgbench")


def split_of(video_uid: str) -> str:
    h = int(hashlib.sha256(f"cgbench:{video_uid}".encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else ("dev" if h < 8 else "test")


def main() -> None:
    have = {p.stem for p in (OUT / "videos").glob("*.mp4")} & {p.stem for p in (OUT / "transcripts").glob("*.srt")}
    rows, counts, vids = [], collections.Counter(), collections.defaultdict(set)
    for q in json.loads(RAW.read_text(encoding="utf-8")):
        if q["video_uid"] not in have:
            continue
        choices = q["choices"] if isinstance(q["choices"], list) else ast.literal_eval(q["choices"])
        gold = "ABCDEFGHIJKLMNOP".index(q["right_answer"].strip())
        clues = q["clue_intervals"] if isinstance(q["clue_intervals"], list) else ast.literal_eval(q["clue_intervals"])
        split = split_of(q["video_uid"])
        rows.append({"qa_id": f"cgbench:{q['qid']}", "video_id": q["video_uid"], "question": q["question"],
                     "options": list(choices), "gold_option_index": gold, "gold_answer": q["answer"],
                     "evidence_intervals_s": [[float(a), float(b)] for a, b in clues],
                     "source_dataset": "CG-Bench (CG-Bench/CG-Bench)", "official_split": "none",
                     "experiment_split": split, "source_split": split,
                     "evidence_type": f"{q['domain']}:{q['sub_category']}", "is_synthetic": False,
                     "duration_s": float(q["duration"]),
                     "provenance_and_license": "CG-Bench human-annotated QA with clue intervals (MIT); videos for "
                                               "academic research under the dataset agreement"})
        counts[split] += 1
        vids[split].add(q["video_uid"])
    (OUT / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"dataset": "CG-Bench (English-subtitled subset we hold)", "videos": len(have), "questions": len(rows),
                "questions_per_split": dict(counts), "videos_per_split": {k: len(v) for k, v in vids.items()},
                "split_rule": "sha256('cgbench:'+video_uid) % 10: 0-5 train, 6-7 dev, 8-9 test",
                "gold_position_counts": dict(collections.Counter("ABCDEFGHIJ"[r["gold_option_index"]] for r in rows)),
                "options_per_question": dict(collections.Counter(len(r["options"]) for r in rows))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
