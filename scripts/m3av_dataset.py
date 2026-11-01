"""M3AV questions in the project's local dataset layout (src/videoqa/datasets.py), for the lecture study's systems.

    data/m3av_<split>/qa.jsonl                  one QAItem per question; evidence_intervals_s = visible spans (F4),
                                                so the answer stage's chosen_hits_evidence (1 s tolerance) is the
                                                primary measure, source-slide interval hit@4
    data/m3av_<split>/transcripts/<lecture>.json  M3AV's timed sentences (as scripts/m3av_head_a_data.py)
    data/m3av_<split>/videos/<lecture>.mp4      hard link to the 360p picture stream (no copy)

Test is built only from the frozen cohort file, after the method freeze (docs/v3/m3av_protocol.md).

    python scripts/m3av_dataset.py --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_head_a_data import lines  # noqa: E402
from m3av_questions import OUT  # noqa: E402
from m3av_video_check import video_of  # noqa: E402

LICENSE = "CC BY-NC-SA 4.0 (M3AV v1.0); source videos keep their own rights"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    a = p.parse_args()
    root = Path(f"data/m3av_{a.split}")
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(exist_ok=True)
    qs = [json.loads(x) for x in (OUT / f"questions_{a.split}_visible.jsonl").read_text(encoding="utf-8").splitlines()
          if x.strip()]
    rows = []
    for q in qs:
        rows.append({"qa_id": q["qa_id"], "video_id": q["lecture"], "question": q["question"],
                     "gold_answer": q["options"][q["answer_index"]], "options": q["options"],
                     "gold_option_index": q["answer_index"], "evidence_intervals_s": q["visible_spans"],
                     "source_dataset": "M3AV", "source_split": a.split, "provenance_and_license": LICENSE,
                     "evidence_type": "visual_support", "official_split": a.split, "experiment_split": a.split,
                     "is_synthetic": True})
    for lec in sorted({q["lecture"] for q in qs}):
        dst = root / "videos" / f"{lec}.mp4"
        if not dst.exists():
            os.link(video_of(lec), dst)
        (root / "transcripts" / f"{lec}.json").write_text(
            json.dumps([{"start": s, "end": e, "text": t} for s, e, t in lines(lec)], ensure_ascii=False), encoding="utf-8")
    (root / "qa.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"dataset": "M3AV v1.0 lecture questions (generated)", "license": LICENSE,
                                                    "split": a.split, "questions": len(rows),
                                                    "lectures": len({q["lecture"] for q in qs})}, indent=1), encoding="utf-8")
    print(f"{root}: {len(rows)} questions, {len({q['lecture'] for q in qs})} lectures")


if __name__ == "__main__":
    main()
