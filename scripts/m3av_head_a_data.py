"""Head A data from M3AV lectures (docs/v3/m3av_protocol.md, system S2m): question + timed speech lines + evidence.

Same record format as scripts/v3_head_a_data.py, so the Head A scripts train S2m unchanged (they read
HEAD_A_DATA / HEAD_A_RUN). Speech lines are M3AV's sentences (`speech/final+timestamps.json`, written form,
times from `timestr` in ms). Evidence: train = the question's valid segments (train lectures have no video);
dev = its visible spans (the primary measure's target). Test is never read.

    python scripts/m3av_head_a_data.py          # data/head_a_m3av/{train,dev}.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_questions import OUT, RAW, segments  # noqa: E402

DEST = Path("data/head_a_m3av")


def lines(lecture: str) -> list[list]:
    out = []
    for s in json.loads((RAW / lecture / "speech" / "final+timestamps.json").read_text(encoding="utf-8")):
        a, b = (int(x) / 1000 for x in s["timestr"].split("_"))
        if s["final_written"].strip() and b > a:
            out.append([round(a, 2), round(b, 2), s["final_written"].strip()])
    return sorted(out)


def record(q: dict, evidence: list) -> dict:
    ls = lines(q["lecture"])
    end = max([segments(q["lecture"])[-1]["end"]] + [b for _, b, _ in ls[-1:]])
    return {"qa_id": q["qa_id"], "source": "m3av", "video_id": q["lecture"], "question": q["question"],
            "options": q["options"], "lines": ls, "evidence": evidence, "duration_s": float(end), "weak": False}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for split, src, ev in (("train", "questions_train.jsonl", "valid_segments"),
                           ("dev", "questions_dev_visible.jsonl", "visible_spans")):
        qs = [json.loads(x) for x in (OUT / src).read_text(encoding="utf-8").splitlines() if x.strip()]
        recs = [record(q, q[ev]) for q in qs]
        (DEST / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                                             encoding="utf-8")
        print(f"{split}: {len(recs)} questions, {len({r['video_id'] for r in recs})} lectures, "
              f"no speech lines: {sum(not r['lines'] for r in recs)}")


if __name__ == "__main__":
    main()
