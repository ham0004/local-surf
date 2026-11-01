"""Question review of the lecture study (docs/v3/m3av_protocol.md, "Question review" and amendment 3): the
two-stage 100-question audit sample of the frozen test cohort and one packet per question.

Sample: lectures with a cohort question in SHA-256 order of `m3av-review-lecture:<lecture>`, the first 100; within a
lecture, questions in SHA-256 order of `m3av-review:<qa_id>`; further passes take a second question per lecture
only if fewer than 100 lectures exist. Packet (no system names, outputs or results): question, options, key,
evidence text, the saved slide image at full resolution, 3 frames from the proposed visible span at 360p (the
evaluation stream), the proposed visible spans, the transcript lines inside them, and the other valid occurrences.

Writes reports/m3av/review_sample_<split>.json (the sampled ids, committed) and the packets, which hold test
content and M3AV images, to runs/m3av/review_<split>/ (not in git): <n>_<qa_id>/{packet.md, slide.png,
frame_1..3.jpg}, index.md and an empty verdict form (form.csv) for each reviewer to fill alone.

    python scripts/m3av_review.py --split test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_questions import OUT, RAW  # noqa: E402
from m3av_video_check import video_of  # noqa: E402

N = 100
FORM = ["n", "qa_id", "verdict (accept/reject/uncertain)", "key correct (y/n)", "one best answer (y/n)",
        "supported by the slide (y/n)", "legible at 360p in the span (y/n)", "shortcut (none/knowledge/transcript)",
        "reason code", "note"]


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sample(qs: list[dict]) -> list[dict]:
    by: dict = {}
    for q in qs:
        by.setdefault(q["lecture"], []).append(q)
    lectures = sorted(by, key=lambda x: h(f"m3av-review-lecture:{x}"))
    queues = {lec: sorted(by[lec], key=lambda q: h(f"m3av-review:{q['qa_id']}")) for lec in lectures}
    out, depth = [], 0
    while len(out) < N and any(len(v) > depth for v in queues.values()):
        for lec in lectures[:N] if depth == 0 else lectures:
            if len(out) < N and len(queues[lec]) > depth:
                out.append(queues[lec][depth])
        depth += 1
    return out


def frames(lecture: str, spans: list, dest: Path) -> list[float]:
    """3 frames at 1/4, 1/2 and 3/4 of the total visible time (spans joined), from the 360p picture stream."""
    import av  # noqa: PLC0415

    total = sum(b - a for a, b in spans)
    want, times = [total * f for f in (0.25, 0.5, 0.75)], []
    for w in want:
        for a, b in spans:
            if w <= b - a:
                times.append(a + w)
                break
            w -= b - a
    with av.open(str(video_of(lecture))) as c:
        s = c.streams.video[0]
        for i, t in enumerate(times, 1):
            c.seek(int(max(0.0, t - 2.0) / s.time_base), stream=s, backward=True)
            for f in c.decode(s):
                if f.time is not None and f.time >= t:
                    f.to_image().save(dest / f"frame_{i}.jpg", quality=95)
                    break
    return [round(t, 1) for t in times]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("test",))
    a = p.parse_args()
    cohort = OUT / f"questions_{a.split}_visible.jsonl"
    qs = [json.loads(x) for x in cohort.read_text(encoding="utf-8").splitlines()]
    picked = sample(qs)
    rec = {"cohort": cohort.as_posix(), "cohort_questions": len(qs), "cohort_lectures": len({q["lecture"] for q in qs}),
           "sampled": len(picked), "sampled_lectures": len({q["lecture"] for q in picked}),
           "qa_ids": [q["qa_id"] for q in picked],
           "lecture_questions": {q["lecture"]: sum(x["lecture"] == q["lecture"] for x in qs) for q in picked}}
    Path("reports/m3av").mkdir(parents=True, exist_ok=True)
    Path(f"reports/m3av/review_sample_{a.split}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    root = Path(f"runs/m3av/review_{a.split}")
    root.mkdir(parents=True, exist_ok=True)
    index = [f"# Question review packets, M3AV {a.split} ({len(picked)} questions)", "",
             "Each folder holds packet.md, slide.png (M3AV's saved slide, full resolution) and frame_1..3.jpg (360p, "
             "from the proposed visible span). Record verdicts alone in a copy of form.csv.", ""]
    for n, q in enumerate(picked, 1):
        d = root / f"{n:03d}_{q['qa_id'].replace(':', '_')}"
        d.mkdir(exist_ok=True)
        shutil.copyfile(RAW / q["lecture"] / "ocr" / "seg_imgs" / q["source_image"], d / "slide.png")
        ts = frames(q["lecture"], q["visible_spans"], d)
        lines = json.loads(Path(f"data/m3av_{a.split}/transcripts/{q['lecture']}.json").read_text(encoding="utf-8"))
        inside = [x for x in lines if any(x["end"] > s and x["start"] < e for s, e in q["visible_spans"])]
        others = [v for v in q["valid_segments"] if v != q["source_segment"]]
        md = [f"# {n:03d} {q['qa_id']}", "", f"**Question.** {q['question']}", ""]
        md += [f"{'**' if i == q['answer_index'] else ''}{'ABCD'[i]}. {o}{' (key)**' if i == q['answer_index'] else ''}"
               for i, o in enumerate(q["options"])]
        md += ["", f"**Evidence text.** {q['evidence_text']}", "",
               f"**Source slide segment.** {q['source_segment'][0]:.1f}-{q['source_segment'][1]:.1f} s (slide.png)",
               "**Proposed visible spans.** " + ", ".join(f"{s:.1f}-{e:.1f} s" for s, e in q["visible_spans"])
               + f" ({q['visible_s']:.1f} s in all)",
               "**Frames.** " + ", ".join(f"frame_{i}.jpg at {t} s" for i, t in enumerate(ts, 1)),
               "**Other valid occurrences of the slide.** " + (", ".join(f"{s:.1f}-{e:.1f} s" for s, e in others)
                                                              if others else "none"), "",
               "**Transcript lines inside the visible spans.**", ""]
        md += [f"- [{x['start']:.1f}-{x['end']:.1f}] {x['text']}" for x in inside] or ["- (none)"]
        (d / "packet.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        index.append(f"- [{n:03d} {q['qa_id']}]({d.name}/packet.md)")
        print(f"{n}/{len(picked)}", flush=True)
    (root / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    with open(root / "form.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FORM)
        w.writerows([[n, q["qa_id"]] + [""] * (len(FORM) - 2) for n, q in enumerate(picked, 1)])
    print(f"{rec['sampled']} questions from {rec['sampled_lectures']} of {rec['cohort_lectures']} lectures; {root}")


if __name__ == "__main__":
    main()
