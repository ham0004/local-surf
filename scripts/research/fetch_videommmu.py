"""Video-MMMU adapter: extract the released videos and write our QA layout.

Input (after accepting the dataset agreement on Hugging Face and downloading lmms-lab/VideoMMMU):
    data/videommmu_raw/{Art,Business,Science,Engineering,Humanities,Medicine}.zip
    data/videommmu_raw/{Perception,Comprehension,Adaptation}/test-00000-of-00001.parquet

Output (videoqa.datasets.load_local_dataset layout):
    data/videommmu/videos/<video_id>.mp4     video_id = the dataset id, e.g. validation_Music_8
    data/videommmu/qa.jsonl                  Perception + Comprehension (multiple-choice only)
    data/videommmu/manifest.json             counts, split rule, provenance

Protocol (declared before any result): Video-MMMU is released as a test set only. To keep a final
test untouched while still running gates and tuning, videos are split by a fixed sha256 rule:
one third DEV (gates, tuning, method selection) and two thirds TEST (used once, frozen). Questions
follow their video. Adaptation questions (which add an exam image) are excluded for now.

    python scripts/research/fetch_videommmu.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import zipfile
from pathlib import Path

RAW = Path("data/videommmu_raw")
OUT = Path("data/videommmu")
SUBJECT_ZIPS = ("Art", "Business", "Science", "Engineering", "Humanities", "Medicine")
TRACKS = ("Perception", "Comprehension")


def split_of(video_id: str) -> str:
    """Fixed video-level split: 1/3 dev, 2/3 test."""
    return "dev" if int(hashlib.sha256(f"videommmu:{video_id}".encode()).hexdigest(), 16) % 3 == 0 else "test"


def main() -> None:
    import pyarrow.parquet as pq  # noqa: PLC0415 - research dependency

    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    have = set()
    for subject in SUBJECT_ZIPS:
        with zipfile.ZipFile(RAW / f"{subject}.zip") as z:
            for name in z.namelist():
                if name.endswith(".mp4") and not name.startswith("__MACOSX") and "/._" not in name:
                    vid = Path(name).stem
                    dest = OUT / "videos" / f"{vid}.mp4"
                    if not dest.exists():
                        with z.open(name) as src, open(dest, "wb") as dst:
                            while chunk := src.read(1 << 22):
                                dst.write(chunk)
                    have.add(vid)
    rows, counts = [], collections.Counter()
    for track in TRACKS:
        for r in pq.read_table(RAW / track / "test-00000-of-00001.parquet").to_pylist():
            if r["question_type"] != "multiple-choice" or r["id"] not in have:
                counts[f"skipped:{track}"] += 1
                continue
            options = list(r["options"])
            gold = "ABCDEFGHIJKLMNOP".index(r["answer"].strip())
            split = split_of(r["id"])
            rows.append({"qa_id": f"vmmmu:{track}:{r['id']}", "video_id": r["id"], "question": r["question"],
                         "options": options, "gold_option_index": gold, "gold_answer": options[gold],
                         "evidence_intervals_s": [], "source_dataset": "Video-MMMU (lmms-lab/VideoMMMU)",
                         "official_split": "test", "experiment_split": split, "source_split": split,
                         "evidence_type": f"{track}:{r['qa_type']}", "is_synthetic": False,
                         "provenance_and_license": "Video-MMMU human-annotated; videos from YouTube via the "
                                                   "dataset release, academic use under the dataset agreement"})
            counts[f"{track}:{split}"] += 1
    OUT.joinpath("qa.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    manifest = {"dataset": "Video-MMMU", "videos_extracted": len(have), "questions": len(rows),
                "counts": dict(counts), "split_rule": "sha256('videommmu:'+video_id) % 3 == 0 -> dev, else test",
                "tracks": list(TRACKS), "excluded": "Adaptation track (adds an exam image); non-multiple-choice"}
    OUT.joinpath("manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
