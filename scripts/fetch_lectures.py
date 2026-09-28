"""Download CC BY-NC-SA MIT OpenCourseWare lectures (Gilbert Strang) from the
olive5/ml-lectures Hugging Face archive, into our local layout:

    <out>/videos/<video_id>.mp4
    <out>/transcripts/<video_id>.vtt     (human en-US if present, else YouTube ASR "en")
    <out>/lectures.json                  video_id -> course, title, caption kind, bytes

Only the two MIT OCW folders are used by default: their licence (CC BY-NC-SA
4.0, attribution Gilbert Strang, MIT OpenCourseWare) permits research use.
The Stanford / YouTube-licensed folders are excluded unless --courses names them.

Selection is a fixed sha256 order of the file name, so "--limit 5" is a
reproducible pilot and a later full run is a superset of it.

    python scripts/fetch_lectures.py --limit 5 --out data/mit_lectures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "olive5/ml-lectures"
MIT = ("18.06SC_LinearAlgebra", "18.065_Strang")
CAPTION_PREF = (".en-US.vtt", ".en.vtt", ".en-orig.vtt")
SKIP = re.compile(r"interview|course introduction", re.I)     # not lectures


def video_id(course: str, stem: str) -> str:
    num = stem.split(" - ")[0].strip()
    return f"{course.split('_')[0]}_{num}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", default=",".join(MIT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/mit_lectures")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    stage = out / "_hf"

    files = {s.rfilename: s.size for s in HfApi().dataset_info(REPO, files_metadata=True).siblings}
    lectures = []
    for f, size in files.items():
        course, _, name = f.partition("/")
        if course in a.courses.split(",") and name.endswith(".mp4") and not SKIP.search(name):
            stem = name[:-4]
            cap = next((f"{course}/{stem}{ext}" for ext in CAPTION_PREF if f"{course}/{stem}{ext}" in files), None)
            if cap:
                lectures.append((hashlib.sha256(f.encode()).hexdigest(), course, stem, f, cap, size))
    lectures.sort()
    chosen = lectures[: a.limit] if a.limit else lectures

    meta_p = out / "lectures.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    for n, (_, course, stem, mp4, cap, size) in enumerate(chosen):
        vid = video_id(course, stem)
        dst_v, dst_t = out / "videos" / f"{vid}.mp4", out / "transcripts" / f"{vid}.vtt"
        if not dst_v.exists():
            shutil.move(hf_hub_download(REPO, mp4, repo_type="dataset", local_dir=stage), dst_v)
        if not dst_t.exists():
            shutil.copy(hf_hub_download(REPO, cap, repo_type="dataset", local_dir=stage), dst_t)
        meta[vid] = {"course": course, "title": stem, "caption": cap.rsplit(".", 2)[-2],
                     "caption_is_asr": not cap.endswith(".en-US.vtt"), "bytes": size,
                     "license": "CC BY-NC-SA 4.0, Gilbert Strang, MIT OpenCourseWare", "source": f"{REPO}/{mp4}"}
        meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[{n + 1}/{len(chosen)}] {vid}  {stem[:70]}  {size / 1e6:.0f} MB", flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"{len(meta)} lectures in {out}, {len(lectures)} available")


if __name__ == "__main__":
    main()
