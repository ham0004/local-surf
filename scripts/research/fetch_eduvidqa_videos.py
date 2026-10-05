"""EduVidQA adapter, step 2: lecture videos (+ captions where missing), in a fixed order.

Reads the official CSVs (data/eduvidqa_raw, from github.com/sourjyadip/eduvidqa-emnlp25),
picks videos of ONE official split in a fixed sha256 order of the video id (so
"--limit 20" is reproducible and a later larger run is a superset), and fetches
each with yt-dlp:

    <out>/videos/<video_id>.mp4           video-only stream, at most 480p (frames used at <= 640 px)
    <out>/transcripts/<video_id>.vtt      English captions (manual preferred, else auto),
                                          only if no transcript exists yet
    <out>/video_access.json               per video: ok / error message, bytes, caption kind

Availability is reported per split; nothing is silently dropped. yt-dlp is a
research-tool dependency (installed into the venv), not a runtime dependency
of the videoqa package. Videos are NPTEL/YouTube content used for research.

    python scripts/research/fetch_eduvidqa_videos.py --split synthetic_train --limit 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

SPLITS = {"synthetic_train": "synthetic_train.csv", "synthetic_test": "synthetic_test.csv",
          "real_world_test": "real_world_test.csv"}


def video_ids(raw: Path, split: str) -> list[str]:
    """Unique YouTube ids of one official split, in fixed sha256 order."""
    ids = set()
    for row in csv.DictReader(open(raw / SPLITS[split], encoding="utf-8")):
        url = row.get("vid_url") or row.get("url") or ""
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=lambda v: hashlib.sha256(v.encode()).hexdigest())


def fetch(vid: str, out: Path) -> dict:
    import yt_dlp  # noqa: PLC0415 - research-tool dependency

    vpath = out / "videos" / f"{vid}.mp4"
    have_transcript = any((out / "transcripts" / f"{vid}{e}").exists() for e in (".json", ".vtt", ".srt"))
    rec = {"video_id": vid, "ok": False}
    if vpath.exists():
        return {**rec, "ok": True, "bytes": vpath.stat().st_size, "cached": True}
    opts = {
        # Video-only stream: frames are all we decode (speech comes from captions), and a
        # single stream needs no ffmpeg merge.
        "format": "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[height<=480]",
        "outtmpl": str(out / "videos" / f"{vid}.%(ext)s"),
        "quiet": True, "no_warnings": True, "noprogress": True, "retries": 3,
        "writesubtitles": not have_transcript, "writeautomaticsub": not have_transcript,
        "subtitleslangs": ["en", "en-US", "en-GB"], "subtitlesformat": "vtt",
        "paths": {"subtitle": str(out / "transcripts")},
    }
    t0 = time.perf_counter()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        rec.update(ok=vpath.exists(), bytes=vpath.stat().st_size if vpath.exists() else 0,
                   duration_s=info.get("duration"), height=info.get("height"),
                   manual_captions=sorted(info.get("subtitles", {}) or {})[:5],
                   seconds=time.perf_counter() - t0)
        # yt-dlp names captions <id>.<lang>.vtt; normalise to <id>.vtt (manual English first).
        for lang in ("en", "en-US", "en-GB"):
            p = out / "transcripts" / f"{vid}.{lang}.vtt"
            if p.exists() and not (out / "transcripts" / f"{vid}.vtt").exists():
                p.rename(out / "transcripts" / f"{vid}.vtt")
                rec["caption_lang"] = lang
    except Exception as e:  # noqa: BLE001 - record every failure reason, never drop silently
        rec.update(error=str(e)[:300], seconds=time.perf_counter() - t0)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/eduvidqa_raw")
    p.add_argument("--out", default="data/eduvidqa")
    p.add_argument("--split", choices=tuple(SPLITS), default="synthetic_train")
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    log_path = out / "video_access.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    ids = video_ids(Path(a.raw), a.split)
    for n, vid in enumerate(ids[: a.limit]):
        rec = fetch(vid, out)
        rec["split"] = a.split
        log[vid] = rec
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"{n + 1}/{min(a.limit, len(ids))} {vid}: {'ok' if rec['ok'] else rec.get('error', 'failed')[:80]}",
              flush=True)
    done = [log[v] for v in ids[: a.limit]]
    print(json.dumps({"split": a.split, "videos_in_split": len(ids), "attempted": len(done),
                      "ok": sum(r["ok"] for r in done),
                      "gigabytes": round(sum(r.get("bytes", 0) for r in done) / 1e9, 2)}, indent=1))


if __name__ == "__main__":
    main()
