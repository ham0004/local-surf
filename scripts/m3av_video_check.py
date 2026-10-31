"""M3AV dev/test videos: is every picture stream complete, and do M3AV's slide times match our YouTube copy?

Only the picture stream is needed (yt-dlp could not merge picture and sound without ffmpeg; `<id>.f605.mp4` is the
360p picture stream). For each lecture: decodable duration vs the end of M3AV's last slide segment, and the
normalised correlation (64 x 36 grey) between the saved slide image and our frame at that segment's last second,
for 5 segments spread over the lecture. A copy with a different intro would give low correlations.

    python scripts/m3av_video_check.py          # reports/m3av/video_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_questions import RAW, SPLIT, UNAVAILABLE  # noqa: E402

VIDEOS = Path("data/m3av_videos")


def video_of(lecture: str) -> Path:
    url = next((RAW / lecture / "raw").glob("*.ytbUrl")).read_text(encoding="utf-8").strip()
    return VIDEOS / f"{url.split('v=')[-1][:11]}.f605.mp4"


def small(img) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((64, 36)), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def frames_at(path: Path, times: list[float]) -> dict:
    """The decoded frame nearest each requested time (seconds), as small grey vectors."""
    import av  # noqa: PLC0415

    out = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for t in sorted(times):
            c.seek(int(max(0.0, t - 2.0) / s.time_base), stream=s, backward=True)
            best = None
            for f in c.decode(s):
                if f.time is None:
                    continue
                best = f
                if f.time >= t:
                    break
            if best is not None:
                out[t] = small(best.to_image())
    return out


def duration(path: Path) -> float:
    import av  # noqa: PLC0415

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        last = 0.0
        c.seek(int(1e7 / s.time_base), stream=s, backward=True)      # jump far past the end: lands on the last key
        for f in c.decode(s):
            last = max(last, f.time or 0.0)
        return last


def main() -> None:
    from PIL import Image  # noqa: PLC0415

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = []
    for lec in sorted(k for k, v in split.items() if v in ("dev", "test") and k not in UNAVAILABLE):
        path, segs = video_of(lec), json.loads((RAW / lec / "ocr" / "seg_speech_ocr.json").read_text(encoding="utf-8"))
        row = {"lecture": lec, "split": split[lec], "video": path.name, "exists": path.exists(),
               "m3av_end": segs[-1]["end"] if segs else None}
        if not segs:
            row["error"] = "no slide segments in M3AV"
        elif path.exists():
            try:
                row["video_end"] = round(duration(path), 1)
                pick = [segs[i] for i in np.linspace(1, len(segs) - 1, min(5, len(segs) - 1)).round().astype(int)]
                got = frames_at(path, [s["end"] - 0.5 for s in pick])
                cors = [float(small(Image.open(RAW / lec / "ocr" / "seg_imgs" / s["name"])) @ got[s["end"] - 0.5] / (64 * 36))
                        for s in pick if s["end"] - 0.5 in got]
                row["corr"] = [round(c, 2) for c in cors]
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(lec, row.get("video_end"), row.get("m3av_end"), row.get("corr"), row.get("error", ""), flush=True)
    out = Path("reports/m3av")
    out.mkdir(parents=True, exist_ok=True)
    (out / "video_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    med = [np.median(r["corr"]) for r in rows if r.get("corr")]
    print(f"{len(rows)} lectures; videos present {sum(r['exists'] for r in rows)}; errors {sum('error' in r for r in rows)}; "
          f"median correlation {np.median(med):.2f}; lectures with median < 0.5: {sum(m < 0.5 for m in med)}")


if __name__ == "__main__":
    main()
