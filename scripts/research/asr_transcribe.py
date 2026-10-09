"""Speech-to-text for benchmark videos that ship without transcripts (e.g. Video-MMMU).

Uses Whisper large-v3-turbo through transformers on the local GPU (no extra CUDA libraries),
with audio decoded by PyAV and resampled to 16 kHz mono. Word timestamps are grouped into
caption-like lines (break after a 1 s pause, 14 words or 6 s). Output per video, in our transcript
layout (videoqa.transcript.load_transcript):

    <out>/<video_id>.json     [{"start": s, "end": s, "text": "..."}]
    <out>/_asr_meta.json      per video: model, revision, language, seconds, segments, latin_share

The transcript is a DECLARED part of the protocol: Video-MMMU's paper also used Whisper ASR, but a
different model and settings, so our transcript-only and frame results are not the paper's numbers.

    python scripts/research/asr_transcribe.py --videos data/videommmu/videos --out data/videommmu/transcripts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from videoqa.v3.language import latin_share  # noqa: E402

MODEL = "openai/whisper-large-v3-turbo"


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Mono float32 audio at ``sr`` Hz decoded with PyAV."""
    import av  # noqa: PLC0415

    out = []
    with av.open(str(path)) as c:
        stream = next(s for s in c.streams if s.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in c.decode(stream):
            for f in resampler.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def group_words(words, max_words: int = 14, max_gap_s: float = 1.0, max_len_s: float = 6.0) -> list[dict]:
    """Word timestamps -> caption-like lines (a new line after a pause, ~14 words or ~6 s)."""
    segs, cur = [], []
    for w in words:
        s, e = w["timestamp"]
        if s is None:
            continue
        e = e if e is not None else s + 0.3
        if cur and (len(cur) >= max_words or s - cur[-1][1] > max_gap_s or e - cur[0][0] > max_len_s):
            segs.append({"start": cur[0][0], "end": cur[-1][1], "text": " ".join(t for _, _, t in cur).strip()})
            cur = []
        cur.append((float(s), float(e), w["text"].strip()))
    if cur:
        segs.append({"start": cur[0][0], "end": cur[-1][1], "text": " ".join(t for _, _, t in cur).strip()})
    return segs


def sanitize(segs: list[dict]) -> list[dict]:
    """Make segments valid and time-ordered: Whisper word timestamps occasionally run backwards
    (e.g. [68.76, 61.94]). Each segment starts no earlier than the previous one ends and has end >= start."""
    out, prev_end = [], 0.0
    for x in segs:
        s = max(float(x["start"]), prev_end)
        e = max(float(x["end"]), s + 0.01)
        out.append({"start": round(s, 3), "end": round(e, 3), "text": x["text"]})
        prev_end = e
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--ids", default=None, help="optional file with one video id per line (transcribe these first)")
    p.add_argument("--repair", action="store_true", help="only re-sanitize existing transcript files, no ASR")
    p.add_argument("--batch", type=int, default=1, help="word timestamps keep attention maps; >1 can exhaust 16 GB")
    a = p.parse_args()
    out = Path(a.out)
    if a.repair:
        fixed = 0
        for f in out.glob("*.json"):
            if f.name.startswith("_"):
                continue
            segs = json.loads(f.read_text(encoding="utf-8"))
            clean = sanitize(segs)
            if clean != segs:
                f.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
                fixed += 1
        print(f"repaired {fixed} transcript files")
        return
    import torch  # noqa: PLC0415
    from transformers import pipeline  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "_asr_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    asr = pipeline("automatic-speech-recognition", model=MODEL, dtype=torch.float16, device="cuda:0",
                   model_kwargs={"cache_dir": "cache/hf/hub"})
    revision = getattr(asr.model.config, "_commit_hash", None)
    videos = sorted(Path(a.videos).glob("*.mp4"))
    if a.ids:
        keep = {x.strip() for x in Path(a.ids).read_text(encoding="utf-8").splitlines() if x.strip()}
        videos = [v for v in videos if v.stem in keep]
    for n, v in enumerate(videos):
        if (out / f"{v.stem}.json").exists():
            continue
        t0 = time.perf_counter()
        audio = load_audio(v)
        if not len(audio):
            meta[v.stem] = {"error": "no audio"}
            continue
        res = asr({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=a.batch,
                  return_timestamps="word", generate_kwargs={"language": a.language, "task": "transcribe"})
        segs = sanitize(group_words(res.get("chunks", [])))
        (out / f"{v.stem}.json").write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")
        meta[v.stem] = {"model": MODEL, "revision": revision, "language": a.language,
                        "audio_seconds": round(len(audio) / 16000, 1), "seconds": round(time.perf_counter() - t0, 1),
                        "segments": len(segs), "latin_share": round(latin_share(" ".join(x["text"] for x in segs)), 3)}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{n + 1}/{len(videos)} {v.stem}: {len(segs)} segments, {meta[v.stem]['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
