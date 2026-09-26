"""Frame decoding with real presentation timestamps (PTS).

Key rules (tested in tests/test_frames.py):

* Requests are served in ascending time order in ONE pass per call, so a batch
  of candidates costs one sequential decode, not N random seeks.
* For each requested time we return the first decoded frame whose PTS is
  >= the request (or the last frame for requests past the end), and we report
  that frame's true ``decoded_pts_s``.  Citations use the true PTS, which
  matters for variable-frame-rate video where "frame index / fps" is wrong.
* Every frame the decoder had to touch is counted in ``frames_visited`` so the
  cost trace includes frames decoded but never shown to the answerer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .schemas import Frame


@dataclasses.dataclass
class VideoInfo:
    path: str
    duration_s: float
    width: int
    height: int
    avg_fps: float


def probe(path: str | Path) -> VideoInfo:
    """Read duration/size/fps from the container without decoding frames."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = 0.0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        return VideoInfo(str(path), duration, stream.codec_context.width, stream.codec_context.height, fps)


def average_hash(image: Image.Image, size: int = 8) -> str:
    """64-bit average hash (hex).  Cheap near-duplicate detector.

    Caveat: two slides that differ only by a minus sign can hash identically,
    so hashes are used to *count* redundancy, never to silently delete frames
    the controller asked for.
    """
    small = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


@dataclasses.dataclass
class DecodeResult:
    frames: list[Frame]
    frames_visited: int    # every frame the decoder produced, including skipped ones


def decode_at(path: str | Path, times_s: list[float], max_side: int | None = None,
              id_prefix: str = "f") -> DecodeResult:
    """Decode one frame per requested time in a single sequential pass.

    ``max_side`` downsizes frames (keeping aspect ratio) right after decoding so
    large videos do not blow up memory; crop coordinates would be relative to
    this resized image, and the scale is recoverable from width/height.

    Returned frames are in the SAME order as ``times_s`` (not sorted), so the
    caller can zip them with its candidate list.
    """
    if not times_s:
        return DecodeResult([], 0)
    order = sorted(range(len(times_s)), key=lambda i: times_s[i])
    results: dict[int, Frame] = {}
    visited = 0

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek once to the nearest keyframe before the earliest request.
        first_t = max(0.0, times_s[order[0]])
        if first_t > 1.0 and stream.time_base is not None:
            container.seek(int(first_t / stream.time_base), stream=stream, backward=True, any_frame=False)

        pending = list(order)
        last_frame = None
        for frame in container.decode(stream):
            visited += 1
            pts = float(frame.pts * stream.time_base) if frame.pts is not None else float(frame.time or 0.0)
            last_frame = (frame, pts)
            # Serve every pending request whose time has been reached.
            while pending and pts + 1e-6 >= times_s[pending[0]]:
                idx = pending.pop(0)
                results[idx] = _to_frame(frame, pts, times_s[idx], max_side, f"{id_prefix}{idx:03d}")
            if not pending:
                break
        # Requests beyond the last frame get the last frame (clamped to video end).
        for idx in pending:
            if last_frame is not None:
                frame, pts = last_frame
                results[idx] = _to_frame(frame, pts, times_s[idx], max_side, f"{id_prefix}{idx:03d}")

    return DecodeResult([results[i] for i in range(len(times_s)) if i in results], visited)


def _to_frame(frame: av.VideoFrame, pts: float, requested: float, max_side: int | None, fid: str) -> Frame:
    image = frame.to_image()  # PIL RGB
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.BILINEAR)
    return Frame(id=fid, requested_s=requested, decoded_pts_s=round(pts, 4), width=image.width,
                 height=image.height, image=image, phash=average_hash(image))
