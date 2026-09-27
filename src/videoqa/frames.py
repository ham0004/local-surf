"""Frame decoding with real presentation timestamps (PTS).

Key rules (tested in tests/test_frames.py):

* Requests are served in ascending time order.  Nearby requests share one
  forward decode; far-apart requests trigger a keyframe seek instead of
  decoding everything in between (``seek_gap_s``).
* For each requested time we return the first decoded frame whose PTS is
  >= the request (or the last frame for requests past the end), and we report
  that frame's true ``decoded_pts_s``.  Citations use the true PTS, which
  matters for variable-frame-rate video where "frame index / fps" is wrong.
* Every frame the decoder had to touch is counted in ``frames_visited`` so the
  cost trace includes frames decoded but never shown to the answerer.
"""

from __future__ import annotations

import dataclasses
import hashlib
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
              id_prefix: str = "f", video_id: str = "", seek_gap_s: float = 4.0) -> DecodeResult:
    """Decode one frame per requested time, in ascending time order.

    Strategy: decode forward from the nearest keyframe before a request; when
    the NEXT request is more than ``seek_gap_s`` ahead of the current frame,
    seek again instead of decoding everything in between.  Sparse candidates
    therefore cost roughly one GOP each rather than the whole span.
    ``seek_gap_s=float('inf')`` gives the old single-pass behaviour.

    ``max_side`` downsizes frames (keeping aspect ratio) right after decoding so
    large videos do not blow up memory.

    Returned frames are in the SAME order as ``times_s`` (not sorted), so the
    caller can zip them with its candidate list.
    """
    if not times_s:
        return DecodeResult([], 0)
    order = sorted(range(len(times_s)), key=lambda i: times_s[i])
    results: dict[int, Frame] = {}
    visited = 0

    def serve(idx: int, frame, pts: float) -> None:
        results[idx] = _to_frame(frame, pts, times_s[idx], max_side, f"{id_prefix}{idx:03d}")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        k = 0                      # index into ``order`` of the next request to serve
        last = None                # (frame, pts) of the most recent decoded frame
        while k < len(order):
            target = max(0.0, times_s[order[k]])
            # Seek (backward) to the keyframe at or before the next request.
            if stream.time_base is not None:
                container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
            jumped = False
            served_here = False        # has this seek segment served a request yet?
            for frame in container.decode(stream):
                visited += 1
                pts = float(frame.pts * stream.time_base) if frame.pts is not None else float(frame.time or 0.0)
                last = (frame, pts)
                while k < len(order) and pts + 1e-6 >= times_s[order[k]]:
                    serve(order[k], frame, pts)
                    k += 1
                    served_here = True
                if k >= len(order):
                    break
                # Jump ahead only after this segment made progress; otherwise a
                # keyframe far before the target would re-seek to itself forever.
                if served_here and times_s[order[k]] - pts > seek_gap_s:
                    jumped = True          # next request is far ahead: seek instead
                    break
            if not jumped:
                break                      # served everything, or reached end of stream
        # Requests beyond the last frame get the last frame (clamped to video end).
        while k < len(order) and last is not None:
            serve(order[k], *last)
            k += 1

    frames = [results[i] for i in range(len(times_s)) if i in results]
    for f in frames:
        f.video_id = video_id or Path(path).stem   # default: file name without extension
    return DecodeResult(frames, visited)


def _to_frame(frame: av.VideoFrame, pts: float, requested: float, max_side: int | None, fid: str) -> Frame:
    image = frame.to_image()  # PIL RGB
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.BILINEAR)
    return Frame(id=fid, requested_s=requested, decoded_pts_s=round(pts, 4), width=image.width,
                 height=image.height, image=image, phash=average_hash(image), digest=pixel_digest(image))


def pixel_digest(image: Image.Image) -> str:
    """sha256 of the exact RGB pixels (plus size). Two frames with the same
    digest are the same visual evidence, whatever time they were requested at."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
