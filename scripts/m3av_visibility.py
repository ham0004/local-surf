"""M3AV visibility proxy, filter F4 (docs/v3/m3av_protocol.md): in which seconds is the saved slide on screen?

For every valid evidence segment of a question, frames at 1 per second (360p picture stream) are compared with that
segment's saved slide image (M3AV keeps the LAST frame of a segment, so a slide that builds up matches only once
complete) and with the saved images of the previous and next segments. Measure (amendment 2 of the protocol): the
correlation of edge maps (grey, 256 x 144, Gaussian blur 1 px, absolute horizontal + vertical differences). A second
is visible when the own correlation is >= tau and not below either neighbour's (a neighbour that is itself a valid
evidence segment of the question is not compared). The 64 x 36 grey correlation first declared could not separate
a different slide with the same layout (up to 0.97) from the right slide under burned-in subtitles (0.58).

    python scripts/m3av_visibility.py curves --split dev        # per-second correlations, cached
    python scripts/m3av_visibility.py sheets --split dev        # 20 build segments as picture sheets (to choose tau)
    python scripts/m3av_visibility.py sheets --split dev --exclude reports/m3av/sheets   # 20 not yet seen (held-out check)
    python scripts/m3av_visibility.py apply --split dev --tau T # visible spans; drops questions under 2 s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_questions import OUT, RAW, segments  # noqa: E402
from m3av_video_check import video_of  # noqa: E402

CURVES = Path("runs/m3av/curves_edge")
MIN_VISIBLE_S = 2.0


def edges(img) -> np.ndarray:
    """Normalised edge map: grey 256 x 144, blur 1 px, |horizontal| + |vertical| differences."""
    from PIL import ImageFilter  # noqa: PLC0415

    a = np.asarray(img.convert("L").resize((256, 144)).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
    g = np.zeros_like(a)
    g[:, 1:] += np.abs(np.diff(a, axis=1))
    g[1:] += np.abs(np.diff(a, axis=0))
    g = g.ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def segment_curve(lecture: str, segs: list[dict], k: int) -> list[list[float | None]]:
    """[(time, own, previous, next)] edge correlations at 1 frame per second inside segment k."""
    import av  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    def ref(j: int):                                # None: no such segment, or M3AV has no saved image for it
        path = RAW / lecture / "ocr" / "seg_imgs" / segs[j]["name"] if 0 <= j < len(segs) else None
        return edges(Image.open(path)) if path is not None and path.exists() else None

    refs = [ref(k), ref(k - 1), ref(k + 1)]
    start, end = segs[k]["start"], segs[k]["end"]
    out, want = [], start
    with av.open(str(video_of(lecture))) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, start - 2.0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.time is None or f.time < want:
                continue
            if f.time > end + 0.5:
                break
            e = edges(f.to_image())
            out.append([round(float(f.time), 2)] + [None if r is None else round(float(e @ r / r.size), 4) for r in refs])
            want = f.time + 1.0
    return out


def stage_curves(a) -> None:
    CURVES.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(x) for x in (OUT / f"questions_{a.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    by_lecture: dict = {}
    for q in qs:
        by_lecture.setdefault(q["lecture"], set()).update(tuple(v) for v in q["valid_segments"])
    for n, (lec, spans) in enumerate(sorted(by_lecture.items())):
        path = CURVES / f"{lec}.json"
        have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        segs = segments(lec)
        index = {s["start"]: k for k, s in enumerate(segs)}
        for st, _ in sorted(spans):
            if str(st) not in have:
                have[str(st)] = segment_curve(lec, segs, index[st])
        path.write_text(json.dumps(have), encoding="utf-8")
        print(f"{n + 1}/{len(by_lecture)} {lec}: {len(spans)} segments", flush=True)


def stage_sheets(a) -> None:
    """20 segments whose correlation rises inside the segment (a slide that builds up), picked in a fixed hash order;
    each sheet shows the saved slide and frames at 5 times with their correlations."""
    import av  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    builds = []
    for path in sorted(CURVES.glob("*.json")):
        for st, curve in json.loads(path.read_text(encoding="utf-8")).items():
            c = [x[1] for x in curve]                   # own-slide correlation
            if len(c) >= 4 and min(c[: len(c) // 2]) < 0.9 and max(c) > 0.95:
                builds.append((hashlib.sha256(f"m3av-sheets:{path.stem}:{st}".encode()).hexdigest(), path.stem, float(st), curve))
    builds.sort()
    seen = {p.stem for p in Path(a.exclude).glob("*.jpg")} if a.exclude else set()
    builds = [b for b in builds if f"{b[1]}_{b[2]:g}" not in seen]
    out = Path("reports/m3av/sheets_edge")
    out.mkdir(parents=True, exist_ok=True)
    for _, lec, st, curve in builds[:20]:
        seg = next(s for s in segments(lec) if s["start"] == st)
        picks = [curve[round(i)] for i in np.linspace(0, len(curve) - 1, 5)]
        tiles = [Image.open(RAW / lec / "ocr" / "seg_imgs" / seg["name"]).convert("RGB").resize((320, 180))]
        labels = ["saved slide (segment end)"]
        with av.open(str(video_of(lec))) as c:
            s = c.streams.video[0]
            for t, corr, prev, nxt in picks:
                c.seek(int(max(0.0, t - 2.0) / s.time_base), stream=s, backward=True)
                fr = next(f for f in c.decode(s) if f.time is not None and f.time >= t - 0.01)
                tiles.append(fr.to_image().convert("RGB").resize((320, 180)))
                nb = max((x for x in (prev, nxt) if x is not None), default=float("nan"))
                labels.append(f"t={t:.2f}s own={corr:.2f} neighbour={nb:.2f}")
        sheet = Image.new("RGB", (320 * 3, 200 * 2), "white")
        d = ImageDraw.Draw(sheet)
        for i, (im, lab) in enumerate(zip(tiles, labels, strict=True)):
            x, y = 320 * (i % 3), 200 * (i // 3)
            sheet.paste(im, (x, y))
            d.text((x + 4, y + 182), lab, fill="black")
        sheet.save(out / f"{lec}_{st:g}.jpg", quality=85)
    print(f"{len(builds)} build segments not excluded; {min(20, len(builds))} sheets in {out}")


def stage_apply(a) -> None:
    qs = [json.loads(x) for x in (OUT / f"questions_{a.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    curves = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in CURVES.glob("*.json")}
    kept, dropped = [], 0
    for q in qs:
        spans, valid = [], {st for st, _ in q["valid_segments"]}
        segs = segments(q["lecture"])
        index = {s["start"]: k for k, s in enumerate(segs)}
        for st, en in q["valid_segments"]:
            k = index[st]
            prev_ok = k > 0 and segs[k - 1]["start"] not in valid          # compare only with non-evidence neighbours
            next_ok = k + 1 < len(segs) and segs[k + 1]["start"] not in valid
            vis = [t for t, own, prev, nxt in curves[q["lecture"]][str(st)]
                   if own >= a.tau and not (prev_ok and prev is not None and prev > own)
                   and not (next_ok and nxt is not None and nxt > own)]
            if vis:
                spans.append([min(vis), min(en, max(vis) + 1.0)])
        q["visible_spans"], q["visible_s"] = spans, round(sum(e - s for s, e in spans), 2)
        if q["visible_s"] >= MIN_VISIBLE_S:
            kept.append(q)
        else:
            dropped += 1
    (OUT / f"questions_{a.split}_visible.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in kept),
                                                             encoding="utf-8")
    print(f"{a.split}: tau {a.tau}: kept {len(kept)}, dropped {dropped} (visible under {MIN_VISIBLE_S} s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("curves", "sheets", "apply"))
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--exclude", default=None, help="sheets: skip segments that already have a sheet in this folder")
    a = p.parse_args()
    {"curves": stage_curves, "sheets": stage_sheets, "apply": stage_apply}[a.stage](a)


if __name__ == "__main__":
    main()
