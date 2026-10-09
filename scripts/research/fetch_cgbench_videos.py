"""Fetch selected CG-Bench videos from inside the remote release zips, by HTTP byte range.

The release (Hugging Face CG-Bench/CG-Bench, after accepting its agreement) packs 770 videos into
32 zips of ~5 GB. Only videos with ENGLISH subtitles fit our transcript-guided pipeline (51 of
them are in the zips), and they are spread over 25 zips. Instead of downloading ~128 GB, this
reads each zip's central directory remotely (a few KB) and then downloads only the wanted
members (stored or deflated), writing:

    data/cgbench/videos/<video_uid>.mp4
    data/cgbench/transcripts/<video_uid>.srt
    data/cgbench/fetch_log.json

    python scripts/research/fetch_cgbench_videos.py
"""

from __future__ import annotations

import json
import re
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from videoqa.v3.language import latin_share  # noqa: E402

RAW = Path("data/cgbench_raw")
OUT = Path("data/cgbench")
REPO = "CG-Bench/CG-Bench"


def _get(url: str, token: str, start: int, end: int | None = None):
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Range": rng})
    return urllib.request.urlopen(req, timeout=300)


def central_directory(url: str, token: str) -> tuple[int, list[dict]]:
    """Members of a remote zip (name, method, sizes, local header offset), zip64-aware."""
    with _get(url, token, 0, 0) as r:
        size = int(r.headers["Content-Range"].split("/")[-1])
    with _get(url, token, max(0, size - 200_000)) as r:
        tail = r.read()
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    with _get(url, token, cd_off, cd_off + cd_size - 1) as r:
        cd = r.read()
    members, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + n].decode("utf-8", "replace")
        extra = cd[p + 46 + n:p + 46 + n + e]
        q = 0
        while q + 4 <= len(extra):                     # zip64 extended information
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, k = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[k:k + 8])[0]
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[k:k + 8])[0]
                    k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", vals[k:k + 8])[0]
            q += 4 + hlen
        members.append({"name": name, "method": method, "csize": csize, "usize": usize, "offset": off})
        p += 46 + n + e + c
    return size, members


def extract_member(url: str, token: str, m: dict, dest: Path) -> None:
    with _get(url, token, m["offset"], m["offset"] + 29) as r:
        h = r.read()
    n, e = struct.unpack("<HH", h[26:30])
    start = m["offset"] + 30 + n + e
    tmp = dest.with_suffix(".part")
    inflater = zlib.decompressobj(-15) if m["method"] == 8 else None
    if m["method"] not in (0, 8):
        raise ValueError(f"unsupported compression {m['method']}")
    with _get(url, token, start, start + m["csize"] - 1) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(inflater.decompress(chunk) if inflater else chunk)
        if inflater:
            f.write(inflater.flush())
    if tmp.stat().st_size != m["usize"]:
        raise ValueError(f"size mismatch for {m['name']}")
    tmp.replace(dest)


def english_subtitled() -> dict[str, bytes]:
    z = zipfile.ZipFile(RAW / "subtitles.zip")
    out = {}
    for n in z.namelist():
        if n.endswith(".srt"):
            raw = z.read(n)
            text = re.sub(r"[\d:,\->\n]", " ", raw.decode("utf-8", "replace"))
            if latin_share(text) >= 0.9:
                out[Path(n).stem] = raw
    return out


def main() -> None:
    from huggingface_hub import get_token, hf_hub_url  # noqa: PLC0415

    token = get_token()
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "transcripts").mkdir(parents=True, exist_ok=True)
    wanted = english_subtitled()
    log_path = OUT / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for k in range(1, 33):
        fname = f"video_chunk_{k:02d}.zip"
        url = hf_hub_url(REPO, fname, repo_type="dataset")
        _, members = central_directory(url, token)
        for m in members:
            vid = Path(m["name"]).stem
            if vid not in wanted or m["name"].endswith("/") or "__MACOSX" in m["name"]:
                continue
            dest = OUT / "videos" / f"{vid}.mp4"
            if not dest.exists():
                extract_member(url, token, m, dest)
            (OUT / "transcripts" / f"{vid}.srt").write_bytes(wanted[vid])
            log[vid] = {"zip": fname, "bytes": dest.stat().st_size}
            log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
            print(f"{fname} {vid} {dest.stat().st_size / 1e6:.0f} MB", flush=True)
    print(json.dumps({"english_subtitled": len(wanted), "videos_fetched": len(log),
                      "gigabytes": round(sum(v["bytes"] for v in log.values()) / 1e9, 2)}, indent=1))


if __name__ == "__main__":
    main()
