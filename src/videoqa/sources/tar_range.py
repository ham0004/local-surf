"""Selective extraction from a (possibly multi-part) remote tar via HTTP Range requests.

LongVideoBench ships its 753 videos as ONE 161.6 GB tar split into 31 equal-size
parts (see longvideobench.py). Downloading that to fetch a handful of videos
would be wasteful and mostly pointless for a pilot. Tar is a sequential format
(filename + size header, then padded data, repeated), but every header and
every data block is 512-byte aligned, and each part size here is itself a
multiple of 512 — verified for this dataset — so no read ever straddles a part
boundary and we can jump straight to the next header via Range requests
without downloading the bytes of files we do not want.

Cost: one small HTTP request per archive entry (usually just the 512-byte
header; a bit more for the ustar long-name extension) plus the actual payload
for every entry that matches ``wanted``. For LongVideoBench this is roughly
750 small requests plus the wanted videos' bytes, not 161.6 GB.

This module is dataset-shaped generically (part sizes + resolve URLs); it is
not specific to LongVideoBench beyond the "part.aa, part.ab, ..." naming used
by that repository's `hf_hub_download`/tar-split convention.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


@dataclasses.dataclass
class TarPart:
    name: str
    size: int          # bytes; must be an exact multiple of 512


@dataclasses.dataclass
class RemoteTarEntry:
    name: str
    global_offset: int  # offset of this entry's HEADER in the virtual concatenated tar
    size: int
    is_dir: bool


class RemoteMultipartTar:
    """A tar archive spread across ``parts``, addressed by one HTTP GET per part."""

    def __init__(self, base_url: str, parts: list[TarPart], token: str | None = None,
                 timeout: float = 30.0, max_retries: int = 6, retry_backoff_s: float = 2.0) -> None:
        if any(p.size % 512 for p in parts):
            raise ValueError("every part size must be a multiple of 512 for header-safe range reads")
        self.base_url = base_url.rstrip("/") + "/"
        self.parts = parts
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._cum = []
        total = 0
        for p in parts:
            self._cum.append(total)
            total += p.size
        self.total_size = total

    def _locate(self, global_offset: int) -> tuple[int, int]:
        """(part_index, offset_within_part) for a global byte offset."""
        for i in range(len(self.parts) - 1, -1, -1):
            if global_offset >= self._cum[i]:
                return i, global_offset - self._cum[i]
        raise ValueError(f"offset {global_offset} before start of archive")

    def read(self, global_offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``global_offset``. Must not straddle
        a part boundary (guaranteed by 512-aligned headers/data + part sizes).

        A full scan of LongVideoBench's archive is ~750 requests; over a real
        network some of those WILL time out or reset transiently (measured: a
        connection timeout killed a 40-video download after only 1 file).
        Transient failures (timeouts, connection errors, HTTP 429/5xx) are
        retried with exponential backoff; a non-transient HTTP error (401,
        404, ...) is raised immediately since retrying cannot fix it.
        """
        if length <= 0:
            return b""
        part_idx, local_off = self._locate(global_offset)
        part = self.parts[part_idx]
        if local_off + length > part.size:
            raise ValueError(f"read of {length}B at {global_offset} would straddle part {part.name}; "
                             "this should not happen with 512-aligned tar part sizes")
        url = self.base_url + part.name
        headers = {"Range": f"bytes={local_off}-{local_off + length - 1}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"unexpected status {resp.status} reading {part.name}")
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    last_error = e
                    continue
                raise RuntimeError(f"HTTP {e.code} reading {part.name} bytes {local_off}-{local_off+length-1}: "
                                   f"{e.read()[:200]!r}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_error = e
                continue
        raise RuntimeError(
            f"giving up after {self.max_retries + 1} attempts reading {part.name} "
            f"bytes {local_off}-{local_off+length-1}: {last_error!r}"
        ) from last_error

    def iter_entries(self, stop_after: set[str] | None = None):
        """Yield :class:`RemoteTarEntry` in archive order.

        If ``stop_after`` is given, stop as soon as every name in it has been
        yielded (saves scanning the rest of a large archive once all wanted
        files are found). GNU long-name (typeflag 'L') headers are followed
        transparently so long filenames are reported correctly.
        """
        offset = 0
        found: set[str] = set()
        pending_long_name: str | None = None
        while offset + 512 <= self.total_size:
            header = self.read(offset, 512)
            if header == b"\x00" * 512:
                break  # end-of-archive marker
            name = header[0:100].split(b"\x00", 1)[0].decode("utf-8", "replace")
            typeflag = header[156:157]
            size_field = header[124:136].split(b"\x00", 1)[0].strip()
            size = int(size_field, 8) if size_field else 0
            data_blocks = (size + 511) // 512
            if typeflag == b"L":
                # GNU long-name entry: its "payload" IS the real name of the NEXT header.
                pending_long_name = self.read(offset + 512, size).split(b"\x00", 1)[0].decode("utf-8", "replace")
                offset += 512 + data_blocks * 512
                continue
            if pending_long_name is not None:
                name, pending_long_name = pending_long_name, None
            if name:  # skip the empty/padding records some tars end with
                entry = RemoteTarEntry(name=name.rstrip("\x00"), global_offset=offset, size=size,
                                       is_dir=(typeflag == b"5"))
                yield entry
                if stop_after and entry.name in stop_after:
                    found.add(entry.name)
                    if found >= stop_after:
                        return
            offset += 512 + data_blocks * 512

    def extract_to(self, entry: RemoteTarEntry, dest: str | Path, chunk_size: int = 8 * 1024 * 1024) -> int:
        """Download one entry's payload to ``dest`` in chunks (bounded memory)."""
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        data_offset = entry.global_offset + 512
        written = 0
        with open(dest, "wb") as fh:
            while written < entry.size:
                n = min(chunk_size, entry.size - written)
                fh.write(self.read(data_offset + written, n))
                written += n
        return written


def hf_dataset_tar_parts(dataset_id: str, part_names: list[str], token: str | None = None) -> tuple[str, list[TarPart]]:
    """Fetch exact sizes for named files of an HF dataset repo and build TarParts.

    Uses :func:`huggingface_hub.HfApi.dataset_info` (not a manual API call) so
    it works the same way the rest of this codebase talks to the Hub.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415 - optional heavy import

    info = HfApi().dataset_info(dataset_id, files_metadata=True, token=token)
    sizes = {s.rfilename: s.size for s in info.siblings}
    missing = [n for n in part_names if n not in sizes]
    if missing:
        raise KeyError(f"parts not found in {dataset_id}: {missing}")
    base = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/"
    return base, [TarPart(n, sizes[n]) for n in part_names]


def save_manifest(entries: list[RemoteTarEntry], path: str | Path) -> None:
    """Persist a full directory scan so it never needs to be re-fetched."""
    Path(path).write_text(json.dumps([dataclasses.asdict(e) for e in entries]), encoding="utf-8")


def load_manifest(path: str | Path) -> list[RemoteTarEntry]:
    return [RemoteTarEntry(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
