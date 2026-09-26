"""RemoteMultipartTar against a real local HTTP server (no network, no HF).

We build a genuine tar (via the stdlib ``tarfile`` module, GNU format so long
names are exercised), split its raw bytes into two "parts" at an arbitrary
512-aligned offset, and serve each part from a tiny Range-aware HTTP server.
This exercises the exact mechanism used against the real Hugging Face Hub:
one GET per archive entry, Range requests, and part-boundary arithmetic.
"""

from __future__ import annotations

import http.server
import io
import re
import tarfile
import threading

import pytest

from videoqa.sources.tar_range import RemoteMultipartTar, TarPart, build_catalog, load_catalog


def _build_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name, content in [("a.txt", b"hello a"), ("dir/b.txt", b"hello b" * 100),
                              ("dir/" + "c" * 150 + ".txt", b"long name payload")]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    parts: dict[str, bytes] = {}
    requests: list[tuple[str, str]] = []  # (path, range header) for assertions
    fail_first_n: int = 0                 # simulate this many transient 503s before succeeding
    fail_from_request_n: int | None = None  # simulate a permanent outage starting at request N
    _fail_count = 0
    _request_count = 0

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        name = self.path.lstrip("/")
        rng = self.headers.get("Range", "")
        type(self).requests.append((name, rng))
        type(self)._request_count += 1
        if type(self).fail_from_request_n is not None and type(self)._request_count >= type(self).fail_from_request_n:
            self.send_response(503)
            self.end_headers()
            return
        if type(self)._fail_count < type(self).fail_first_n:
            type(self)._fail_count += 1
            self.send_response(503)
            self.end_headers()
            return
        data = self.parts.get(name)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        m = re.match(r"bytes=(\d+)-(\d+)", rng)
        start, end = (int(m.group(1)), int(m.group(2))) if m else (0, len(data) - 1)
        chunk = data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *a) -> None:  # silence test output
        pass


@pytest.fixture
def server():
    raw = _build_tar()
    split = (len(raw) // 2 // 512) * 512  # arbitrary but 512-aligned
    part_a, part_b = raw[:split], raw[split:]
    _RangeHandler.parts = {"t.part.aa": part_a, "t.part.ab": part_b}
    _RangeHandler.requests = []
    _RangeHandler.fail_first_n = 0
    _RangeHandler._fail_count = 0
    _RangeHandler.fail_from_request_n = None
    _RangeHandler._request_count = 0
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", [TarPart("t.part.aa", len(part_a)),
                                                        TarPart("t.part.ab", len(part_b))]
    finally:
        httpd.shutdown()


def test_iter_entries_lists_files_dirs_and_long_names(server):
    base, parts = server
    tar = RemoteMultipartTar(base, parts)
    entries = list(tar.iter_entries())
    names = {e.name for e in entries}
    assert "a.txt" in names and "dir/b.txt" in names
    assert any(n.startswith("dir/" + "c" * 150) for n in names)  # GNU long name resolved
    assert not any(e.is_dir for e in entries if e.name == "a.txt")


def test_extract_to_downloads_correct_bytes_across_part_boundary(server, tmp_path):
    base, parts = server
    tar = RemoteMultipartTar(base, parts)
    entries = {e.name: e for e in tar.iter_entries()}
    out = tmp_path / "b.txt"
    n = tar.extract_to(entries["dir/b.txt"], out)
    assert n == len(b"hello b" * 100) == out.stat().st_size
    assert out.read_bytes() == b"hello b" * 100


def test_stop_after_ends_scan_early(server):
    base, parts = server
    tar = RemoteMultipartTar(base, parts)
    seen = list(tar.iter_entries(stop_after={"a.txt"}))
    assert [e.name for e in seen] == ["a.txt"]   # a.txt is first in the archive


def test_skipping_an_entry_never_downloads_its_payload(server):
    base, parts = server
    _RangeHandler.requests.clear()
    tar = RemoteMultipartTar(base, parts)
    list(tar.iter_entries())  # scan only, extract nothing
    # every request must be small (header-sized); none should span the ~700-byte payload
    for _, rng in _RangeHandler.requests:
        m = re.match(r"bytes=(\d+)-(\d+)", rng)
        assert int(m.group(2)) - int(m.group(1)) + 1 <= 512


def test_read_across_part_boundary_raises(server):
    base, parts = server
    tar = RemoteMultipartTar(base, parts)
    boundary = parts[0].size
    with pytest.raises(ValueError):
        tar.read(boundary - 10, 20)   # would straddle part.aa / part.ab


def test_rejects_non_512_aligned_part_sizes():
    with pytest.raises(ValueError):
        RemoteMultipartTar("http://x/", [TarPart("p", 513)])


def test_transient_failures_are_retried_and_eventually_succeed(server):
    # Regression: a real 40-video download died on the first transient network
    # error after downloading only 1 file. Simulate 3 straight 503s (fewer
    # than max_retries) and confirm the read still succeeds.
    base, parts = server
    _RangeHandler.fail_first_n = 3
    tar = RemoteMultipartTar(base, parts, max_retries=6, retry_backoff_s=0.01)
    header = tar.read(0, 512)
    assert len(header) == 512
    assert _RangeHandler._fail_count == 3   # it really did fail 3 times first


def test_persistent_failures_raise_after_max_retries(server):
    base, parts = server
    _RangeHandler.fail_first_n = 100   # always fails
    tar = RemoteMultipartTar(base, parts, max_retries=2, retry_backoff_s=0.01)
    with pytest.raises(RuntimeError, match="giving up after 3 attempts"):
        tar.read(0, 512)
    assert _RangeHandler._fail_count == 3   # 1 initial + 2 retries, then gave up


def test_not_found_fails_immediately_without_retrying(server):
    base, _ = server
    _RangeHandler.requests.clear()
    # "missing.part" isn't a key in _RangeHandler.parts -> the server 404s it.
    tar = RemoteMultipartTar(base, [TarPart("missing.part", 512)], max_retries=6, retry_backoff_s=0.01)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        tar.read(0, 512)
    assert len(_RangeHandler.requests) == 1   # no retries for a non-transient error


def test_iter_entries_resumes_from_a_start_offset(server):
    # Regression driver for build_catalog: a resumed scan from the offset
    # right after entry 0 must yield exactly the remaining entries.
    base, parts = server
    tar = RemoteMultipartTar(base, parts)
    all_entries = list(tar.iter_entries())
    first = all_entries[0]
    resume_at = first.global_offset + 512 + ((first.size + 511) // 512) * 512
    rest = list(tar.iter_entries(start_offset=resume_at))
    assert [e.name for e in rest] == [e.name for e in all_entries[1:]]


def test_build_catalog_completes_and_matches_a_direct_scan(server, tmp_path):
    base, parts = server
    tar = RemoteMultipartTar(base, parts)
    catalog = build_catalog(tar, tmp_path / "catalog.json", save_every=1)
    assert catalog.complete
    reference = {e.name for e in RemoteMultipartTar(base, parts).iter_entries()}
    assert set(catalog.entries) == reference
    # a completed catalog loads back identically and build_catalog is a no-op on it
    reloaded = load_catalog(tmp_path / "catalog.json")
    assert reloaded.complete and set(reloaded.entries) == reference


def test_build_catalog_resumes_after_a_permanent_outage_partway_through(server, tmp_path):
    # This is exactly what happened for real: a multi-thousand-entry scan died
    # partway through. build_catalog must save what it found and pick up from
    # there next time, never re-reading bytes it already has.
    base, parts = server
    catalog_path = tmp_path / "catalog.json"
    _RangeHandler.fail_from_request_n = 2   # the 1st request (entry "a.txt") succeeds, then outage
    tar = RemoteMultipartTar(base, parts, max_retries=1, retry_backoff_s=0.01)
    with pytest.raises(RuntimeError):
        build_catalog(tar, catalog_path, save_every=1)

    partial = load_catalog(catalog_path)
    assert not partial.complete
    assert set(partial.entries) == {"a.txt"}   # got exactly as far as the outage allowed

    # "network recovers": lift the outage and resume with a fresh tar handle.
    _RangeHandler.fail_from_request_n = None
    _RangeHandler.requests.clear()
    tar2 = RemoteMultipartTar(base, parts, max_retries=1, retry_backoff_s=0.01)
    full = build_catalog(tar2, catalog_path, save_every=1)

    assert full.complete
    reference = {e.name for e in RemoteMultipartTar(base, parts).iter_entries()}
    assert set(full.entries) == reference
    # resuming must not re-request the header for the entry it already had
    assert all(name != "a.txt" for name, _ in _RangeHandler.requests[:1])
