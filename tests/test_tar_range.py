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

from videoqa.sources.tar_range import RemoteMultipartTar, TarPart


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
    _fail_count = 0

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        name = self.path.lstrip("/")
        rng = self.headers.get("Range", "")
        type(self).requests.append((name, rng))
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
