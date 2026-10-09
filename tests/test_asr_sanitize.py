"""ASR segment sanitising: Whisper word timestamps can run backwards or overlap at chunk joins."""
import importlib.util
from pathlib import Path


def _asr():
    spec = importlib.util.spec_from_file_location("asr", Path(__file__).resolve().parents[1] / "scripts/research/asr_transcribe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sanitize_orders_and_repairs_backwards_segments():
    segs = [{"start": 60.0, "end": 66.0, "text": "a"}, {"start": 68.76, "end": 61.94, "text": "b"},
            {"start": 64.0, "end": 70.0, "text": "c"}]
    out = _asr().sanitize(segs)
    assert all(x["end"] > x["start"] for x in out)
    assert all(b["start"] >= a["end"] for a, b in zip(out, out[1:]))
    assert [x["text"] for x in out] == ["a", "b", "c"]
