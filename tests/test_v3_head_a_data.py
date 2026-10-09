"""Head A data helpers and the dense-features Path B scope."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from v3_head_a_data import strip_timestamp  # noqa: E402

from videoqa.v2.candidates import _feature_scan_times  # noqa: E402


def test_strip_leading_time():
    assert strip_timestamp("At 3:54, can you explain X?") == ("Can you explain X?", 234.0)


def test_strip_opening_clause_with_time():
    q, t = strip_timestamp("Referring to the slide at 1:46, the instructor discusses three types.")
    assert (q, t) == ("The instructor discusses three types.", 106.0)


def test_strip_time_inside_sentence():
    assert strip_timestamp("Why does the bound at 12:30 hold?") == ("Why does the bound hold?", 750.0)


def test_strip_hours_and_no_time():
    assert strip_timestamp("At 1:02:03, why?")[1] == 3723.0
    assert strip_timestamp("no time here") == ("no time here", None)


def test_feature_scan_times_ranks_and_spaces(tmp_path):
    times = np.arange(0, 20, 2.0, dtype=np.float32)
    emb = np.zeros((10, 4), np.float16)
    emb[:, 0] = np.linspace(0, 1, 10)            # later frames more similar to the query
    np.savez(tmp_path / "v.npz", times=times, emb=emb)
    out = _feature_scan_times(tmp_path / "v.npz", np.array([1, 0, 0, 0], np.float32), cap=3, gap=4.0)
    assert out == [18.0, 14.0, 10.0]


def test_short_answer_verdict_parser():
    from videoqa.v3.factqa import parse_verdict

    assert parse_verdict("the colour matches.\nVerdict: CORRECT") is True
    assert parse_verdict("**Verdict:** incorrect") is False
    assert parse_verdict("Verdict: INCORRECT ... on reflection Verdict: CORRECT") is True
    assert parse_verdict("no verdict") is None
