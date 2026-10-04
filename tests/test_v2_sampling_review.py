"""Regression checks for capped visual coverage and feature-cost accounting."""

import pytest

from videoqa.retrieval import Window
from videoqa.v2.candidates import scan_timestamps
from videoqa.v2.pilot import summarize


def test_balanced_scan_covers_late_windows_and_full_long_window_under_same_cap():
    windows = [Window("a", [], 0, 200, 1), Window("b", [], 500, 700, 5),
               Window("c", [], 900, 1100, 3), Window("d", [], 1500, 1700, 2)]
    old = scan_timestamps(windows, 1800, 5, 24, "legacy")
    new = scan_timestamps(windows, 1800, 5, 24, "balanced")
    assert len(old) == len(new) == 24
    assert max(old) == 115
    assert len(set(new)) == 24
    assert all(sum(w.start_s <= t < w.end_s for t in new) == 6 for w in windows)
    assert 195 in new and 1695 in new


def test_balanced_scan_small_budget_prioritizes_relevance_and_empty_inputs():
    windows = [Window("a", [], 0, 20, 1), Window("b", [], 60, 80, 5)]
    assert 60 <= scan_timestamps(windows, 100, 5, 1, "balanced")[0] < 80
    assert scan_timestamps([], 100, 5, 24, "balanced") == []
    assert scan_timestamps(windows, 100, 5, 0, "balanced") == []
    with pytest.raises(ValueError):
        scan_timestamps(windows, 100, 0, 24, "balanced")


def test_no_ocr_arm_still_pays_for_nearby_text_encoder():
    from types import SimpleNamespace

    pool = SimpleNamespace(timings={"retrieval": 1.0, "text_emb": 3.0, "ocr": 5.0})
    rows = [{"qa_id": "q", "video_id": "v", "A_clip": 0, "C_noocr_s0": 0, "C_ocr_s0": 0}]
    out = summarize(rows, {"q": pool}, {k: [0.0] for k in rows[0] if k not in ("qa_id", "video_id")},
                    [2.0], pairs=[], ocr_prefixes=("C_ocr",))
    lat = out["latency_s_median"]
    assert lat["C_noocr"] == lat["A_clip"] + 3
    assert lat["C_ocr"] == lat["C_noocr"] + 5
    assert "legacy combined" in out["latency_note"]
