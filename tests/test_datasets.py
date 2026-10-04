"""Local dataset layout: loading, hashed splits, and missing-file reporting."""

import json

import pytest

from videoqa import fixtures
from videoqa.datasets import load_local_dataset


def test_load_synthetic_dataset_assigns_splits_and_reports_missing(tmp_path):
    fixtures.write_dataset(tmp_path, n_videos=3, seed0=200)
    (tmp_path / "videos" / "synth_0201.mp4").unlink()            # simulate a dead link
    items, report = load_local_dataset(tmp_path)
    assert report.loaded == 6
    assert len(report.missing_video) == 3                          # 3 questions of the missing video
    assert all(it.qa.source_split in {"train", "dev", "calibration", "test"} for it in items)
    assert all(len(it.transcript.segments) == 11 for it in items)


def test_split_filter(tmp_path):
    fixtures.write_dataset(tmp_path, n_videos=4, seed0=300)
    all_items, _ = load_local_dataset(tmp_path)
    some = {all_items[0].qa.source_split}
    filtered, _ = load_local_dataset(tmp_path, splits=tuple(some))
    assert filtered and all(it.qa.source_split in some for it in filtered)


def test_official_split_is_preserved_separately_from_experiment_split(tmp_path):
    import json
    fixtures.write_dataset(tmp_path, n_videos=1, seed0=500)
    rows = [json.loads(line) for line in (tmp_path / "qa.jsonl").read_text().splitlines()]
    for r in rows:
        r["official_split"] = "validation"
    (tmp_path / "qa.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items, _ = load_local_dataset(tmp_path)
    for it in items:
        assert it.qa.official_split == "validation"
        assert it.qa.experiment_split in {"train", "dev", "calibration", "test"}
        assert it.qa.source_split == it.qa.experiment_split


def test_leakage_is_checked_before_split_filtering(tmp_path):
    rows = [{"qa_id": "a", "video_id": "v", "question": "q", "gold_answer": "x", "experiment_split": "train"},
            {"qa_id": "b", "video_id": "v", "question": "q", "gold_answer": "x", "experiment_split": "test"}]
    (tmp_path / "qa.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="appears in both"):
        load_local_dataset(tmp_path, splits=("train",))     # one split alone would hide the shared video
