"""CLI policy-spec parsing."""

import pytest

from videoqa.cli import parse_policy_spec


def test_policy_names_parse_to_policies():
    for name in ("transcript_only", "uniform", "retrieval", "scout_similarity", "heuristic"):
        assert parse_policy_spec(name).name == name


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        parse_policy_spec("learned=head.npz")


def test_open_ended_evaluate_hides_options(tmp_path):
    import json

    from videoqa.cli import main

    data, out = tmp_path / "syn", tmp_path / "eval"
    main(["make-synthetic", "--out", str(data), "--n", "2"])
    main(["evaluate", "--data", str(data), "--split", "train,dev,calibration,test", "--config", "configs/cpu.yaml",
          "--clean-only", "--open-ended", "--policies", "transcript_only", "--limit", "1", "--out", str(out)])
    recs = [json.loads(x) for x in (out / "records.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert recs and all(r["option"] is None for r in recs)      # no option letter without options
