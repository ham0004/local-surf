"""CLI policy-spec parsing."""

import pytest

from videoqa.cli import parse_policy_spec


def test_policy_names_parse_to_policies():
    for name in ("transcript_only", "uniform", "retrieval", "scout_similarity", "heuristic"):
        assert parse_policy_spec(name).name == name


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        parse_policy_spec("learned=head.npz")
