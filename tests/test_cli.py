"""CLI policy-spec parsing (threshold overrides for accuracy-vs-cost sweeps)."""

import numpy as np

from videoqa.cli import parse_policy_spec
from videoqa.features import CANDIDATE_FEATURES
from videoqa.heads import TrainConfig, UtilityHead


def test_simple_and_learned_specs(tmp_path):
    head = UtilityHead(len(CANDIDATE_FEATURES), 4, 0)
    head.fit(np.ones((3, len(CANDIDATE_FEATURES))), np.zeros(3), np.zeros((0, 2), int), TrainConfig(epochs=1))
    path = tmp_path / "head_paired.npz"
    head.save(path, stop_threshold=0.8)
    assert parse_policy_spec("uniform").name == "uniform"
    learned = parse_policy_spec(f"learned={path}")
    assert learned.name == "learned[head_paired]" and learned.stop_threshold == 0.8
    swept = parse_policy_spec(f"learned={path}@0.25")
    assert swept.name == "learned[head_paired]@0.25" and swept.stop_threshold == 0.25
