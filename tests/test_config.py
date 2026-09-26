"""Config inheritance must deep-merge so profiles only list what they change."""

from pathlib import Path

from videoqa.config import budget_from_config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_cpu_profile_inherits_and_overrides():
    cfg = load_config(CONFIGS / "cpu.yaml")
    assert cfg["device"] == "cpu"
    # overridden in cpu.yaml
    assert cfg["budget"]["max_frames"] == 3
    # inherited unchanged from base.yaml
    assert cfg["budget"]["max_rounds"] == 6
    assert cfg["retrieval"]["bm25_top_k"] == 64


def test_every_profile_builds_a_budget():
    for name in ("base.yaml", "cpu.yaml", "gpu_8gb.yaml", "gpu_12gb.yaml"):
        budget = budget_from_config(load_config(CONFIGS / name))
        assert budget.max_frames > 0
