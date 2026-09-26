"""Training + threshold tuning on fixture labels (tiny, CPU)."""

import dataclasses

import numpy as np

from videoqa import fixtures
from videoqa.answerer import FixtureAnswerer
from videoqa.config import budget_from_config, load_config
from videoqa.damage import make_triple
from videoqa.heads import TrainConfig, UtilityHead
from videoqa.labels import LabelRow, label_items, write_labels
from videoqa.schemas import DamageType
from videoqa.scout import PixelStatsScout
from videoqa.train import realised_net_gain, train_paired_and_unpaired, tune_threshold

CFG = load_config("configs/cpu.yaml")


def _row(qa, cond, gain, cid="c1"):
    return LabelRow(qa, "v", cond, 0, "look_at_this_moment", cid, 1.0, [1.0, gain], 0.0, gain, gain, 1.0, 64, False)


def test_realised_gain_counts_stop_as_zero():
    rows = [_row("q", "clean", 0.0, "c1"), _row("q", "clean", 0.0, "c2")]
    head = UtilityHead(2, 4, 0)
    assert realised_net_gain(head, rows, threshold=10.0, look_cost=0.02) == 0.0


def test_threshold_prefers_stopping_when_nothing_helps():
    rows = [_row(f"q{i}", "clean", 0.0) for i in range(5)]
    head = UtilityHead(2, 4, 0)
    t, v = tune_threshold(head, rows, look_cost=0.02)
    assert v == 0.0     # best achievable is to stop


def test_train_paired_and_unpaired_end_to_end(lecture_video, tmp_path):
    budget = dataclasses.replace(budget_from_config(CFG), max_frames=8)
    items = []
    for name in ("q_lr", "q_acc"):
        qa = {q.qa_id.split("/")[-1]: q for q in fixtures.qa_items()}[name]
        qa = dataclasses.replace(qa, source_split="train")
        triple = make_triple(qa, fixtures.transcript(), dtype=DamageType.DELETE, seed=0)
        if triple:
            items.append((qa, str(lecture_video), triple))
    rows, ex, _ = label_items(items, CFG, budget, PixelStatsScout(), FixtureAnswerer())
    write_labels(tmp_path / "train", rows, ex, {})
    write_labels(tmp_path / "dev", rows, ex, {})   # same rows: this test checks plumbing only
    res = train_paired_and_unpaired(tmp_path / "train", tmp_path / "dev", tmp_path / "ckpt",
                                    TrainConfig(epochs=50))
    assert set(res) == {"paired", "unpaired"}
    assert res["unpaired"]["train_config"]["lambda_pair"] == 0.0
    assert res["paired"]["train_config"]["lambda_pair"] == 1.0
    assert (tmp_path / "ckpt" / "head_paired.npz").exists()
    assert np.isfinite(res["paired"]["final_train_loss"])
