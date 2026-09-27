"""V0/V1/V2 training on assay labels: plumbing, guards and calibration."""

import dataclasses

import numpy as np
import pytest

from videoqa import fixtures
from videoqa.answerer import FixtureAnswerer
from videoqa.assay import label_assay, write_assay
from videoqa.config import budget_from_config, load_config
from videoqa.heads import TrainConfig
from videoqa.scout import PixelStatsScout
from videoqa.train_assay import calibrate, fit_variant, simulate, train_assay

CFG = load_config("configs/cpu.yaml")


@pytest.fixture(scope="module")
def label_dirs(lecture_video, tmp_path_factory):
    items = [(dataclasses.replace(q, experiment_split="train"), str(lecture_video), fixtures.transcript())
             for q in fixtures.qa_items()]
    rows, pools, qs, st = label_assay(items, CFG, budget_from_config(CFG), PixelStatsScout(), FixtureAnswerer())
    root = tmp_path_factory.mktemp("assay")
    write_assay(root / "train", rows, pools, qs, st, {})
    # Plumbing only: dev is the same labels under different video ids.
    dev = [dataclasses.replace(r, video_id=r.video_id + "_dev", experiment_split="dev") for r in rows]
    write_assay(root / "dev", dev, pools, qs, st, {})
    return root, rows


def test_all_variants_train_and_calibrate(label_dirs, tmp_path):
    root, _ = label_dirs
    res = train_assay(root / "train", root / "dev", tmp_path, TrainConfig(epochs=80), price_per_s=0.1, seeds=(0,))
    assert set(res["variants"]) == {"V0", "V1", "V2"}
    for v in res["variants"].values():
        s = v[0]
        assert np.isfinite(s["final_train_loss"]) and s["lambda"] >= 0
        assert s["dev_at_lambda"]["all"]["utility"] <= res["reference"]["oracle_gain"]["all"]["utility"] + 1e-9


def test_train_dev_video_overlap_is_refused(label_dirs, tmp_path):
    root, _ = label_dirs
    with pytest.raises(ValueError):
        train_assay(root / "train", root / "train", tmp_path, TrainConfig(epochs=5), price_per_s=0.1, seeds=(0,))


def test_variants_differ_only_in_loss(label_dirs):
    _, rows = label_dirs
    from videoqa.assay import build_triplets
    trip, _ = build_triplets(rows)
    a, _ = fit_variant("V0", rows, trip, TrainConfig(epochs=1, seed=3))
    b, _ = fit_variant("V1", rows, trip, TrainConfig(epochs=1, seed=3))
    # same initialisation and standardisation; one step apart only by the loss
    assert np.allclose(a.mu, b.mu) and a.W1.shape == b.W1.shape


def test_stop_scores_zero_and_high_lambda_stops(label_dirs):
    _, rows = label_dirs
    pred = np.full(len(rows), 0.5)
    sim = simulate(pred, rows, lam=1e6, price=0.1)
    assert sim["all"]["act_rate"] == 0.0 and sim["all"]["utility"] == 0.0
    from videoqa.heads import UtilityHead
    head = UtilityHead(len(rows[0].features), 4, 0)
    lam, curve = calibrate(head, rows, 0.1)
    assert len(curve) > 5 and any(c["lambda"] == lam for c in curve)
