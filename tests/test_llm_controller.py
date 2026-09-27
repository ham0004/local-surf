"""LLM controller guards that need no model download or GPU."""

import dataclasses

import pytest

from videoqa import fixtures
from videoqa.answerer import FixtureAnswerer
from videoqa.assay import label_assay, write_assay
from videoqa.config import budget_from_config, load_config
from videoqa.llm_controller import LLMTrainConfig, train_llm
from videoqa.scout import PixelStatsScout

CFG = load_config("configs/cpu.yaml")


def _dirs(video, tmp_path, strip_text=False):
    qa = [q for q in fixtures.qa_items() if q.qa_id.endswith("q_lr")][0]
    items = [(dataclasses.replace(qa, experiment_split="train"), str(video), fixtures.transcript())]
    rows, pools, qs, st = label_assay(items, CFG, budget_from_config(CFG), PixelStatsScout(), FixtureAnswerer())
    if strip_text:
        rows = [dataclasses.replace(r, obs_text="") for r in rows]
    write_assay(tmp_path / "train", rows, pools, qs, st, {})
    dev = [dataclasses.replace(r, video_id=r.video_id + "_dev", experiment_split="dev") for r in rows]
    write_assay(tmp_path / "dev", dev, pools, qs, st, {})
    return tmp_path / "train", tmp_path / "dev"


def test_refuses_labels_without_serialized_text(lecture_video, tmp_path):
    tr, dv = _dirs(lecture_video, tmp_path, strip_text=True)
    with pytest.raises(ValueError, match="obs_text"):
        train_llm(tr, dv, tmp_path / "out", LLMTrainConfig(steps=1), price_per_s=0.1, device="cpu")


def test_refuses_train_dev_overlap(lecture_video, tmp_path):
    tr, _ = _dirs(lecture_video, tmp_path)
    with pytest.raises(ValueError, match="share videos"):
        train_llm(tr, tr, tmp_path / "out", LLMTrainConfig(steps=1), price_per_s=0.1, device="cpu")
