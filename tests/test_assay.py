"""Controlled assay labels: common pool, exact pair identity, split guards."""

import dataclasses

import pytest

from videoqa import fixtures
from videoqa.answerer import FixtureAnswerer
from videoqa.assay import (
    build_triplets,
    check_media_overlap,
    check_split,
    label_assay,
    read_assay,
    write_assay,
)
from videoqa.config import budget_from_config, load_config
from videoqa.schemas import DamageType
from videoqa.scout import PixelStatsScout

CFG = load_config("configs/cpu.yaml")


def _items(video, split="train"):
    qa = [q for q in fixtures.qa_items() if q.qa_id.endswith("q_lr")][0]
    return [(dataclasses.replace(qa, experiment_split=split), str(video), fixtures.transcript())]


@pytest.fixture(scope="module")
def assay(lecture_video):
    return label_assay(_items(lecture_video), CFG, budget_from_config(CFG), PixelStatsScout(), FixtureAnswerer(),
                       dtype=DamageType.DELETE)


def test_every_condition_sees_the_same_pool_and_pairs_are_exact(assay):
    rows, pools, questions, stats = assay
    assert questions and rows
    triplets, drops = build_triplets(rows)
    assert triplets, drops
    for t in triplets:
        c, tg, ct = rows[t["clean"]], rows[t["targeted"]], rows[t["control"]]
        assert c.pixel_digest == tg.pixel_digest == ct.pixel_digest
        assert c.decoded_pts_s == tg.decoded_pts_s == ct.decoded_pts_s
        assert c.history_fp == tg.history_fp == ct.history_fp
        assert c.canonical_key == tg.canonical_key == ct.canonical_key


def test_expand_is_never_in_a_visual_triplet(assay):
    rows = assay[0]
    triplets, _ = build_triplets(rows)
    assert any(r.action_class == "expand" for r in rows)
    assert all(rows[t[k]].action_class == "look" for t in triplets for k in ("clean", "targeted", "control"))


def test_both_histories_are_labelled_and_traces_saved(assay):
    rows = assay[0]
    assert {r.history_id for r in rows} == {"h0", "h1"}
    assert all(isinstance(r.answer_after, str) for r in rows)
    # targeted damage removes the spoken answer, so some look must recover it
    assert any(r.condition == "targeted_damage" and r.gain > 0 for r in rows)


def test_features_never_contain_condition_or_gold(assay):
    rows = assay[0]
    # same frame, same history, CLEAN vs CONTROL: features may differ only
    # through the observable transcript, never through a condition flag
    from videoqa.features import CANDIDATE_FEATURES
    assert len(rows[0].features) == len(CANDIDATE_FEATURES)
    assert not any("condition" in n or "gold" in n or "damage" in n for n in CANDIDATE_FEATURES)


def test_test_split_is_refused(lecture_video):
    qa = _items(lecture_video, "test")[0][0]
    with pytest.raises(ValueError):
        check_split(qa)
    with pytest.raises(ValueError):
        check_split(dataclasses.replace(qa, experiment_split="unassigned"))


def test_media_overlap_across_splits_is_refused(lecture_video):
    a = _items(lecture_video, "train")[0][0]
    b = dataclasses.replace(a, qa_id="other", experiment_split="dev")
    with pytest.raises(ValueError):
        check_media_overlap([(a, str(lecture_video)), (b, str(lecture_video))])
    assert len(check_media_overlap([(a, str(lecture_video))])) == 1


def test_write_and_read_round_trip(assay, tmp_path):
    rows, pools, questions, stats = assay
    report = write_assay(tmp_path, rows, pools, questions, stats, {"note": "test"})
    back, triplets = read_assay(tmp_path)
    assert len(back) == len(rows) and report["triplets"] == len(triplets) and report["note"] == "test"
