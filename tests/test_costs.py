"""Cost meter must sum counts, max memory, and never drop a stage."""

from videoqa.costs import CostMeter, normalised_cost
from videoqa.schemas import CostRecord


def test_meter_sums_counts_and_maxes_memory():
    meter = CostMeter()
    with meter.stage("decode") as r:
        r.decoded_frames = 3
    with meter.stage("decode") as r:
        r.decoded_frames = 2
        r.visual_tokens = 100
    total = meter.total()
    assert total.decoded_frames == 5
    assert total.visual_tokens == 100
    assert total.peak_ram_bytes > 0
    assert meter.by_stage()["decode"].decoded_frames == 5


def test_stage_is_recorded_even_if_block_raises():
    meter = CostMeter()
    try:
        with meter.stage("answer"):
            raise RuntimeError("model crashed")
    except RuntimeError:
        pass
    assert [r.stage for r in meter.records] == ["answer"]


def test_normalised_cost_combines_time_and_tokens():
    rec = CostRecord(stage="x", elapsed_ms=1000, visual_tokens=512)
    assert normalised_cost(rec, max_seconds=10, max_visual_tokens=1024) == 0.1 + 0.5
