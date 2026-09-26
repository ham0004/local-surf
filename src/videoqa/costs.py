"""Cost metering.

Every expensive step (decode, scout, controller, answerer) runs inside
``meter.stage(...)`` so its wall time, frame counts, tokens and memory are
recorded.  The research claim is about accuracy *per unit of real cost*, so
nothing expensive may run un-metered - the pipeline tests check that the
trace accounts for every decoded frame.

Memory numbers:
  * peak RAM   = process resident set size sampled at stage end (psutil).
                 This is a coarse upper bound, not a per-stage allocation.
  * peak VRAM  = torch.cuda.max_memory_allocated() since the stage started,
                 only if torch + CUDA are available.  Allocator-reserved memory
                 and other processes are NOT included.
"""

from __future__ import annotations

import contextlib
import sys
import dataclasses
import time
from collections.abc import Iterator

import psutil

from .schemas import CostRecord


def _torch_cuda():
    """Return torch if it is ALREADY imported and CUDA is initialised, else None.

    We deliberately do not import torch here: importing it (and initialising
    CUDA) would add seconds of overhead to CPU-only runs and would itself show
    up in the cost trace.  If a GPU model is in use, torch is already loaded.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return torch


class CostMeter:
    """Accumulates :class:`CostRecord` entries for one question."""

    def __init__(self) -> None:
        self.records: list[CostRecord] = []
        self._proc = psutil.Process()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[CostRecord]:
        """Time a block and capture memory.  The caller fills in counts
        (decoded_frames, visual_tokens, ...) on the yielded record."""
        torch = _torch_cuda()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        record = CostRecord(stage=name)
        t0 = time.perf_counter()
        try:
            yield record
        finally:
            record.elapsed_ms += (time.perf_counter() - t0) * 1000.0
            record.peak_ram_bytes = max(record.peak_ram_bytes, self._proc.memory_info().rss)
            if torch is not None:
                record.peak_vram_bytes = max(record.peak_vram_bytes, int(torch.cuda.max_memory_allocated()))
            self.records.append(record)

    def total(self) -> CostRecord:
        """Sum counts and time across stages; memory is the max, not the sum."""
        out = CostRecord(stage="total")
        for r in self.records:
            out.elapsed_ms += r.elapsed_ms
            out.decoded_frames += r.decoded_frames
            out.scored_frames += r.scored_frames
            out.selected_frames += r.selected_frames
            out.visual_tokens += r.visual_tokens
            out.text_tokens += r.text_tokens
            out.peak_ram_bytes = max(out.peak_ram_bytes, r.peak_ram_bytes)
            out.peak_vram_bytes = max(out.peak_vram_bytes, r.peak_vram_bytes)
        return out

    def by_stage(self) -> dict[str, CostRecord]:
        """Aggregate records that share a stage name (e.g. several 'decode' calls)."""
        out: dict[str, CostRecord] = {}
        for r in self.records:
            agg = out.setdefault(r.stage, CostRecord(stage=r.stage))
            for f in dataclasses.fields(CostRecord):
                if f.name == "stage":
                    continue
                a, b = getattr(agg, f.name), getattr(r, f.name)
                setattr(agg, f.name, max(a, b) if f.name.startswith("peak") else a + b)
        return out


def normalised_cost(record: CostRecord, max_seconds: float, max_visual_tokens: int,
                    token_weight: float = 1.0) -> float:
    """Scalar cost used only to RANK actions (utility / cost).

    c = elapsed / time_budget + token_weight * visual_tokens / token_budget

    Hard caps are enforced separately; this number never excuses exceeding one.
    """
    return (record.elapsed_ms / 1000.0) / max(max_seconds, 1e-9) + token_weight * (
        record.visual_tokens / max(max_visual_tokens, 1)
    )
