"""One cost definition, used everywhere a decision trades answer gain for work.

Before 2026-09-28 the code used three different costs: a flat 0.02 per frame in
the online controller, 0.005 per transcript expansion online but 0 in label
preferences and threshold tuning, and a per-candidate SHARE of batch decode
time in the logged "measured" cost. This module replaces all of them.

Action costs come from single-action measurements on the target device
(``scripts/profile_costs.py`` -> ``configs/cost_model_<device>.yaml``) and
depend on what is already cached:

  look(frame)   = decode (if not yet decoded) + scout (if not yet scouted and
                  the policy scouts) + prefill_per_visual_token * tokens(frame)
  expand(text)  = prefill_per_text_token * tokens(text)
  stop          = 0  (the final answer is paid by every policy either way)

A decision scores each legal action as

  net(a) = predicted_gain(a) - lambda_per_s * cost_ms(a) / 1000

and STOPs when no action has net > 0. ``lambda_per_s`` (answer-quality units
per second of work) is the ONE trade-off knob; it is calibrated on
development data and swept to draw the quality-cost (Pareto) curve.
Predicted gain and cost stay separate so the same gain model can be used on
another device with that device's cost file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class CostModel:
    decode_ms_per_frame: float
    scout_ms_per_frame: float
    scout_neighbour_decode_ms: float
    prefill_ms_per_visual_token: float
    prefill_ms_per_text_token: float
    lambda_per_s: float = 0.1
    measured: bool = False            # False = placeholder numbers (tests / CPU double)
    source: str = "defaults"

    def look_ms(self, visual_tokens: int, decoded: bool, scouted: bool, will_scout: bool = False) -> float:
        """Marginal cost of promoting one frame to the answerer."""
        ms = 0.0 if decoded else self.decode_ms_per_frame
        if will_scout and not scouted:
            ms += self.scout_ms_per_frame + self.scout_neighbour_decode_ms
        return ms + self.prefill_ms_per_visual_token * visual_tokens

    def scout_ms(self, decoded: bool) -> float:
        """Cost of one scout observation (frame decode if needed + neighbour + encoder)."""
        return (0.0 if decoded else self.decode_ms_per_frame) + self.scout_neighbour_decode_ms + self.scout_ms_per_frame

    def expand_ms(self, text_tokens: int) -> float:
        return self.prefill_ms_per_text_token * text_tokens

    def net(self, gain: float, cost_ms: float, lambda_per_s: float | None = None) -> float:
        lam = self.lambda_per_s if lambda_per_s is None else lambda_per_s
        return gain - lam * cost_ms / 1000.0

    def with_lambda(self, lambda_per_s: float) -> CostModel:
        return dataclasses.replace(self, lambda_per_s=lambda_per_s)


# Placeholder numbers for tests and the CPU fixture path, in the same units.
# NOT measurements; ``measured`` is False so reports can say so.
DEFAULT = CostModel(decode_ms_per_frame=40.0, scout_ms_per_frame=45.0, scout_neighbour_decode_ms=18.0,
                    prefill_ms_per_visual_token=0.36, prefill_ms_per_text_token=0.19)


def load_cost_model(path: str | Path | None, lambda_per_s: float | None = None) -> CostModel:
    """Read a measured cost file; fall back to DEFAULT when no path is given."""
    if not path:
        cm = DEFAULT
    else:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cost_model"]
        cm = CostModel(decode_ms_per_frame=raw["decode_ms_per_frame"], scout_ms_per_frame=raw["scout_ms_per_frame"],
                       scout_neighbour_decode_ms=raw["scout_neighbour_decode_ms"],
                       prefill_ms_per_visual_token=raw["prefill_ms_per_visual_token"],
                       prefill_ms_per_text_token=raw["prefill_ms_per_text_token"], measured=True, source=str(path))
    return cm.with_lambda(lambda_per_s) if lambda_per_s is not None else cm


def cost_model_from_config(cfg: dict) -> CostModel:
    section = cfg.get("cost_model", {}) or {}
    return load_cost_model(section.get("path"), section.get("lambda_per_s"))
