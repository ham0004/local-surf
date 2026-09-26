"""Configuration loading.

Configs are small YAML files in ``configs/``.  A config may ``inherit`` from
another file; child keys override parent keys (deep merge).  Keeping hardware
profiles as separate files makes it obvious which numbers were used for a run,
and the resolved config is written into every run's output directory.

All numbers in the shipped configs are *starting hyperparameters*, not tuned or
measured optima.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import yaml

from .schemas import Budget


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` merged in recursively (neither mutated)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``inherit: other.yaml`` chains.

    ``inherit`` paths are relative to the file that declares them.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent_name = raw.pop("inherit", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        raw = _deep_merge(parent, raw)
    return raw


def budget_from_config(cfg: dict[str, Any]) -> Budget:
    """Build a :class:`Budget` from the ``budget`` section, ignoring unknown keys
    so that old configs keep loading after new fields are added."""
    section = cfg.get("budget", {})
    known = {f.name for f in dataclasses.fields(Budget)}
    return Budget(**{k: v for k, v in section.items() if k in known})
