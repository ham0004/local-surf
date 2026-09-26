"""Leakage-safe splits: assign ORIGINAL videos (or whole courses) to splits
BEFORE creating any transcript variants.

A question's clean / targeted / control transcripts all come from the same
video, so splitting at the question or variant level would put near-identical
evidence in train and test.  We therefore hash a *group key* (video id, or a
course/channel id when known) into a split.  Hashing is deterministic, needs no
global list, and never looks at answers.

Splits: train (label generation + fitting), dev (tuning, stop threshold),
calibration (held-out threshold/risk fitting), test (touched once).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .schemas import QAItem

DEFAULT_FRACTIONS = (("train", 0.6), ("dev", 0.15), ("calibration", 0.1), ("test", 0.15))


def split_of(group_key: str, fractions=DEFAULT_FRACTIONS, salt: str = "videoqa-v1") -> str:
    """Map a group key to a split via a stable hash in [0, 1)."""
    h = int(hashlib.sha256(f"{salt}:{group_key}".encode()).hexdigest()[:12], 16) / 16**12
    acc = 0.0
    for name, frac in fractions:
        acc += frac
        if h < acc:
            return name
    return fractions[-1][0]


def assign_splits(items: Iterable[QAItem], group_of=lambda qa: qa.video_id, **kw) -> dict[str, list[QAItem]]:
    """Group QA items by split.  ``group_of`` can return a course id instead of
    a video id for course-disjoint evaluation."""
    out: dict[str, list[QAItem]] = {}
    for qa in items:
        out.setdefault(split_of(group_of(qa), **kw), []).append(qa)
    return out


def check_no_leakage(splits: dict[str, list[QAItem]], group_of=lambda qa: qa.video_id) -> None:
    """Raise if any group appears in more than one split."""
    seen: dict[str, str] = {}
    for name, items in splits.items():
        for qa in items:
            g = group_of(qa)
            if seen.setdefault(g, name) != name:
                raise ValueError(f"group {g!r} appears in both {seen[g]!r} and {name!r}")
