"""Frame-selection strategies behind one interface (Strategy pattern).

Every selector receives the SAME QuestionPool and the same budget k, and returns
k candidates. Arms therefore differ only in how they choose, which is what the
pilot compares.

    SimilarityTopK   A: highest MobileCLIP question-frame similarity
    MMRSelector      B: similarity minus redundancy with already-chosen frames
    UnaryUtility     C: a trained head scores every candidate once (empty history), top-k
    GreedyUtility    D: a trained head re-scores the remaining candidates after EVERY pick
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .head_b import featurize
from .records import FrameCandidate, QuestionPool


class Selector(Protocol):
    name: str

    def select(self, pool: QuestionPool, k: int) -> list[FrameCandidate]: ...


class SimilarityTopK:
    name = "A_mobileclip_topk"

    def select(self, pool, k):
        return sorted(pool.candidates, key=lambda c: -c.clip_sim)[:k]


class MMRSelector:
    """Maximal marginal relevance: lam * similarity - (1 - lam) * max similarity to chosen frames."""

    name = "B_mobileclip_mmr"

    def __init__(self, lam: float = 0.7) -> None:
        self.lam = lam

    def select(self, pool, k):
        chosen, rest = [], list(pool.candidates)
        while rest and len(chosen) < k:
            def score(c):
                red = max((float(c.emb @ h.emb) for h in chosen), default=0.0)
                return self.lam * c.clip_sim - (1 - self.lam) * red
            best = max(rest, key=score)
            chosen.append(best)
            rest.remove(best)
        return chosen


class UnaryUtility:
    """Score each candidate once with an empty history; take the top k."""

    def __init__(self, head, name: str = "C_independent_utility", use_ocr: bool = True,
                 use_transcript: bool = True) -> None:
        self.head, self.name, self.use_ocr, self.use_transcript = head, name, use_ocr, use_transcript

    def select(self, pool, k):
        X = np.stack([featurize(pool, c, [], self.use_ocr, self.use_transcript) for c in pool.candidates])
        order = np.argsort(-self.head.predict(X))
        return [pool.candidates[i] for i in order[:k]]


class GreedyUtility:
    """Pick the best candidate, add it to the history, re-score the rest, repeat."""

    def __init__(self, head, name: str = "D_history_utility", use_ocr: bool = True) -> None:
        self.head, self.name, self.use_ocr = head, name, use_ocr

    def select(self, pool, k):
        chosen, rest = [], list(pool.candidates)
        while rest and len(chosen) < k:
            X = np.stack([featurize(pool, c, chosen, self.use_ocr) for c in rest])
            best = rest[int(np.argmax(self.head.predict(X)))]
            chosen.append(best)
            rest.remove(best)
        return chosen


class ScoreTopK:
    """Top-k by any precomputed per-candidate score (used for Head A credit arms).

    ``score_fn(pool) -> {candidate_id: score}``.
    """

    def __init__(self, score_fn, name: str) -> None:
        self.score_fn, self.name = score_fn, name

    def select(self, pool, k):
        scores = self.score_fn(pool)
        return sorted(pool.candidates, key=lambda c: -scores[c.id])[:k]


def timed_select(selector, pool, k) -> tuple[list[FrameCandidate], float]:
    """Run a selector and return (choice, selection seconds)."""
    t0 = time.perf_counter()
    out = selector.select(pool, k)
    return out, time.perf_counter() - t0
