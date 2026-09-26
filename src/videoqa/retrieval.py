"""Cheap temporal retrieval and candidate-time generation.

Stage 1 of the coarse-to-fine pipeline.  Nothing here looks at pixels:

    question --BM25 over transcript units--> ranked units
             --neighbour expansion + merge--> a few time WINDOWS
             --window start/mid/end + uniform coverage + rescue grid--> CANDIDATES

Candidates are just times.  The scout (scout.py) later attaches cheap visual
hints to them and the controller (controller.py) decides which to look at.

Why BM25 first?  It is fast, needs no model, and is a strong, transparent
baseline for lecture transcripts full of domain terms.  A dense retriever can
be fused in later via reciprocal rank fusion (``rrf_fuse``).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter

from .schemas import Candidate, CandidateSource
from .transcript import TranscriptUnit

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    "a an the of to in on at for and or but is are was were be it this that with as by "
    "what which who how why when where do does did".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; keeps decimals like 0.3 intact."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclasses.dataclass
class Window:
    """A retrieved stretch of video, traceable to the units that produced it."""

    id: str
    unit_ids: list[str]
    start_s: float
    end_s: float
    score: float
    provenance: str = "bm25"


def bm25_scores(query_tokens: list[str], corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 with the non-negative IDF  log(1 + (N - n + 0.5) / (n + 0.5)).

    Why not ``rank_bm25.BM25Okapi``?  Its classic IDF, log((N - n + 0.5)/(n + 0.5)),
    is NEGATIVE for any term present in more than half the documents.  Measured
    on real LongVideoBench clips: 29/40 videos form a single retrieval unit, so
    every term is in "all" documents and even a unit containing every query term
    scored -0.55; retrieval then found no window for 35/40 videos and the
    answerer silently received NO transcript.  With this IDF a unit scores > 0
    exactly when it shares at least one query term, for any corpus size.
    """
    n_docs = len(corpus)
    if n_docs == 0:
        return []
    avg_len = sum(len(d) for d in corpus) / n_docs or 1.0
    doc_freq = Counter(t for d in corpus for t in set(d))
    idf = {t: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}
    scores = []
    for doc in corpus:
        tf = Counter(doc)
        norm = k1 * (1.0 - b + b * len(doc) / avg_len)
        scores.append(sum(idf[t] * tf[t] * (k1 + 1.0) / (tf[t] + norm) for t in query_tokens if t in tf))
    return scores


def bm25_rank(question: str, units: list[TranscriptUnit], top_k: int = 64) -> list[tuple[TranscriptUnit, float]]:
    """Rank transcript units by BM25 score against the question.

    Units with a zero score are dropped: they share no terms with the question,
    so ranking among them would be arbitrary.
    """
    if not units:
        return []
    corpus = [tokenize(u.text) for u in units]
    scores = bm25_scores(tokenize(question), corpus)
    ranked = sorted(zip(units, scores, strict=True), key=lambda x: -x[1])
    return [(u, float(s)) for u, s in ranked[:top_k] if s > 0]


def rrf_fuse(rankings: list[list[str]], k0: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: score(id) = sum_r 1 / (k0 + rank_r(id)).

    Used to combine BM25 with a dense retriever without calibrating scores."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k0 + rank)
    return fused


def build_windows(ranked: list[tuple[TranscriptUnit, float]], units: list[TranscriptUnit],
                  max_windows: int = 4, neighbour_expansion: int = 1,
                  video_duration_s: float | None = None) -> list[Window]:
    """Turn top-ranked units into merged, clipped windows.

    Each anchor unit is expanded by ``neighbour_expansion`` units on each side
    (definitions often sit just before the sentence that matches the question).
    Overlapping windows are merged and keep the best score, and all windows are
    clipped to the video duration.
    """
    index = {u.id: i for i, u in enumerate(units)}
    raw: list[Window] = []
    for unit, score in ranked:
        i = index[unit.id]
        lo, hi = max(0, i - neighbour_expansion), min(len(units) - 1, i + neighbour_expansion)
        span = units[lo : hi + 1]
        raw.append(Window(id="", unit_ids=[u.id for u in span], start_s=span[0].start_s,
                          end_s=span[-1].end_s, score=score))

    # Merge overlaps (sorted by start time), keeping the max score.
    raw.sort(key=lambda w: w.start_s)
    merged: list[Window] = []
    for w in raw:
        if merged and w.start_s <= merged[-1].end_s:
            m = merged[-1]
            m.end_s = max(m.end_s, w.end_s)
            m.unit_ids = sorted(set(m.unit_ids) | set(w.unit_ids), key=lambda u: index[u])
            m.score = max(m.score, w.score)
        else:
            merged.append(dataclasses.replace(w, unit_ids=list(w.unit_ids)))

    merged.sort(key=lambda w: -w.score)
    merged = merged[:max_windows]
    for n, w in enumerate(sorted(merged, key=lambda w: w.start_s)):
        w.id = f"w{n}"
        if video_duration_s is not None:
            w.start_s = max(0.0, min(w.start_s, video_duration_s))
            w.end_s = max(w.start_s, min(w.end_s, video_duration_s))
    return sorted(merged, key=lambda w: w.start_s)


def _inside_any(t: float, windows: list[Window]) -> bool:
    return any(w.start_s <= t <= w.end_s for w in windows)


def generate_candidates(windows: list[Window], video_duration_s: float, per_window: int = 3,
                        uniform: int = 8, rescue: int = 4, min_separation_s: float = 1.0) -> list[Candidate]:
    """Propose candidate times from three sources.

    * TRANSCRIPT_RETRIEVAL: evenly spaced points inside each window
      (start / middle / end for per_window=3).
    * UNIFORM: evenly spaced coverage of the whole video, independent of speech.
    * GLOBAL_RESCUE: evenly spaced points OUTSIDE all windows, used by the
      LOOK_ELSEWHERE action for facts that are visible but never spoken.

    Candidates closer than ``min_separation_s`` to an earlier one are dropped
    (retrieval candidates win, because they come first).  All times are clipped
    to ``[0, video_duration_s)``.  A sparse rescue grid can still miss a brief
    silent event - this is a known ceiling, not a guarantee.
    """
    out: list[Candidate] = []
    last_t = max(0.0, video_duration_s - 1e-3)

    def add(t: float, source: CandidateSource, window_id: str | None = None) -> None:
        t = min(max(0.0, t), last_t)
        if all(abs(t - c.time_s) >= min_separation_s for c in out):
            out.append(Candidate(id=f"c{len(out):03d}", time_s=round(t, 3), source=source, window_id=window_id))

    for w in windows:
        for k in range(per_window):
            frac = 0.5 if per_window == 1 else k / (per_window - 1)
            add(w.start_s + frac * (w.end_s - w.start_s), CandidateSource.TRANSCRIPT_RETRIEVAL, w.id)

    for k in range(uniform):
        add((k + 0.5) * video_duration_s / uniform, CandidateSource.UNIFORM)

    # Rescue grid: a denser grid, keep only points outside retrieved windows.
    grid = [(k + 0.5) * video_duration_s / (rescue * 3) for k in range(rescue * 3)]
    outside = [t for t in grid if not _inside_any(t, windows)]
    step = max(1, len(outside) // max(rescue, 1))
    for t in outside[::step][:rescue]:
        add(t, CandidateSource.GLOBAL_RESCUE)
    return out
