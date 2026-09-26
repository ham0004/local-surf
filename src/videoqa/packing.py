"""Pack a short, verbatim, timestamped transcript excerpt for the answer VLM.

Goals:
* Stay under a word budget (a cheap proxy for text tokens).
* Keep text VERBATIM - never paraphrase - so numbers, negations and equations
  survive exactly as spoken (an abstractive summary can silently flip a sign).
* Keep each line's timestamp and segment id so the answer can cite it.
* Prefer segments inside retrieved windows; within those, prefer segments that
  contain "protected" content (numbers, negations) and share terms with the
  question.

Dependency closure (keeping a definition that a later line refers to) is
approximated by always including one preceding segment for each selected
segment when budget allows.  This is a simple rule, not a guarantee.
"""

from __future__ import annotations

import re

from .retrieval import Window, tokenize
from .schemas import TranscriptSegment

_NEGATION = re.compile(r"\b(not|no|never|without|none|cannot|isn't|doesn't|don't|won't)\b", re.I)
_NUMBER = re.compile(r"\d")


def _priority(seg: TranscriptSegment, q_tokens: set[str]) -> float:
    """Higher = more worth keeping.  Overlap with the question dominates;
    numbers and negations are protected because they are cheap to keep and
    catastrophic to lose."""
    overlap = len(q_tokens & set(tokenize(seg.text)))
    protected = bool(_NUMBER.search(seg.text)) + bool(_NEGATION.search(seg.text))
    return overlap * 2.0 + protected


def pack_excerpt(question: str, segments: list[TranscriptSegment], windows: list[Window],
                 unit_to_segments: dict[str, list[str]], max_words: int = 120,
                 extra_segment_ids: list[str] | None = None) -> list[TranscriptSegment]:
    """Choose segments for the answer prompt, returned in time order.

    ``extra_segment_ids`` are segments added by EXPAND_TRANSCRIPT actions; they
    are always considered first.
    """
    by_id = {s.id: s for s in segments}
    ordered = sorted(segments, key=lambda s: s.start_s)
    position = {s.id: i for i, s in enumerate(ordered)}

    # Candidate pool: segments of retrieved windows (+ explicit expansions).
    pool_ids: list[str] = list(extra_segment_ids or [])
    for w in windows:
        for uid in w.unit_ids:
            pool_ids.extend(unit_to_segments.get(uid, []))
    pool = [by_id[i] for i in dict.fromkeys(pool_ids) if i in by_id]
    # If retrieval proposed nothing (no lexical overlap with the question), the
    # answerer must still see the speech: fall back to the whole transcript,
    # ranked by the same priority. Without this, a retrieval miss silently
    # became "no transcript at all" (measured: 35/40 real clips, before the
    # BM25 IDF fix), which also made every transcript-damage condition a no-op.
    if not pool:
        pool = list(ordered)

    q_tokens = set(tokenize(question))
    pool.sort(key=lambda s: -_priority(s, q_tokens))

    chosen: dict[str, TranscriptSegment] = {}
    used = 0

    def try_add(seg: TranscriptSegment) -> bool:
        nonlocal used
        n = len(seg.text.split())
        if seg.id in chosen or used + n > max_words:
            return False
        chosen[seg.id] = seg
        used += n
        return True

    for seg in pool:
        if try_add(seg):
            # Dependency closure approximation: keep the preceding line too.
            idx = position[seg.id]
            if idx > 0:
                try_add(ordered[idx - 1])
    return sorted(chosen.values(), key=lambda s: s.start_s)


def format_excerpt(segments: list[TranscriptSegment]) -> str:
    """Render as '[s00002 16.0-22.0s] verbatim text' lines for the prompt."""
    return "\n".join(f"[{s.id} {s.start_s:.1f}-{s.end_s:.1f}s] {s.text}" for s in segments)
