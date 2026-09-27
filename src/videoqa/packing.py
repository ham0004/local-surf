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

import dataclasses
import re
from collections.abc import Callable

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


@dataclasses.dataclass
class PackResult:
    """What was packed, and what an EXPAND request actually achieved."""

    segments: list[TranscriptSegment]
    requested_kept: list[str]        # explicitly requested segment ids that made it in
    requested_dropped: list[str]     # requested but over budget
    words: int
    tokens: int | None               # None when no tokenizer was supplied


def pack(question: str, segments: list[TranscriptSegment], windows: list[Window],
         unit_to_segments: dict[str, list[str]], max_words: int = 120,
         extra_segment_ids: list[str] | None = None, max_tokens: int | None = None,
         count_tokens: Callable[[str], int] | None = None) -> PackResult:
    """Choose segments for the answer prompt, returned in time order.

    Order of admission:
      1. ``extra_segment_ids`` (added by EXPAND_TRANSCRIPT), in the order given.
         They used to enter the same priority sort as everything else and could
         be silently displaced; now an expansion either gets in or is reported
         in ``requested_dropped``.
      2. Segments of retrieved windows by priority (question words, then
         numbers and negations), each followed by its preceding line.
      3. If retrieval found nothing: the whole transcript by the same priority.

    Budgets: ``max_words`` always; ``max_tokens`` too when ``count_tokens``
    (the answer model's own tokenizer) is supplied. Text is never paraphrased.
    """
    by_id = {s.id: s for s in segments}
    ordered = sorted(segments, key=lambda s: s.start_s)
    position = {s.id: i for i, s in enumerate(ordered)}
    requested = [i for i in dict.fromkeys(extra_segment_ids or []) if i in by_id]

    pool_ids: list[str] = []
    for w in windows:
        for uid in w.unit_ids:
            pool_ids.extend(unit_to_segments.get(uid, []))
    pool = [by_id[i] for i in dict.fromkeys(pool_ids) if i in by_id]
    # If retrieval proposed nothing, the answerer must still see the speech
    # (measured: 35/40 real clips once got NO transcript through this path).
    if not pool:
        pool = list(ordered)
    q_tokens = set(tokenize(question))
    pool.sort(key=lambda s: -_priority(s, q_tokens))

    chosen: dict[str, TranscriptSegment] = {}
    used_words = 0
    used_tokens = 0

    def try_add(seg: TranscriptSegment) -> bool:
        nonlocal used_words, used_tokens
        if seg.id in chosen:
            return False
        n = len(seg.text.split())
        k = count_tokens(seg.text) if count_tokens else 0
        if used_words + n > max_words or (max_tokens is not None and count_tokens and used_tokens + k > max_tokens):
            return False
        chosen[seg.id] = seg
        used_words += n
        used_tokens += k
        return True

    kept = [i for i in requested if try_add(by_id[i])]
    for seg in pool:
        if try_add(seg):
            idx = position[seg.id]           # dependency closure approximation:
            if idx > 0:                      # keep the preceding line too
                try_add(ordered[idx - 1])
    return PackResult(segments=sorted(chosen.values(), key=lambda s: s.start_s), requested_kept=kept,
                      requested_dropped=[i for i in requested if i not in kept], words=used_words,
                      tokens=used_tokens if count_tokens else None)


def pack_excerpt(question: str, segments: list[TranscriptSegment], windows: list[Window],
                 unit_to_segments: dict[str, list[str]], max_words: int = 120,
                 extra_segment_ids: list[str] | None = None) -> list[TranscriptSegment]:
    """Backward-compatible wrapper: just the packed segments."""
    return pack(question, segments, windows, unit_to_segments, max_words, extra_segment_ids).segments


def format_excerpt(segments: list[TranscriptSegment]) -> str:
    """Render as '[s00002 16.0-22.0s] verbatim text' lines for the prompt."""
    return "\n".join(f"[{s.id} {s.start_s:.1f}-{s.end_s:.1f}s] {s.text}" for s in segments)
