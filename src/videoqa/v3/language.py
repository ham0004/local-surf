"""Is a transcript really English? (EduVidQA caption tracks can be mislabelled.)

NPTEL uploads carry subtitle tracks in several Indian languages. In the 20-video sample,
6 tracks labelled "en" were in fact Tamil, which silently turned "transcript only" into
"almost no usable transcript" for those videos. The check is by script, not by label:
share of alphabetic characters that are ASCII Latin letters.
"""

from __future__ import annotations

ENGLISH_MIN_LATIN_SHARE = 0.9


def latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isascii() for ch in letters) / len(letters)


def is_english_transcript(segments) -> bool:
    """True when at least 90% of the letters in the segments' text are Latin script."""
    return latin_share(" ".join(s.text for s in segments)) >= ENGLISH_MIN_LATIN_SHARE
