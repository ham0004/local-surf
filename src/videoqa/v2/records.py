"""Plain data records shared by the v2 stages.

Kept separate from the stage modules so that every stage depends only on these
small types, not on each other (makes each stage testable in isolation).
"""

from __future__ import annotations

import dataclasses
from typing import Any


# ---------------------------------------------------------------------------
# A candidate frame in the merged pool
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FrameCandidate:
    """One frame the selector may choose.

    ``paths`` records which proposal path(s) found it ("A" = hot moment from the
    transcript scorer, "B" = visual scan). ``image`` is the decoded PIL image
    (never serialised); ``digest`` is its exact pixel identity, used in cache keys.
    """

    id: str                       # stable within one question, e.g. "q12:f03"
    time_s: float                 # decoded presentation timestamp
    paths: tuple[str, ...]        # ("A",), ("B",) or ("A", "B")
    digest: str                   # sha256 of the exact pixels
    phash: str                    # perceptual hash (near-duplicate detection)
    clip_sim: float = 0.0         # MobileCLIP question-frame cosine (raw, not rescaled)
    head_a_text: float = 0.0      # Head A text-usefulness score of the nearest segment
    head_a_visual: float = 0.0    # Head A frame-usefulness score of the nearest segment
    ocr_text: str = ""            # text visible on the frame (RapidOCR)
    near_text: str = ""           # transcript within +/- 15 s of the frame
    # Cached frozen embeddings (MobileCLIP space); never serialised to JSON.
    emb: Any = dataclasses.field(default=None, repr=False, compare=False)
    ocr_emb: Any = dataclasses.field(default=None, repr=False, compare=False)
    near_emb: Any = dataclasses.field(default=None, repr=False, compare=False)
    image: Any = dataclasses.field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Everything known about one question before selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QuestionPool:
    """The fixed evidence for one question: retained transcript + candidate pool.

    Every selection arm sees exactly this object, so arms differ only in WHICH
    frames they pick (the experimental control the pilot relies on).
    """

    qa_id: str
    video_id: str
    question: str
    options: list[str]
    gold_option_index: int
    transcript: list[Any]         # retained (fixed) TranscriptSegment list, in time order
    candidates: list[FrameCandidate]
    duration_s: float = 0.0
    question_emb: Any = dataclasses.field(default=None, repr=False, compare=False)
    timings: dict = dataclasses.field(default_factory=dict)   # measured seconds per stage

    @property
    def transcript_text(self) -> str:
        return " ".join(s.text for s in self.transcript)


# ---------------------------------------------------------------------------
# One supervised training row for Head B
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UtilityRow:
    """utility(candidate | context) = R(S + {candidate}) - R(S).

    ``history`` is the ordered list of already-selected candidate ids (the
    context S). ``gain`` is in {-1, 0, +1} for binary correctness. ``before`` and
    ``after`` keep the raw qualities so "already correct" and "still wrong" zero
    labels can be told apart. ``scheme`` is "independent" or "prefix".
    """

    qa_id: str
    video_id: str
    candidate_id: str
    history: list[str]
    before: float
    after: float
    gain: float
    scheme: str
