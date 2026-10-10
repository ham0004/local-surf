"""Typed data contracts shared by every stage of the pipeline.

Why a separate module?
    The research hypothesis depends on *what information each component is allowed
    to see*.  Encoding that in types (rather than in comments scattered across the
    code) lets tests check the rules mechanically:

    * The visual scout may only emit numbers and a coarse category
      (:class:`ScoutSignals`) - never OCR text or captions that could answer the
      question.
    * The controller receives a :class:`ControllerObservation`, which has *no* field
      for the gold answer, the transcript-condition name, or measured action utility.
    * Training-only metadata (gold answer, damage record, utility labels) lives in
      :class:`TrainingExample` and :class:`DamageRecord`, which the controller's
      inference path never receives.

All records are plain dataclasses so they serialise to JSON with
:func:`to_jsonable` and stay easy to inspect in a debugger.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TranscriptCondition(str, enum.Enum):
    """The three transcript states used for paired training and evaluation.

    CLEAN            - original transcript, untouched.
    TARGETED_DAMAGE  - answer-relevant speech is removed or corrupted.
    CONTROL_DAMAGE   - the *same amount and type* of damage is applied to speech
                       that is NOT relevant to the answer (the matched control).

    The name of the condition is training/evaluation metadata only.  A deployed
    controller never sees it; it only sees the resulting (possibly damaged) text.
    """

    CLEAN = "clean"
    TARGETED_DAMAGE = "targeted_damage"
    CONTROL_DAMAGE = "control_damage"


class DamageType(str, enum.Enum):
    """How a transcript segment is damaged.  Targeted and control damage in the
    same pair always use the same type (checked by tests)."""

    DELETE = "delete"            # the segment disappears (missing subtitle)
    MASK_WORDS = "mask_words"    # content words replaced by an unintelligible token
    NUMBER_SWAP = "number_swap"  # digits / number words altered (common ASR error)
    TIME_SHIFT = "time_shift"    # text kept but timestamps shifted (misalignment)


class ActionKind(str, enum.Enum):
    """The controller's small, typed action vocabulary.

    LOOK_AT_THIS_MOMENT - decode + show the final VLM a frame at a candidate time.
    LOOK_ELSEWHERE      - bounded "global rescue": probe a time outside the
                          transcript-retrieved windows (for silent visual facts).
    EXPAND_TRANSCRIPT   - add neighbouring transcript context to the excerpt.
    STOP                - stop acquiring evidence and answer now.
    """

    LOOK_AT_THIS_MOMENT = "look_at_this_moment"
    LOOK_ELSEWHERE = "look_elsewhere"
    EXPAND_TRANSCRIPT = "expand_transcript"
    STOP = "stop"


class SceneType(str, enum.Enum):
    """Coarse visual category the scout may report.  Deliberately coarse: it tells
    the controller *what kind* of frame this is, not *what it says*."""

    SLIDE = "slide"              # mostly static, text/diagram-like layout
    BOARD = "board"              # handwriting / whiteboard
    CODE = "code"                # terminal / editor
    NATURAL = "natural"          # camera footage, people, objects
    BLANK = "blank"              # black / uniform frame
    UNKNOWN = "unknown"


class CandidateSource(str, enum.Enum):
    """Why a time was proposed (kept for provenance and ablations)."""

    TRANSCRIPT_RETRIEVAL = "transcript_retrieval"
    UNIFORM = "uniform"
    SCENE_CHANGE = "scene_change"
    GLOBAL_RESCUE = "global_rescue"


# ---------------------------------------------------------------------------
# Transcript records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TranscriptSegment:
    """One timed piece of speech.

    ``confidence`` is a raw ASR score when available.  It is a *feature*, never a
    calibrated probability (ASR confidences are notoriously over-confident).
    """

    id: str
    start_s: float
    end_s: float
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        # Reject impossible times early: a negative or reversed interval would
        # silently break retrieval windows and evidence citations later.
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)):
            raise ValueError(f"segment {self.id}: non-finite time")
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError(
                f"segment {self.id}: invalid interval [{self.start_s}, {self.end_s}]"
            )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclasses.dataclass
class Transcript:
    """An ordered list of segments plus provenance (subtitle file, ASR model...)."""

    video_id: str
    segments: list[TranscriptSegment]
    source: str = "unknown"  # e.g. "supplied_srt", "faster-whisper:small", "fixture"

    def by_id(self) -> dict[str, TranscriptSegment]:
        return {s.id: s for s in self.segments}

    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)


@dataclasses.dataclass
class DamageRecord:
    """TRAINING/EVAL METADATA describing how a transcript variant was produced.

    Never passed to the controller at inference.  Used by tests to verify that
    targeted and control damage are matched in amount and type.
    """

    condition: TranscriptCondition
    damage_type: DamageType | None       # None for CLEAN
    damaged_segment_ids: list[str]
    words_affected: int
    seed: int
    # Effective-edit record (what the operation actually changed), so matching
    # can be checked on severity, not just on the number of segments touched.
    changed_tokens: int = 0              # tokens deleted / masked / numbers altered / words moved
    time_coverage_s: float = 0.0         # total duration of the damaged segments
    intervals_s: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    changed: bool = False                # False = the operation was a no-op
    displacement_s: float = 0.0          # TIME_SHIFT only: mean effective shift after clamping
    role: str = ""                       # "targeted" | "control" | "control_near" | "control_distant"
    support_type: str = ""               # evidence type of the targeted span (see QAItem.evidence_type)


# ---------------------------------------------------------------------------
# Question / dataset records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QAItem:
    """One question about one video, as loaded from a dataset adapter.

    ``evidence_segment_ids`` and ``evidence_intervals_s`` are gold annotations or
    automatically derived answer-relevant spans.  They are used ONLY to build
    targeted damage on TRAIN videos and for evaluation metrics - never by the
    controller or answerer at inference.
    """

    qa_id: str
    video_id: str
    question: str
    gold_answer: str
    options: list[str] | None = None            # multiple-choice options, if any
    gold_option_index: int | None = None
    evidence_segment_ids: list[str] = dataclasses.field(default_factory=list)
    evidence_intervals_s: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    source_dataset: str = "unknown"
    source_split: str = "unknown"          # = experiment_split (kept for backward compatibility)
    provenance_and_license: str = "unknown"
    # What the evidence interval IS. A quoted subtitle that only tells you WHEN to
    # look is a "temporal_anchor", not speech containing the answer.
    #   spoken_answer | temporal_anchor | visual_support | inferred | unknown
    evidence_type: str = "unknown"
    official_split: str = "unknown"        # split in the dataset's own release (e.g. "validation")
    experiment_split: str = "unknown"      # our split (train / dev / calibration / test)
    is_synthetic: bool = False             # question or answer generated by us, not by the dataset


# ---------------------------------------------------------------------------
# Visual records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Candidate:
    """A time the controller may choose to look at."""

    id: str
    time_s: float
    source: CandidateSource
    window_id: str | None = None   # transcript window that proposed it, if any
    rank: int | None = None        # relevance rank of that window (0 = best match); None if not retrieved


@dataclasses.dataclass
class Frame:
    """A decoded frame.  ``decoded_pts_s`` is the real presentation timestamp of
    the frame we actually got, which can differ from ``requested_s`` (keyframes,
    variable frame rate).  Citations always use ``decoded_pts_s``."""

    id: str
    requested_s: float
    decoded_pts_s: float
    width: int
    height: int
    image: Any = dataclasses.field(default=None, repr=False, compare=False)  # PIL.Image
    phash: str = ""  # perceptual hash used for cheap dedup
    digest: str = ""  # sha256 of the exact pixels: the frame's identity in paired actions
    video_id: str = ""  # provenance: which video this frame was decoded from


# The ONLY keys a scout is allowed to emit.  Anything else (OCR text, captions,
# object names...) could leak the answer to the controller.  See ScoutSignals.
ALLOWED_SCOUT_FIELDS = frozenset(
    {"similarity", "scene_type", "visual_change", "quality", "uncertainty"}
)


@dataclasses.dataclass(frozen=True)
class ScoutSignals:
    """Non-answer-bearing hints about one candidate frame.

    similarity    - question-frame embedding similarity, rescaled to [0, 1].
    scene_type    - coarse category (slide / board / code / natural / blank).
    visual_change - how different this frame is from its temporal neighbour [0, 1].
    quality       - sharpness / exposure proxy [0, 1].
    uncertainty   - how unsure the scout is about the above [0, 1].

    The class is frozen and validated so a scout implementation cannot smuggle
    free text through (e.g. putting OCR into ``scene_type``).
    """

    similarity: float
    scene_type: SceneType
    visual_change: float
    quality: float
    uncertainty: float

    def __post_init__(self) -> None:
        if not isinstance(self.scene_type, SceneType):
            raise TypeError("scene_type must be a SceneType enum, not free text")
        for name in ("similarity", "visual_change", "quality", "uncertainty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"scout field {name} must be a float in [0, 1], got {value!r}")

    def as_vector(self) -> list[float]:
        """Numeric feature vector for learned heads (scene type one-hot)."""
        one_hot = [1.0 if self.scene_type == s else 0.0 for s in SceneType]
        return [self.similarity, self.visual_change, self.quality, self.uncertainty, *one_hot]


# ---------------------------------------------------------------------------
# Budget, actions and costs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Budget:
    """Hard caps.  Each is enforced independently; a scalar cost ratio is only
    used to *rank* actions, never to excuse exceeding a cap."""

    max_frames: int = 8
    max_rounds: int = 6
    max_visual_tokens: int = 4096
    max_seconds: float = 120.0
    max_expansions: int = 2
    rescue_fraction: float = 0.25   # share of max_frames reserved for LOOK_ELSEWHERE


@dataclasses.dataclass
class Action:
    kind: ActionKind
    candidate_id: str | None = None
    predicted_utility: float | None = None
    reason: str = ""   # short reason code, e.g. "SPEECH_GAP_NEAR_CANDIDATE"


@dataclasses.dataclass
class CostRecord:
    """Metered work for one stage or action.  Summed into a CostTrace."""

    stage: str
    elapsed_ms: float = 0.0
    decoded_frames: int = 0
    scored_frames: int = 0
    selected_frames: int = 0
    visual_tokens: int = 0
    text_tokens: int = 0
    peak_ram_bytes: int = 0
    peak_vram_bytes: int = 0


# ---------------------------------------------------------------------------
# What the controller sees  vs.  what training stores
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ControllerObservation:
    """Everything the controller is allowed to see at one decision step.

    Note the absences: no gold answer, no TranscriptCondition name, no damage
    record, no measured utility.  A test asserts these fields never appear.
    """

    question: str
    options: list[str] | None
    transcript: list[TranscriptSegment]           # possibly damaged, as observed
    candidates: list[Candidate]
    scout: dict[str, ScoutSignals]                # candidate_id -> signals
    looked_at: list[str]                          # candidate ids already acquired
    expansions_used: int
    frames_remaining: int
    rounds_remaining: int
    video_duration_s: float


@dataclasses.dataclass
class Answer:
    text: str                              # raw model output, verbatim
    option_index: int | None = None        # None when the model abstained
    # Frame times the model actually CITED in its text, validated against the
    # frames it was given. (It used to be a copy of every supplied frame.)
    citations_s: list[float] = dataclasses.field(default_factory=list)
    confidence: float | None = None        # answerer's own score; NOT calibrated
    missing_evidence: str = ""
    supplied_frame_times_s: list[float] = dataclasses.field(default_factory=list)
    cited_segment_ids: list[str] = dataclasses.field(default_factory=list)   # transcript lines cited
    invalid_citations: list[str] = dataclasses.field(default_factory=list)   # cited times matching nothing supplied
    abstained: bool = False                # model said the evidence is insufficient
    forced_choice: bool = False            # no letter in the text; option = model's most likely letter
    option_probs: list[float] | None = None  # softmax over the option letters at the first answer token


@dataclasses.dataclass
class ActionOutcome:
    """Measured result of one candidate action during label generation."""

    action: Action
    quality_before: float
    quality_after: float
    cost: CostRecord

    @property
    def gain(self) -> float:
        return self.quality_after - self.quality_before


@dataclasses.dataclass
class TrainingExample:
    """One row of the controller training set (fields match the research spec).

    Inputs available to the controller at inference are a subset of these
    fields; everything prefixed 'gold' / 'answer_quality' / 'preferred' is a
    training target or audit field.
    """

    video_id: str
    qa_id: str
    question: str
    options: list[str] | None
    gold_answer: str
    timed_transcript: list[TranscriptSegment]
    transcript_condition: TranscriptCondition      # metadata, NOT a model input
    damage: DamageRecord
    candidate_timestamps: list[Candidate]
    frozen_scout_signals_for_each_candidate: dict[str, ScoutSignals]
    available_budget: Budget
    answer_quality_before_each_action: dict[str, float]
    answer_quality_after_each_action: dict[str, float]
    measured_action_cost: dict[str, CostRecord]
    preferred_action_or_stop: Action
    source_dataset: str
    source_split: str
    provenance_and_license: str


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums to JSON-friendly values.

    PIL images (``Frame.image``) are dropped: we store frame *references*
    (id + timestamp), not pixels, in JSON logs.
    """
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if f.name == "image":
                continue
            out[f.name] = to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
