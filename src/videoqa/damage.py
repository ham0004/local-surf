"""Transcript damage generators - the heart of the paired training idea.

For ONE (video, question) we build three transcripts:

    CLEAN            the original
    TARGETED_DAMAGE  damage applied to answer-relevant segments
    CONTROL_DAMAGE   the SAME damage type applied to the SAME number of segments,
                     with as close a word count as possible, chosen from speech
                     that is NOT relevant to the answer

If the controller spends more visual effort under TARGETED than under CONTROL,
it is reacting to *what* was lost, not merely to *how much* was lost.  That is
exactly the behaviour the research hypothesis wants to test, so the matching
rules here are checked by tests (see tests/test_damage.py).

Relevance labels come from dataset evidence annotations or, as a fallback, from
lexical overlap with the gold answer.  Both use gold information and therefore
may only be run on TRAIN videos (or for evaluation bookkeeping), never inside
the inference path.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import random
import re

from .schemas import (
    DamageRecord,
    DamageType,
    QAItem,
    Transcript,
    TranscriptCondition,
    TranscriptSegment,
)

MASK_TOKEN = "[inaudible]"

# Tiny stopword list: words we never count as "answer-bearing" when matching
# answer tokens against transcript text.
_STOP = frozenset(
    "a an the of to in on at for and or but is are was were be been it this that these "
    "those with as by from we you i they he she our your their so then than there here "
    "what which who how why when where do does did not no yes".split()
)

_NUMBER_WORDS = {
    "zero": "one", "one": "two", "two": "three", "three": "four", "four": "five",
    "five": "six", "six": "seven", "seven": "eight", "eight": "nine", "nine": "ten", "ten": "eleven",
}


def _content_tokens(text: str) -> set[str]:
    """Lower-cased alphanumeric tokens minus stopwords (used for relevance)."""
    return {t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOP}


# ---------------------------------------------------------------------------
# Which segments are "answer-relevant"?  (TRAIN / EVAL ONLY - uses gold data)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Relevance:
    segment_ids: list[str]
    method: str   # "annotation_ids" | "annotation_intervals" | "answer_overlap" | "none"


def derive_relevant_segments(qa: QAItem, transcript: Transcript) -> Relevance:
    """Find answer-relevant segments, preferring human annotations.

    1. Explicit ``evidence_segment_ids`` from the dataset.
    2. Segments overlapping annotated ``evidence_intervals_s``.
    3. Fallback: segments sharing a content token with the gold answer.
       This is noisy (it can miss paraphrases) and is recorded as such.
    """
    known = {s.id for s in transcript.segments}
    ids = [sid for sid in qa.evidence_segment_ids if sid in known]
    if ids:
        return Relevance(ids, "annotation_ids")

    if qa.evidence_intervals_s:
        hits = [
            s.id for s in transcript.segments
            if any(min(b, s.end_s) - max(a, s.start_s) > 0 for a, b in qa.evidence_intervals_s)
        ]
        if hits:
            return Relevance(hits, "annotation_intervals")

    answer_tokens = _content_tokens(qa.gold_answer)
    if answer_tokens:
        hits = [s.id for s in transcript.segments if answer_tokens & _content_tokens(s.text)]
        if hits:
            return Relevance(hits, "answer_overlap")
    return Relevance([], "none")


# ---------------------------------------------------------------------------
# Single-segment damage operators
# ---------------------------------------------------------------------------


def _mask_words(seg: TranscriptSegment, rng: random.Random) -> tuple[TranscriptSegment | None, int]:
    """Replace every content word with MASK_TOKEN (keeps stopwords, so the line
    still *looks* like speech with holes, as garbled ASR does)."""
    words = seg.text.split()
    changed = 0
    out = []
    for w in words:
        core = re.sub(r"[^\w.]", "", w.lower())
        if core and core not in _STOP:
            out.append(MASK_TOKEN)
            changed += 1
        else:
            out.append(w)
    return dataclasses.replace(seg, text=" ".join(out)), changed


def _swap_numbers(seg: TranscriptSegment, rng: random.Random) -> tuple[TranscriptSegment | None, int]:
    """Alter every number (digits or small number words) - a classic ASR error
    that silently changes meaning ("minus 3" -> "minus 8")."""
    changed = 0

    def swap_digits(m: re.Match) -> str:
        nonlocal changed
        s = m.group(0)
        chars = list(s)
        idx = [i for i, c in enumerate(chars) if c.isdigit()]
        i = rng.choice(idx)
        # choose a different digit so the value really changes
        chars[i] = rng.choice([d for d in "0123456789" if d != chars[i]])
        changed += 1
        return "".join(chars)

    text = re.sub(r"\d+(?:\.\d+)?", swap_digits, seg.text)
    words = text.split()
    for i, w in enumerate(words):
        key = w.lower().strip(".,;:!?")
        if key in _NUMBER_WORDS:
            words[i] = _NUMBER_WORDS[key]
            changed += 1
    return dataclasses.replace(seg, text=" ".join(words)), changed


def _time_shift(seg: TranscriptSegment, rng: random.Random, shift_s: float) -> tuple[TranscriptSegment, int]:
    """Move the segment in time by ``shift_s`` (signed; misaligned subtitles).

    The SIGN is chosen by the caller once per triple, so the targeted and the
    control segment move in the same direction by the same amount; clamping at
    0 s can still shorten the effective shift, which the damage record measures.
    """
    start = max(0.0, seg.start_s + shift_s)
    end = max(start, seg.end_s + shift_s)
    return dataclasses.replace(seg, start_s=start, end_s=end), len(seg.text.split())


def _apply(seg: TranscriptSegment, dtype: DamageType, rng: random.Random,
           shift_s: float) -> tuple[TranscriptSegment | None, int]:
    """Apply one damage type; returns (new segment or None if deleted, tokens changed)."""
    if dtype == DamageType.DELETE:
        return None, len(seg.text.split())
    if dtype == DamageType.MASK_WORDS:
        return _mask_words(seg, rng)
    if dtype == DamageType.NUMBER_SWAP:
        return _swap_numbers(seg, rng)
    if dtype == DamageType.TIME_SHIFT:
        return _time_shift(seg, rng, shift_s)
    raise ValueError(dtype)


def effect_size(seg: TranscriptSegment, dtype: DamageType, shift_s: float = 15.0) -> int:
    """Tokens the operation would change in ``seg`` (0 = it would be a no-op).

    The COUNT of changed tokens does not depend on which replacement digit is
    drawn, so this is a seed-independent severity measure for matching.
    """
    new_seg, n = _apply(seg, dtype, random.Random(0), shift_s)
    if dtype == DamageType.TIME_SHIFT and new_seg is not None and new_seg.start_s == seg.start_s:
        return 0   # clamped to no movement at all
    return n


def damage_transcript(transcript: Transcript, segment_ids: list[str], dtype: DamageType,
                      condition: TranscriptCondition, seed: int, shift_s: float = 15.0,
                      role: str = "", support_type: str = "") -> tuple[Transcript, DamageRecord]:
    """Return a damaged COPY of ``transcript`` plus a record of what actually changed.

    The original transcript object is never modified.
    """
    rng = random.Random(seed)
    targets = set(segment_ids)
    new_segments: list[TranscriptSegment] = []
    changed_tokens = 0
    coverage = 0.0
    intervals: list[tuple[float, float]] = []
    shifts: list[float] = []
    for seg in transcript.segments:
        if seg.id in targets:
            new_seg, n = _apply(seg, dtype, rng, shift_s)
            if dtype == DamageType.TIME_SHIFT and new_seg is not None:
                shifts.append(new_seg.start_s - seg.start_s)
                if new_seg.start_s == seg.start_s:
                    n = 0
            changed_tokens += n
            coverage += seg.duration_s
            intervals.append((seg.start_s, seg.end_s))
            if new_seg is not None:
                new_segments.append(new_seg)
        else:
            new_segments.append(copy.copy(seg))
    new_segments.sort(key=lambda s: s.start_s)
    record = DamageRecord(condition=condition, damage_type=dtype, damaged_segment_ids=sorted(targets),
                          words_affected=sum(len(s.text.split()) for s in transcript.segments if s.id in targets),
                          seed=seed, changed_tokens=changed_tokens, time_coverage_s=round(coverage, 3),
                          intervals_s=sorted(intervals), changed=changed_tokens > 0,
                          displacement_s=round(sum(shifts) / len(shifts), 3) if shifts else 0.0,
                          role=role, support_type=support_type)
    return Transcript(transcript.video_id, new_segments, source=f"{transcript.source}+{condition.value}"), record


# ---------------------------------------------------------------------------
# Matched control selection
# ---------------------------------------------------------------------------

# PREDECLARED on 2026-09-28, before any new outcome was inspected: a control is
# matched when its effective changed-token count is within max(1, 20 %) of the
# targeted count. Sets outside this tolerance are dropped, never used.
SEVERITY_TOLERANCE = 0.20


def severity_matched(targeted_tokens: int, control_tokens: int, tolerance: float = SEVERITY_TOLERANCE) -> bool:
    return abs(control_tokens - targeted_tokens) <= max(1, round(tolerance * targeted_tokens))


def _eligible_for_type(seg: TranscriptSegment, dtype: DamageType) -> bool:
    """A segment can take ``dtype`` only if the damage would actually change
    something (so NUMBER_SWAP needs a number, and no-op controls are rejected)."""
    return effect_size(seg, dtype) > 0


def _control_pool(transcript: Transcript, all_relevant_ids: list[str], dtype: DamageType,
                  answer_text: str, min_distance_s: float) -> list[TranscriptSegment]:
    """Segments that may serve as controls: not relevant (the WHOLE known
    relevant set, not only the capped targets), far from every relevant
    segment, not answer-bearing, and damageable by ``dtype``."""
    by_id = transcript.by_id()
    relevant = [by_id[i] for i in all_relevant_ids if i in by_id]
    answer_tokens = _content_tokens(answer_text)
    excluded = set(all_relevant_ids)

    def far(seg: TranscriptSegment) -> bool:
        return all(seg.start_s >= r.end_s + min_distance_s or seg.end_s <= r.start_s - min_distance_s
                   for r in relevant)

    return [s for s in transcript.segments
            if s.id not in excluded and far(s) and not (answer_tokens & _content_tokens(s.text))
            and _eligible_for_type(s, dtype)]


def _combos(targets: list[TranscriptSegment], pool: list[TranscriptSegment], dtype: DamageType,
            k: int = 6) -> list[tuple[str, ...]]:
    """All severity-matched control sets: one control per target, drawn from
    each target's ``k`` closest-effect pool segments (bounded search)."""
    t_total = sum(effect_size(t, dtype) for t in targets)
    options = []
    for t in targets:
        te = effect_size(t, dtype)
        options.append(sorted(pool, key=lambda s: (abs(effect_size(s, dtype) - te), s.start_s))[:k])
    out = set()
    for combo in itertools.product(*options):
        ids = tuple(s.id for s in combo)
        if len(set(ids)) != len(ids):
            continue
        if severity_matched(t_total, sum(effect_size(s, dtype) for s in combo)):
            out.add(tuple(sorted(ids)))
    return sorted(out)


def select_matched_control(transcript: Transcript, relevant_ids: list[str], dtype: DamageType,
                           answer_text: str, seed: int, min_distance_s: float = 10.0,
                           all_relevant_ids: list[str] | None = None) -> list[str] | None:
    """Pick one severity-matched control set for the targeted segments.

    Rules (each is tested):
      * same number of segments and same damage type as the targeted set;
      * never a segment from the complete known relevant set
        (``all_relevant_ids``; defaults to ``relevant_ids``), never within
        ``min_distance_s`` of one, never sharing a content token with the answer;
      * effective changed tokens within the predeclared tolerance
        (:data:`SEVERITY_TOLERANCE`), and never a no-op;
      * among all valid sets, a seeded random choice.

    Returns None when no matched control exists; the caller drops the example.
    """
    by_id = transcript.by_id()
    targets = [by_id[i] for i in relevant_ids if i in by_id]
    pool = _control_pool(transcript, all_relevant_ids or relevant_ids, dtype, answer_text, min_distance_s)
    combos = _combos(targets, pool, dtype)
    if not combos:
        return None
    return list(random.Random(seed).choice(combos))


# ---------------------------------------------------------------------------
# The triple used for training / evaluation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TranscriptTriple:
    clean: Transcript
    targeted: Transcript
    control: Transcript
    records: dict[TranscriptCondition, DamageRecord]
    relevance_method: str
    # Extra matched controls for ablations: role -> (transcript, record).
    # "control_near": the valid control set closest in time to the targets;
    # "control_distant": the farthest. Absent when identical to the primary.
    extra_controls: dict[str, tuple[Transcript, DamageRecord]] = dataclasses.field(default_factory=dict)

    def by_condition(self) -> dict[TranscriptCondition, Transcript]:
        return {
            TranscriptCondition.CLEAN: self.clean,
            TranscriptCondition.TARGETED_DAMAGE: self.targeted,
            TranscriptCondition.CONTROL_DAMAGE: self.control,
        }


def auto_min_distance_s(transcript: Transcript, fraction: float = 0.15, floor_s: float = 1.0,
                        cap_s: float = 10.0) -> float:
    """Scale the matched-control separation to how much speech there actually is.

    A flat 10 s assumes a lecture-length video; it was measured to reject 22 of
    40 real LongVideoBench clips (9-70 s, often only 1-5 transcript segments).
    Scaling by the transcript's own covered span keeps a full-lecture fixture at
    10 s (span large enough to hit ``cap_s``) while giving short clips a
    proportionally smaller separation.
    """
    if not transcript.segments:
        return floor_s
    span = max(s.end_s for s in transcript.segments) - min(s.start_s for s in transcript.segments)
    return max(floor_s, min(cap_s, span * fraction))


# Relevance sources that come from dataset annotations (as opposed to the
# gold-answer word-overlap fallback).
ANNOTATION_RELEVANCE = frozenset({"annotation_ids", "annotation_intervals"})

# Every reason a question can be excluded from the paired experiment.
DROP_REASONS = ("no_relevant_segment", "relevance_source_excluded", "targets_not_damageable",
                "no_matched_control")


def build_triple(qa: QAItem, transcript: Transcript, dtype: DamageType = DamageType.DELETE,
                 seed: int = 0, max_targets: int = 3, min_distance_s: float | None = None,
                 relevance_methods: frozenset[str] | None = None,
                 shift_s: float = 15.0) -> tuple[TranscriptTriple | None, str]:
    """Build CLEAN / TARGETED / CONTROL transcripts, or say exactly why not.

    Returns ``(triple, "ok")`` or ``(None, reason)`` with reason in
    :data:`DROP_REASONS`, so every exclusion can be counted and reported.
    """
    rel = derive_relevant_segments(qa, transcript)
    if not rel.segment_ids:
        return None, "no_relevant_segment"
    if relevance_methods is not None and rel.method not in relevance_methods:
        return None, "relevance_source_excluded"
    by_id = transcript.by_id()
    targets = [t for t in rel.segment_ids[:max_targets] if _eligible_for_type(by_id[t], dtype)]
    if not targets:
        return None, "targets_not_damageable"
    distance = auto_min_distance_s(transcript) if min_distance_s is None else min_distance_s
    target_segs = [by_id[t] for t in targets]
    pool = _control_pool(transcript, rel.segment_ids, dtype, qa.gold_answer, distance)
    combos = _combos(target_segs, pool, dtype)
    if not combos:
        return None, "no_matched_control"
    primary = list(random.Random(seed).choice(combos))

    # One shift sign per triple, so targeted and control move the same way.
    signed_shift = shift_s if random.Random(seed + 1).random() < 0.5 else -shift_s
    support = qa.evidence_type
    clean_record = DamageRecord(TranscriptCondition.CLEAN, None, [], 0, seed, role="clean", support_type=support)
    targeted, t_rec = damage_transcript(transcript, targets, dtype, TranscriptCondition.TARGETED_DAMAGE, seed,
                                        signed_shift, role="targeted", support_type=support)
    control, c_rec = damage_transcript(transcript, primary, dtype, TranscriptCondition.CONTROL_DAMAGE, seed,
                                       signed_shift, role="control", support_type=support)

    def gap(ids: tuple[str, ...]) -> float:
        return min(min(abs(by_id[i].start_s - t.start_s) for t in target_segs) for i in ids)

    extra: dict[str, tuple[Transcript, DamageRecord]] = {}
    for role, pick in (("control_near", min(combos, key=gap)), ("control_distant", max(combos, key=gap))):
        taken = [primary] + [r.damaged_segment_ids for _, r in extra.values()]
        if list(pick) not in taken:
            extra[role] = damage_transcript(transcript, list(pick), dtype, TranscriptCondition.CONTROL_DAMAGE,
                                            seed, signed_shift, role=role, support_type=support)
    return TranscriptTriple(
        clean=transcript, targeted=targeted, control=control,
        records={TranscriptCondition.CLEAN: clean_record, TranscriptCondition.TARGETED_DAMAGE: t_rec,
                 TranscriptCondition.CONTROL_DAMAGE: c_rec},
        relevance_method=rel.method, extra_controls=extra,
    ), "ok"


def make_triple(qa: QAItem, transcript: Transcript, dtype: DamageType = DamageType.DELETE,
                seed: int = 0, max_targets: int = 3,
                min_distance_s: float | None = None,
                relevance_methods: frozenset[str] | None = None) -> TranscriptTriple | None:
    """Backward-compatible wrapper around :func:`build_triple` (drops the reason)."""
    return build_triple(qa, transcript, dtype, seed, max_targets, min_distance_s, relevance_methods)[0]


# ---------------------------------------------------------------------------
# Untargeted "realistic" ASR-style noise (evaluation stress condition)
# ---------------------------------------------------------------------------


def asr_style_noise(transcript: Transcript, word_error_rate: float, seed: int,
                    max_time_jitter_s: float = 0.5) -> Transcript:
    """Untargeted noise resembling ASR output: random word deletions, masking,
    number swaps and small timestamp jitter at roughly ``word_error_rate``.

    This is a *synthetic approximation*.  The research plan requires evaluating
    on genuinely natural ASR errors too (e.g. faster-whisper vs. reference
    subtitles); do not report results on this function as "natural ASR".
    """
    rng = random.Random(seed)
    out = []
    for seg in transcript.segments:
        words = []
        for w in seg.text.split():
            r = rng.random()
            if r < word_error_rate / 3:
                continue                          # deletion
            if r < 2 * word_error_rate / 3:
                words.append(MASK_TOKEN)          # unintelligible
            elif r < word_error_rate and re.search(r"\d", w):
                words.append(re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), w))
            else:
                words.append(w)
        if not words:
            continue
        jitter = rng.uniform(-max_time_jitter_s, max_time_jitter_s)
        start = max(0.0, seg.start_s + jitter)
        out.append(dataclasses.replace(seg, text=" ".join(words), start_s=start,
                                       end_s=max(start, seg.end_s + jitter)))
    return Transcript(transcript.video_id, out, source=f"{transcript.source}+asr_noise{word_error_rate}")
