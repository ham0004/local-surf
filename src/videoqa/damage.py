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
    """Move the segment in time by +/- ``shift_s`` (misaligned subtitles)."""
    delta = shift_s if rng.random() < 0.5 else -shift_s
    start = max(0.0, seg.start_s + delta)
    end = max(start, seg.end_s + delta)
    return dataclasses.replace(seg, start_s=start, end_s=end), len(seg.text.split())


def _apply(seg: TranscriptSegment, dtype: DamageType, rng: random.Random,
           shift_s: float) -> tuple[TranscriptSegment | None, int]:
    """Apply one damage type; returns (new segment or None if deleted, words affected)."""
    if dtype == DamageType.DELETE:
        return None, len(seg.text.split())
    if dtype == DamageType.MASK_WORDS:
        return _mask_words(seg, rng)
    if dtype == DamageType.NUMBER_SWAP:
        return _swap_numbers(seg, rng)
    if dtype == DamageType.TIME_SHIFT:
        return _time_shift(seg, rng, shift_s)
    raise ValueError(dtype)


def damage_transcript(transcript: Transcript, segment_ids: list[str], dtype: DamageType,
                      condition: TranscriptCondition, seed: int,
                      shift_s: float = 15.0) -> tuple[Transcript, DamageRecord]:
    """Return a damaged COPY of ``transcript`` plus a record of what changed.

    The original transcript object is never modified.
    """
    rng = random.Random(seed)
    targets = set(segment_ids)
    new_segments: list[TranscriptSegment] = []
    words_affected = 0
    for seg in transcript.segments:
        if seg.id in targets:
            new_seg, n = _apply(seg, dtype, rng, shift_s)
            words_affected += n
            if new_seg is not None:
                new_segments.append(new_seg)
        else:
            new_segments.append(copy.copy(seg))
    new_segments.sort(key=lambda s: s.start_s)
    record = DamageRecord(condition=condition, damage_type=dtype,
                          damaged_segment_ids=sorted(targets), words_affected=words_affected, seed=seed)
    return Transcript(transcript.video_id, new_segments, source=f"{transcript.source}+{condition.value}"), record


# ---------------------------------------------------------------------------
# Matched control selection
# ---------------------------------------------------------------------------


def _eligible_for_type(seg: TranscriptSegment, dtype: DamageType) -> bool:
    """A control segment must be able to receive the same kind of damage.
    E.g. NUMBER_SWAP control segments must actually contain a number."""
    if dtype == DamageType.NUMBER_SWAP:
        return bool(re.search(r"\d", seg.text)) or any(
            w.lower().strip(".,;:!?") in _NUMBER_WORDS for w in seg.text.split()
        )
    if dtype == DamageType.MASK_WORDS:
        return bool(_content_tokens(seg.text))
    return True


def select_matched_control(transcript: Transcript, relevant_ids: list[str], dtype: DamageType,
                           answer_text: str, seed: int, min_distance_s: float = 10.0) -> list[str] | None:
    """Pick control segments matched to the relevant ones.

    Rules (each is tested):
      * same NUMBER of segments as the targeted set;
      * none of them is relevant, and none shares a content token with the answer;
      * each is at least ``min_distance_s`` away from every relevant segment, so
        neighbouring context that also carries the answer is not touched;
      * word counts are matched greedily: for each relevant segment (longest
        first) take the unused eligible segment with the closest word count,
        breaking ties randomly (seeded).

    Returns None when no matched control exists (e.g. very short video); the
    caller must then drop the example rather than use an unmatched control.
    """
    rng = random.Random(seed)
    by_id = transcript.by_id()
    relevant = [by_id[i] for i in relevant_ids if i in by_id]
    answer_tokens = _content_tokens(answer_text)

    def far_from_relevant(seg: TranscriptSegment) -> bool:
        return all(seg.start_s >= r.end_s + min_distance_s or seg.end_s <= r.start_s - min_distance_s
                   for r in relevant)

    pool = [
        s for s in transcript.segments
        if s.id not in relevant_ids
        and far_from_relevant(s)
        and not (answer_tokens & _content_tokens(s.text))
        and _eligible_for_type(s, dtype)
    ]
    if len(pool) < len(relevant):
        return None

    chosen: list[str] = []
    for r in sorted(relevant, key=lambda s: -len(s.text.split())):
        target_len = len(r.text.split())
        rng.shuffle(pool)  # random tie-breaking, deterministic via seed
        best = min(pool, key=lambda s: abs(len(s.text.split()) - target_len))
        chosen.append(best.id)
        pool.remove(best)
    return chosen


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

    def by_condition(self) -> dict[TranscriptCondition, Transcript]:
        return {
            TranscriptCondition.CLEAN: self.clean,
            TranscriptCondition.TARGETED_DAMAGE: self.targeted,
            TranscriptCondition.CONTROL_DAMAGE: self.control,
        }


def auto_min_distance_s(transcript: Transcript, fraction: float = 0.15, floor_s: float = 1.0,
                        cap_s: float = 10.0) -> float:
    """Scale the matched-control separation to how much speech there actually is.

    The original default (a flat 10 s) assumes a lecture-length video; it was
    measured to reject 22 of 40 real LongVideoBench clips (9-70 s, often only
    1-5 transcript segments spanning barely 10 s) because almost no segment
    could ever be 10 s from another. Scaling by the transcript's own covered
    span keeps a full-lecture fixture at the original 10 s (span large enough
    to hit ``cap_s``) while giving short clips a proportionally smaller,
    still-meaningful separation instead of silently discarding almost every
    question.
    """
    if not transcript.segments:
        return floor_s
    span = max(s.end_s for s in transcript.segments) - min(s.start_s for s in transcript.segments)
    return max(floor_s, min(cap_s, span * fraction))


def make_triple(qa: QAItem, transcript: Transcript, dtype: DamageType = DamageType.DELETE,
                seed: int = 0, max_targets: int = 3,
                min_distance_s: float | None = None) -> TranscriptTriple | None:
    """Build CLEAN / TARGETED / CONTROL transcripts for one question.

    ``min_distance_s`` is the minimum separation required between a targeted
    and a control segment; ``None`` (the default) auto-scales it to the
    transcript's own span via :func:`auto_min_distance_s` rather than assuming
    a fixed video length.

    Returns None if the question has no answer-relevant speech (nothing to
    target) or if no matched control can be found.  Dropping such items is
    safer than silently producing an unfair pair.
    """
    rel = derive_relevant_segments(qa, transcript)
    targets = rel.segment_ids[:max_targets]
    if not targets:
        return None
    # NUMBER_SWAP / MASK_WORDS targets must themselves be damageable.
    by_id = transcript.by_id()
    targets = [t for t in targets if _eligible_for_type(by_id[t], dtype)]
    if not targets:
        return None
    distance = auto_min_distance_s(transcript) if min_distance_s is None else min_distance_s
    control_ids = select_matched_control(transcript, targets, dtype, qa.gold_answer, seed, min_distance_s=distance)
    if control_ids is None:
        return None

    clean_record = DamageRecord(TranscriptCondition.CLEAN, None, [], 0, seed)
    targeted, t_rec = damage_transcript(transcript, targets, dtype, TranscriptCondition.TARGETED_DAMAGE, seed)
    control, c_rec = damage_transcript(transcript, control_ids, dtype, TranscriptCondition.CONTROL_DAMAGE, seed)
    return TranscriptTriple(
        clean=transcript,
        targeted=targeted,
        control=control,
        records={TranscriptCondition.CLEAN: clean_record,
                 TranscriptCondition.TARGETED_DAMAGE: t_rec,
                 TranscriptCondition.CONTROL_DAMAGE: c_rec},
        relevance_method=rel.method,
    )


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
