"""The frozen answerer as a label source, with an exact-identity answer cache.

R(q, T, S) = 1 if the frozen Qwen3-VL answers question q correctly given the
retained transcript T and the frame set S, else 0.

Cache schema v2 (current)
-------------------------
The key is the hash of the COMPLETE effective request: the system prompt, the
exact rendered prompt text (question, options, transcript lines with their ids
and start/end times), every frame's pixel digest with the time label shown to
the model, in the order the model sees them, and the resolved answerer
identity (model, revision, dtype, decoding, image size). The cache stores the
model's PREDICTION; correctness is recomputed against the current gold index,
so a changed gold answer can never be served a stale "correct".

Legacy schema v1 (historical runs, read-only)
---------------------------------------------
v1 keys hashed question, options, transcript (start, text), pixel digests and a
teacher id, but not frame time labels, transcript ids/end times or decoding
settings. v1 records are still used when (a) the v1 key matches, (b) the
question id matches and (c) the record's candidate ids equal the requested
ids (ids encode the frame's position in the SAME saved pool). Callers must
only attach legacy caches built from the same pool files; new records are
always written in v2. Historical results are therefore reproducible, and no
new label depends on an unverified v1 identity.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..answerer import SYSTEM_PROMPT, AnswerRequest, build_prompt_text
from ..schemas import TranscriptSegment
from .records import FrameCandidate, QuestionPool

PROMPT_VERSION = "v1-mc-letter"   # legacy v1 key component; v2 hashes the rendered prompt itself
KEY_SCHEMA = "v2-rendered-request"


def answerer_identity(cfg: dict) -> dict:
    """Resolved, label-relevant answerer settings from a loaded config.

    Mirrors the defaults of ``make_answerer`` so a cache-only reader (no model
    loaded) computes exactly the same identity as the labelling run.
    """
    a = cfg["answerer"]
    return {"backend": a.get("backend", "hf_vlm"), "model_id": a["model_id"], "revision": a.get("revision"),
            "dtype": a.get("dtype", "bfloat16"), "max_new_tokens": a.get("max_new_tokens", 64),
            "decoding": "greedy", "frame_max_side": a.get("frame_max_side"),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}


def _ordered(frames: list[FrameCandidate]) -> list[FrameCandidate]:
    return sorted(frames, key=lambda f: f.time_s)            # the answerer sees frames in time order


def request_key(pool: QuestionPool, frames: list[FrameCandidate], identity: dict,
                transcript: list[TranscriptSegment] | None = None) -> str:
    """Schema v2: hash of everything the answerer receives, plus its identity."""
    excerpt = pool.transcript if transcript is None else transcript
    payload = {
        "schema": KEY_SCHEMA, "identity": identity,
        "prompt": build_prompt_text(AnswerRequest(pool.question, pool.options, excerpt, [])),
        "frames": [[f.digest, f"Frame at {f.time_s:.2f}s:"] for f in _ordered(frames)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evidence_key(pool: QuestionPool, frames: list[FrameCandidate], teacher_id: str,
                 transcript: list[TranscriptSegment] | None = None) -> str:
    """Legacy schema v1 key (kept to read historical caches; never used for new records)."""
    payload = {
        "q": pool.question, "opts": pool.options,
        "T": [(s.start_s, s.text) for s in (pool.transcript if transcript is None else transcript)],
        "S": [f.digest for f in _ordered(frames)],
        "teacher": teacher_id, "prompt": PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CachedTeacher:
    """Answer-quality oracle backed by a persistent JSONL cache.

    ``answerer`` is any object with ``answer(AnswerRequest) -> (Answer, usage)``
    (the real HF answerer, or a fake in tests), or None for cache-only reading.
    ``identity`` is the resolved answerer identity (``answerer_identity(cfg)``);
    without it the bare ``teacher_id`` is used (tests, fakes).
    ``legacy_paths``: additional read-only caches built from the SAME pools.
    Counters separate real calls, v2 hits and verified legacy hits.
    """

    def __init__(self, answerer, teacher_id: str, cache_path: str | Path, identity: dict | None = None,
                 legacy_paths: tuple = ()) -> None:
        self.answerer, self.teacher_id = answerer, teacher_id
        self.identity = identity or {"teacher_id": teacher_id}
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = {}      # v2 key -> record
        self.legacy: dict[str, dict] = {}     # v1 key -> record (read-only)
        for path in (self.cache_path, *map(Path, legacy_paths)):
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        (self.cache if rec.get("schema") == KEY_SCHEMA else self.legacy)[rec["key"]] = rec
        self.calls = 0          # real answerer calls made in this session
        self.hits = 0           # requests served from the cache (v2 or verified legacy)
        self.legacy_hits = 0    # subset of hits served by verified v1 records
        self.seconds = 0.0      # wall time spent inside real calls

    def cached_prediction(self, pool: QuestionPool, frames: list[FrameCandidate],
                          transcript: list[TranscriptSegment] | None = None) -> dict | None:
        """The stored prediction for this exact request, or None (no model call)."""
        rec = self.cache.get(request_key(pool, frames, self.identity, transcript))
        if rec is not None:
            return rec
        old = self.legacy.get(evidence_key(pool, frames, self.teacher_id, transcript))
        if (old is not None and old.get("qa_id") == pool.qa_id and "option" in old
                and sorted(old["frames"]) == sorted(f.id for f in frames)):
            return old
        return None

    # -- the one public operation -------------------------------------------
    def quality(self, pool: QuestionPool, frames: list[FrameCandidate],
                transcript: list[TranscriptSegment] | None = None) -> float:
        """R(q, T, S): 1.0 if the predicted option equals the CURRENT gold, else 0.0.

        ``transcript`` overrides the pool's retained transcript (Head A's text
        interventions); by default the fixed retained transcript is used.
        """
        if (not pool.options or pool.gold_option_index is None or
                not 0 <= pool.gold_option_index < len(pool.options)):
            raise ValueError("Framework 2 CachedTeacher requires multiple-choice options and a valid gold index; "
                             "open-ended datasets need an explicit answer-quality scorer")
        rec = self.cached_prediction(pool, frames, transcript)
        if rec is not None:
            self.hits += 1
            self.legacy_hits += rec.get("schema") != KEY_SCHEMA
            return 1.0 if rec["option"] == pool.gold_option_index else 0.0
        if self.answerer is None:
            raise LookupError(f"{pool.qa_id}: request not cached and no answerer loaded")
        excerpt = pool.transcript if transcript is None else transcript
        ordered = _ordered(frames)
        t0 = time.perf_counter()
        answer, _ = self.answerer.answer(AnswerRequest(pool.question, pool.options, excerpt,
                                                       [_as_frame(f) for f in ordered]))
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        rec = {"schema": KEY_SCHEMA, "key": request_key(pool, frames, self.identity, transcript),
               "qa_id": pool.qa_id, "frames": [f.id for f in ordered], "frame_times": [f.time_s for f in ordered],
               "option": answer.option_index, "text": answer.text[:200]}
        self.cache[rec["key"]] = rec
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 1.0 if answer.option_index == pool.gold_option_index else 0.0


def _as_frame(c: FrameCandidate):
    """Convert a candidate to the answerer's Frame record (pixels + timestamp label)."""
    from ..schemas import Frame  # noqa: PLC0415 - local import keeps records.py dependency-free

    return Frame(id=c.id, requested_s=c.time_s, decoded_pts_s=c.time_s, width=c.image.width,
                 height=c.image.height, image=c.image, phash=c.phash, digest=c.digest)
