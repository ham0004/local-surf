"""The frozen answerer as a label source, with an exact-identity answer cache.

R(q, T, S) = 1 if the frozen Qwen3-VL answers question q correctly given the
retained transcript T and the frame set S, else 0.

Every evaluation is cached under a key built from EVERYTHING that can change the
answer: question, options, transcript text, the exact pixels of each frame (in
temporal order), the teacher's model/revision/decoding, and the prompt version.
Two label schemes that ask the same question with the same evidence therefore
share one call, and the cache tells us exactly how many real calls were made.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..answerer import AnswerRequest
from ..schemas import TranscriptSegment
from .records import FrameCandidate, QuestionPool

PROMPT_VERSION = "v1-mc-letter"   # bump if build_prompt_text changes


def evidence_key(pool: QuestionPool, frames: list[FrameCandidate], teacher_id: str,
                 transcript: list[TranscriptSegment] | None = None) -> str:
    """Stable identity of one (question, transcript, frame set, teacher, prompt) evaluation."""
    ordered = sorted(frames, key=lambda f: f.time_s)          # the answerer sees frames in time order
    payload = {
        "q": pool.question, "opts": pool.options,
        "T": [(s.start_s, s.text) for s in (pool.transcript if transcript is None else transcript)],
        "S": [f.digest for f in ordered],
        "teacher": teacher_id, "prompt": PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CachedTeacher:
    """Answer-quality oracle backed by a persistent JSONL cache.

    ``answerer`` is any object with ``answer(AnswerRequest) -> (Answer, usage)``
    (the real HF answerer, or a fake in tests). Counters separate real calls from
    cache hits so labelling budgets are reported honestly.
    """

    def __init__(self, answerer, teacher_id: str, cache_path: str | Path, frame_lookup=None) -> None:
        self.answerer, self.teacher_id = answerer, teacher_id
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
        self.calls = 0          # real answerer calls made in this session
        self.hits = 0           # requests served from the cache
        self.seconds = 0.0      # wall time spent inside real calls

    # -- the one public operation -------------------------------------------
    def quality(self, pool: QuestionPool, frames: list[FrameCandidate],
                transcript: list[TranscriptSegment] | None = None) -> float:
        """R(q, T, S): 1.0 if the answer is correct, else 0.0 (cached).

        ``transcript`` overrides the pool's retained transcript (Head A's text
        interventions); by default the fixed retained transcript is used.
        """
        if (not pool.options or pool.gold_option_index is None or
                not 0 <= pool.gold_option_index < len(pool.options)):
            raise ValueError("Framework 2 CachedTeacher requires multiple-choice options and a valid gold index; "
                             "open-ended datasets need an explicit answer-quality scorer")
        key = evidence_key(pool, frames, self.teacher_id, transcript)
        if key in self.cache:
            self.hits += 1
            return float(self.cache[key]["quality"])
        excerpt = pool.transcript if transcript is None else transcript
        ordered = sorted(frames, key=lambda f: f.time_s)
        req_frames = [_as_frame(f) for f in ordered]
        t0 = time.perf_counter()
        answer, _ = self.answerer.answer(AnswerRequest(pool.question, pool.options, excerpt, req_frames))
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        quality = 1.0 if answer.option_index == pool.gold_option_index else 0.0
        rec = {"key": key, "qa_id": pool.qa_id, "frames": [f.id for f in ordered], "quality": quality,
               "option": answer.option_index, "text": answer.text[:200]}
        self.cache[key] = rec
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return quality


def _as_frame(c: FrameCandidate):
    """Convert a candidate to the answerer's Frame record (pixels + timestamp label)."""
    from ..schemas import Frame  # noqa: PLC0415 - local import keeps records.py dependency-free

    return Frame(id=c.id, requested_s=c.time_s, decoded_pts_s=c.time_s, width=c.image.width,
                 height=c.image.height, image=c.image, phash=c.phash, digest=c.digest)
