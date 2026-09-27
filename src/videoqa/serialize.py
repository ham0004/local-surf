"""Text serialization of one (observation, action) for the text-LLM controller.

ALLOWLIST: the serialized text is built ONLY from these observation fields.

    question, options                  (what a user asks)
    observed transcript lines           (possibly damaged; the ones within
                                         +/-15 s of the candidate, capped at
                                         MAX_LOCAL_WORDS words)
    candidate time, video duration      (seconds)
    candidate source and retrieval rank (from the observed transcript)
    scout signals IF this candidate was scouted:
        similarity, visual_change, quality, uncertainty, scene type
    times already looked at, frames/rounds/expansions remaining
    global transcript statistics: question-term coverage, masked fraction

Never included: OCR text, captions, the gold answer, condition names,
damage records or deleted spans, answer-quality labels, anything decoded
from pixels other than the five scout signals. There is no field through
which they could enter; tests/test_serialize.py checks that the text is
unchanged when the gold answer, condition or damage metadata change.
"""

from __future__ import annotations

from .features import global_features
from .schemas import ActionKind, Candidate, ControllerObservation

MAX_LOCAL_WORDS = 60
ALLOWED_FIELDS = ("question", "options", "transcript", "candidates", "scout", "looked_at",
                  "expansions_used", "frames_remaining", "rounds_remaining", "video_duration_s")


def _local_transcript(obs: ControllerObservation, t: float, radius: float = 15.0) -> str:
    near = [s for s in obs.transcript if abs(0.5 * (s.start_s + s.end_s) - t) <= radius]
    words = " ".join(s.text for s in sorted(near, key=lambda s: s.start_s)).split()
    text = " ".join(words[:MAX_LOCAL_WORDS])
    return text if text else "(no speech)"


def serialize(obs: ControllerObservation, kind: ActionKind, cand: Candidate | None) -> str:
    g = global_features(obs)
    opts = " ".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(obs.options or []))
    looked = [c for c in obs.candidates if c.id in obs.looked_at]
    lines = [
        f"Question: {obs.question}",
        f"Options: {opts}" if opts else "Options: none",
        f"Video: {obs.video_duration_s:.0f}s. Looked at: "
        + (", ".join(f"{c.time_s:.0f}s" for c in looked) or "nothing")
        + f". Frames left {obs.frames_remaining}, rounds left {obs.rounds_remaining}, "
          f"expansions used {obs.expansions_used}.",
        f"Transcript: question-term coverage {g['question_term_coverage']:.2f}, "
        f"masked fraction {g['masked_fraction']:.2f}.",
    ]
    if kind == ActionKind.EXPAND_TRANSCRIPT or cand is None:
        lines.append("Action: read more transcript around the retrieved windows.")
    else:
        src = cand.source.value + (f" rank {cand.rank}" if cand.rank is not None else "")
        sig = obs.scout.get(cand.id)
        scout = ("not scouted" if sig is None else
                 f"similarity {sig.similarity:.2f}, change {sig.visual_change:.2f}, quality {sig.quality:.2f}, "
                 f"uncertainty {sig.uncertainty:.2f}, scene {sig.scene_type.value}")
        lines += [
            f"Action: look at {cand.time_s:.1f}s ({src}).",
            f"Scout: {scout}.",
            f"Speech near {cand.time_s:.0f}s: {_local_transcript(obs, cand.time_s)}",
        ]
    lines.append("Expected answer improvement:")
    return "\n".join(lines)
