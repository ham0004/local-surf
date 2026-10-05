"""Open-ended answers from the frozen VLM, cached under the complete request identity.

The v1/v2 prompt asks for an option letter or "a short answer" (64 tokens). Open-ended
lecture questions (EduVidQA) have explanatory references of ~40-120 words, so v3 uses its
own instruction and a larger generation budget. The answer model itself is unchanged
(Qwen3-VL-2B, frozen, greedy).

Cache identity = hash of: system prompt, the exact v3 prompt text (question + timestamped
transcript lines), every frame's pixel digest with its "Frame at" label, and the resolved
answerer identity (model, revision, dtype, decoding, max_new_tokens, prompt version).
A CallBudget (fresh calls AND seconds) is charged for every real generation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..answerer import SYSTEM_PROMPT, AnswerRequest
from ..packing import format_excerpt

OPEN_PROMPT_VERSION = "v3-open-1"
OPEN_MAX_NEW_TOKENS = 256


def build_open_prompt(question: str, excerpt) -> str:
    """Text part of an open-ended request (frames are attached separately with time labels)."""
    return "\n\n".join([
        f"QUESTION: {question}",
        "TRANSCRIPT EXCERPT:\n" + (format_excerpt(excerpt) or "(none)"),
        "Answer the question in 2-5 sentences, explaining the reasoning a student needs. "
        "Use the transcript and the frames as evidence; say so if they are insufficient.",
    ])


def answerer_identity(cfg: dict) -> dict:
    a = cfg["answerer"]
    return {"model_id": a["model_id"], "revision": a.get("revision"), "dtype": a.get("dtype", "bfloat16"),
            "decoding": "greedy", "max_new_tokens": OPEN_MAX_NEW_TOKENS, "frame_max_side": a.get("frame_max_side"),
            "prompt_version": OPEN_PROMPT_VERSION,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}


def request_key(question: str, excerpt, frames, identity: dict) -> str:
    """frames: objects with .digest and .decoded_pts_s (schemas.Frame), any order."""
    ordered = sorted(frames, key=lambda f: f.decoded_pts_s)
    payload = {"schema": "v3-open-request", "identity": identity, "prompt": build_open_prompt(question, excerpt),
               "frames": [[f.digest, f"Frame at {f.decoded_pts_s:.2f}s:"] for f in ordered]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_open_answerer(cfg: dict):
    """The frozen HF VLM with the v3 open-ended prompt and generation budget."""
    from pathlib import Path as _P  # noqa: PLC0415

    from ..answerer_hf import HFVLMAnswerer  # noqa: PLC0415 - torch stays optional

    class _OpenEnded(HFVLMAnswerer):
        def _messages(self, req: AnswerRequest) -> list[dict]:
            content: list[dict] = []
            for f in sorted(req.frames, key=lambda f: f.decoded_pts_s):
                content.append({"type": "text", "text": f"Frame at {f.decoded_pts_s:.2f}s:"})
                content.append({"type": "image", "image": f.image})
            content.append({"type": "text", "text": build_open_prompt(req.question, req.excerpt)})
            return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]

    a = cfg["answerer"]
    cache_dir = str(_P(cfg.get("paths", {}).get("cache_dir", "cache")) / "hf" / "hub")
    return _OpenEnded(model_id=a["model_id"], device=cfg.get("device", "cuda"), dtype=a.get("dtype", "bfloat16"),
                      max_new_tokens=OPEN_MAX_NEW_TOKENS, revision=a.get("revision"), cache_dir=cache_dir)


class AnswerCache:
    """Persistent JSONL cache of generated answers; charges a CallBudget for real calls."""

    def __init__(self, path: str | Path, identity: dict, answerer=None, budget=None) -> None:
        self.path, self.identity, self.answerer, self.budget = Path(path), identity, answerer, budget
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec
        self.calls, self.hits, self.seconds = 0, 0, 0.0

    def get(self, question: str, excerpt, frames) -> dict | None:
        return self.records.get(request_key(question, excerpt, frames, self.identity))

    def answer(self, qa_id: str, question: str, excerpt, frames) -> dict:
        key = request_key(question, excerpt, frames, self.identity)
        if key in self.records:
            self.hits += 1
            return self.records[key]
        if self.answerer is None:
            raise LookupError(f"{qa_id}: answer not cached and no answerer loaded")
        if self.budget is not None:
            self.budget.check_one()
        t0 = time.perf_counter()
        ans, usage = self.answerer.answer(AnswerRequest(question, None, list(excerpt),
                                                        sorted(frames, key=lambda f: f.decoded_pts_s)))
        spent = time.perf_counter() - t0
        self.calls += 1
        self.seconds += spent
        if self.budget is not None:
            self.budget.charge(spent)
        rec = {"key": key, "qa_id": qa_id, "frame_times": sorted(f.decoded_pts_s for f in frames),
               "text": ans.text, "seconds": spent, "visual_tokens": usage.visual_tokens,
               "generated_tokens": usage.generated_tokens}
        self.records[key] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec
