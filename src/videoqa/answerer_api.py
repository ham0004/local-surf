"""Frozen answer model behind an OpenAI-compatible API (NVIDIA NIM free tier by default).

Same prompt as the local answerers (``SYSTEM_PROMPT``, one "Frame at Xs:" label before each picture, then
``build_prompt_text``), sent as chat messages with JPEG pictures. The model reasons before answering, so the
generation budget must leave room for it (with 512 tokens the reasoning is cut and the letter is unreliable).

Option probabilities: the API returns log-probabilities for every generated token, including the reasoning.
The answer letter is the last option-letter token of the reply; its top alternatives give the option
distribution (letters not among the alternatives get probability 0, then the rest is renormalised).

The key is read from ``~/<key_file>`` and never written anywhere. Calls are paced across threads to stay under
the free tier's per-minute limit, and transient errors are retried with backoff.
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .answerer import OPTION_LETTERS, SYSTEM_PROMPT, AnswerRequest, AnswerUsage, build_prompt_text, finalize_answer


def _jpeg(image, max_side: int) -> str:
    im = image.convert("RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class APIVLMAnswerer:
    name = "api_vlm"

    def __init__(self, model_id: str, base_url: str = "https://integrate.api.nvidia.com/v1",
                 key_file: str = ".nvidia_api_key", max_tokens: int = 2048, frame_max_side: int = 640,
                 min_interval_s: float = 1.6, top_logprobs: int = 20) -> None:
        self.model_id, self.base_url, self.key_file = model_id, base_url.rstrip("/"), key_file
        self.max_tokens, self.frame_max_side, self.min_interval_s, self.top_logprobs = (
            max_tokens, frame_max_side, min_interval_s, top_logprobs)
        self._lock = threading.Lock()
        self._next = 0.0

    def _pace(self) -> None:
        with self._lock:
            start = max(time.monotonic(), self._next)
            self._next = start + self.min_interval_s
        time.sleep(max(0.0, start - time.monotonic()))

    def _messages(self, req: AnswerRequest) -> list[dict]:
        content: list[dict] = []
        for f in req.frames:
            content.append({"type": "text", "text": f"Frame at {f.decoded_pts_s:.2f}s:"})
            content.append({"type": "image_url", "image_url": {"url": _jpeg(f.image, self.frame_max_side)}})
        content.append({"type": "text", "text": build_prompt_text(req)})
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    def _post(self, body: dict) -> dict:
        key = (Path.home() / self.key_file).read_text(encoding="utf-8-sig").strip()
        data = json.dumps(body).encode()
        for attempt in range(8):
            self._pace()
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(min(120, 10 * 2 ** attempt))
                    continue
                raise RuntimeError(f"API error {e.code}: {e.read()[:300]!r}") from e
            except (TimeoutError, urllib.error.URLError, ConnectionError):
                if attempt < 7:
                    time.sleep(min(120, 10 * 2 ** attempt))
                    continue
                raise
        raise RuntimeError("API kept failing")

    @staticmethod
    def letter_probs(logprobs: dict | None, n_options: int) -> list[float] | None:
        if not logprobs or not logprobs.get("content") or not n_options:
            return None
        letters = OPTION_LETTERS[:n_options]
        for tok in reversed(logprobs["content"]):
            if tok["token"].strip().rstrip(".:)") in letters:
                p = {x["token"].strip().rstrip(".:)"): math.exp(x["logprob"]) for x in tok.get("top_logprobs", [])}
                vals = [p.get(c, 0.0) for c in letters]
                s = sum(vals)
                return [v / s for v in vals] if s > 0 else None
        return None

    def answer(self, req: AnswerRequest):
        """One request; when a multiple-choice reply carries no option letter (about 13% of first replies: the
        reasoning used the whole budget and no final answer was written), ask again once with twice the budget."""
        messages = self._messages(req)
        for budget in (self.max_tokens, 2 * self.max_tokens):
            body = {"model": self.model_id, "messages": messages, "temperature": 0,
                    "max_tokens": budget, "logprobs": True, "top_logprobs": self.top_logprobs}
            out = self._post(body)
            choice = out["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            u = out.get("usage") or {}
            usage = AnswerUsage(text_tokens=int(u.get("prompt_tokens", 0)), generated_tokens=int(u.get("completion_tokens", 0)))
            probs = self.letter_probs(choice.get("logprobs"), len(req.options or []))
            ans = finalize_answer(text, req.options, probs, req.frames, req.excerpt)
            if not req.options or ans.option_index is not None or ans.abstained:
                break
        return ans, usage
