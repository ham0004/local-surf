"""Real frozen VLM answerer via Hugging Face transformers.

Default: Qwen/Qwen3-VL-2B-Instruct at a pinned revision (see configs and
docs/models.md).  Loaded lazily on first use, kept frozen (eval + no_grad).

Token accounting is MEASURED, not estimated: visual tokens are the number of
image placeholder tokens the processor actually inserted into input_ids, and
text tokens are the remainder. ``visual_tokens(frames)`` gives the exact count
for specific images BEFORE generation (from the image processor's patch grid),
so a frame can be refused when it would exceed the budget. (An earlier version
admitted frames using a running average from previous questions.)

For multiple choice we also return ``confidence`` = softmax over the option-
letter logits at the first generated position.  It is an UNCALIBRATED model
score, stored for analysis and for the confidence-threshold baseline only.
"""

from __future__ import annotations

import hashlib

from .answerer import SYSTEM_PROMPT, AnswerRequest, AnswerUsage, build_prompt_text, finalize_answer
from .schemas import Answer, Frame


class HFVLMAnswerer:
    name = "hf_vlm"

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16", max_new_tokens: int = 64,
                 revision: str | None = None, cache_dir: str | None = None) -> None:
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.max_new_tokens, self.revision, self.cache_dir = max_new_tokens, revision, cache_dir
        self._model = self._processor = None

    def _ensure_processor(self) -> None:
        """The processor alone (no weights): enough to count tokens exactly."""
        if self._processor is None:
            from transformers import AutoProcessor  # noqa: PLC0415

            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision,
                                                            cache_dir=self.cache_dir)

    def visual_tokens(self, frames: list[Frame]) -> int:
        """Exact visual tokens these images will cost, computed before generation
        from the image processor's patch grid (verified equal to the placeholder
        count in real calls: 220 for a 640x360 frame)."""
        if not frames:
            return 0
        self._ensure_processor()
        ip = self._processor.image_processor
        grid = ip(images=[f.image for f in frames], return_tensors="pt")["image_grid_thw"]
        return int(sum(int(t) * int(h) * int(w) for t, h, w in grid.tolist()) // ip.merge_size ** 2)

    def count_text_tokens(self, text: str) -> int:
        self._ensure_processor()
        return len(self._processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def fingerprint(self) -> dict:
        """What must be recorded with every label so it can be reproduced."""
        return {"model_id": self.model_id, "revision": self.revision, "dtype": self.dtype,
                "max_new_tokens": self.max_new_tokens, "decoding": "greedy",
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}

    def _load(self) -> None:
        import torch  # noqa: PLC0415 - heavy optional dependency
        from transformers import AutoModelForImageTextToText  # noqa: PLC0415

        self._torch = torch
        kw = dict(revision=self.revision, cache_dir=self.cache_dir)
        self._ensure_processor()
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=getattr(torch, self.dtype), device_map=self.device, **kw
        ).eval()
        cfg = self._model.config
        self._image_token_id = getattr(cfg, "image_token_id", None)
        if self._image_token_id is None:
            raise RuntimeError(f"{self.model_id}: config has no image_token_id; cannot meter visual tokens")

    def _messages(self, req: AnswerRequest) -> list[dict]:
        """Interleave a timestamp label before each frame so the model can cite it."""
        content: list[dict] = []
        for f in req.frames:
            content.append({"type": "text", "text": f"Frame at {f.decoded_pts_s:.2f}s:"})
            content.append({"type": "image", "image": f.image})
        content.append({"type": "text", "text": build_prompt_text(req)})
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]

    def ensure_loaded(self) -> None:
        """Load weights now so loading is metered as its own (cold) stage."""
        if self._model is None:
            self._load()

    def answer(self, req: AnswerRequest) -> tuple[Answer, AnswerUsage]:
        self.ensure_loaded()
        torch = self._torch
        inputs = self._processor.apply_chat_template(
            self._messages(req), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        ids = inputs["input_ids"][0]
        n_visual = int((ids == self._image_token_id).sum())
        usage = AnswerUsage(visual_tokens=n_visual, text_tokens=int(ids.numel()) - n_visual)

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       output_scores=True, return_dict_in_generate=True)
        new_tokens = out.sequences[0, ids.numel():]
        text = self._processor.batch_decode(new_tokens[None], skip_special_tokens=True)[0].strip()
        usage.text_tokens += int(new_tokens.numel())

        probs = None
        if req.options:
            letters = "ABCDEFGH"[: len(req.options)]
            letter_ids = [self._processor.tokenizer.convert_tokens_to_ids(c) for c in letters]
            probs = torch.softmax(out.scores[0][0, letter_ids].float(), dim=-1).tolist()
        return finalize_answer(text, req.options, probs, req.frames, req.excerpt), usage
