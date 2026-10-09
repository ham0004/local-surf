"""Frozen feature extractors, run once per unique frame and cached.

* MobileCLIP-S2 (open_clip): L2-normalised image and text embeddings. The same
  model v1 uses as its scout, so the visual scan and Head B share one encoding
  per frame ("one pass").
* RapidOCR (ONNX, CPU): the text visible on the frame (slides / boards).

Nothing here is trained. Each method takes plain inputs and returns numpy arrays
or strings, so the stages above can be tested with a fake encoder.
"""

from __future__ import annotations

import time

import numpy as np

from ..scout import OpenClipScout


class FrozenEncoders:
    """MobileCLIP embeddings + OCR, with simple wall-time counters for the cost report."""

    def __init__(self, device: str = "cuda", cache_dir: str | None = None, use_ocr: bool = True) -> None:
        self._clip = OpenClipScout(device=device, cache_dir=cache_dir)   # reuse v1's loader (frozen)
        self._ocr = None
        self.use_ocr = use_ocr
        self.seconds = {"clip_image": 0.0, "clip_text": 0.0, "ocr": 0.0}
        self.counts = {"clip_image": 0, "clip_text": 0, "ocr": 0}

    # -- MobileCLIP --------------------------------------------------------
    def embed_images(self, images: list, batch: int = 64) -> np.ndarray:
        """(N, D) unit-norm image embeddings for PIL images.

        Encoded in batches of ``batch``: a whole-video scan can be several hundred frames, which in
        one batch exhausted a 16 GB GPU (5 GB batch-norm allocation on a long Video-MMMU video).
        """
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        self._clip.ensure_loaded()
        torch, model, prep = self._clip._torch, self._clip._model, self._clip._preprocess
        t0 = time.perf_counter()
        parts = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.stack([prep(im) for im in images[i:i + batch]]).to(self._clip.device)
                f = model.encode_image(x)
                parts.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        e = np.concatenate(parts)
        self.seconds["clip_image"] += time.perf_counter() - t0
        self.counts["clip_image"] += len(images)
        return e

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """(N, D) unit-norm text embeddings (MobileCLIP text tower; truncates at 77 tokens)."""
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)
        self._clip.ensure_loaded()
        torch, model, tok = self._clip._torch, self._clip._model, self._clip._tokenizer
        t0 = time.perf_counter()
        with torch.no_grad():
            e = model.encode_text(tok([t if t.strip() else "(empty)" for t in texts]).to(self._clip.device))
            e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        self.seconds["clip_text"] += time.perf_counter() - t0
        self.counts["clip_text"] += len(texts)
        return e

    # -- OCR ---------------------------------------------------------------
    def ocr(self, image) -> str:
        """Visible text on one frame ("" if OCR is disabled or finds nothing)."""
        if not self.use_ocr:
            return ""
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415 - optional dependency

            self._ocr = RapidOCR()
        t0 = time.perf_counter()
        result, _ = self._ocr(np.asarray(image.convert("RGB")))
        self.seconds["ocr"] += time.perf_counter() - t0
        self.counts["ocr"] += 1
        return " ".join(r[1] for r in (result or []))
