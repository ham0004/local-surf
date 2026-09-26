"""Frozen visual scout: cheap, NON-answer-bearing hints about candidate frames.

The scout's output type is :class:`~videoqa.schemas.ScoutSignals` - five
numbers/categories and nothing else.  It must never return OCR text, captions
or object names in the main experiment: those could answer the question before
the final VLM sees the frame, which would make the controller's evidence
selection impossible to attribute.  (A content-rich scout would be a separate,
clearly labelled ablation.)

Backends
--------
PixelStatsScout  no model, CPU only.  Scene type, change and quality come from
                 simple image statistics.  It has NO question awareness, so it
                 reports similarity=0.5 with uncertainty=1.0 - honest "don't
                 know" rather than a fake score.
OpenClipScout    frozen open_clip image/text encoder (e.g. MobileCLIP).
                 similarity = cosine(question, frame) rescaled to [0, 1].
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from .schemas import Frame, SceneType, ScoutSignals


class Scout(Protocol):
    name: str

    def score(self, question: str, frames: list[Frame]) -> dict[str, ScoutSignals]:
        """Return signals keyed by frame id.  ``frames`` may be in any order."""
        ...


# ---------------------------------------------------------------------------
# Shared image statistics
# ---------------------------------------------------------------------------


def _gray(img: Image.Image, side: int = 128) -> np.ndarray:
    return np.asarray(img.convert("L").resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def sharpness(img: Image.Image) -> float:
    """Variance of a 4-neighbour Laplacian, squashed to [0, 1].  Blurry or
    blank frames score low."""
    g = _gray(img)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(1.0 - np.exp(-lap.var() * 50.0))


def change_between(a: Image.Image | None, b: Image.Image, side: int = 64, grid: int = 8) -> float:
    """Largest block-wise mean absolute RGB difference, in [0, 1].

    A global mean dilutes small local changes (an indicator switching colour
    moves the global mean by <0.04), so we split the frame into a grid x grid
    layout of blocks and report the most-changed block, scaled by 2.  RGB, not
    grayscale, because red->green can be nearly invisible in luminance.
    """
    if a is None:
        return 0.0
    rgb_a = np.asarray(a.convert("RGB").resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    rgb_b = np.asarray(b.convert("RGB").resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    diff = np.abs(rgb_a - rgb_b).mean(axis=2)
    cell = side // grid
    block_means = diff.reshape(grid, cell, grid, cell).mean(axis=(1, 3))
    return float(min(1.0, block_means.max() * 2.0))


def classify_scene(img: Image.Image) -> tuple[SceneType, float]:
    """Very coarse scene type from colour statistics, with an uncertainty.

    Slides/boards/code share 'large uniform background + sparse strokes'; we
    only separate bright-background (SLIDE), dark-background (CODE, typical of
    terminals), near-uniform (BLANK) and everything else (NATURAL).  This is a
    weak heuristic and says so via high uncertainty.
    """
    rgb = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
    g = rgb.mean(axis=2)
    if g.std() < 0.02:
        return SceneType.BLANK, 0.2
    # fraction of pixels close to the dominant (median) brightness = background share
    background = float((np.abs(g - np.median(g)) < 0.08).mean())
    if background > 0.6:
        return (SceneType.SLIDE if np.median(g) > 0.5 else SceneType.CODE), 0.5
    return SceneType.NATURAL, 0.6


def _temporal_changes(frames: list[Frame]) -> dict[str, float]:
    """visual_change for each frame = difference from the previous frame in time
    among the frames being scored (the scout never decodes extra frames)."""
    out: dict[str, float] = {}
    prev = None
    for f in sorted(frames, key=lambda f: f.decoded_pts_s):
        out[f.id] = change_between(prev, f.image)
        prev = f.image
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class PixelStatsScout:
    """Model-free scout.  Cheap and question-agnostic by construction."""

    name = "pixel_stats"

    def score(self, question: str, frames: list[Frame]) -> dict[str, ScoutSignals]:
        changes = _temporal_changes(frames)
        out = {}
        for f in frames:
            scene, scene_unc = classify_scene(f.image)
            out[f.id] = ScoutSignals(
                similarity=0.5,          # no text encoder -> "unknown"
                scene_type=scene,
                visual_change=changes[f.id],
                quality=sharpness(f.image),
                uncertainty=1.0,         # similarity is uninformative here
            )
        return out


class OpenClipScout:
    """Frozen open_clip encoder (default: a MobileCLIP checkpoint).

    Model weights are loaded lazily on first use.  Exact checkpoint names must
    be verified against the installed open_clip version (see docs/models.md);
    we do not hard-code a claim that any particular one is available.
    """

    name = "open_clip"

    def __init__(self, model_name: str = "MobileCLIP-S2", pretrained: str = "datacompdr",
                 device: str = "cpu", cache_dir: str | None = None) -> None:
        self.model_name, self.pretrained, self.device = model_name, pretrained, device
        self.cache_dir = cache_dir
        self._model = self._preprocess = self._tokenizer = None

    def _load(self) -> None:
        import open_clip  # noqa: PLC0415 - heavy optional dependency
        import torch  # noqa: PLC0415

        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained,
                                                                     cache_dir=self.cache_dir)
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._torch = torch

    def ensure_loaded(self) -> None:
        """Load weights now, so the pipeline can meter loading separately
        (cold cost) from per-question scoring (warm cost)."""
        if self._model is None:
            self._load()

    def score(self, question: str, frames: list[Frame]) -> dict[str, ScoutSignals]:
        if not frames:
            return {}
        self.ensure_loaded()
        torch = self._torch
        with torch.no_grad():
            images = torch.stack([self._preprocess(f.image) for f in frames]).to(self.device)
            img_emb = self._model.encode_image(images)
            txt_emb = self._model.encode_text(self._tokenizer([question]).to(self.device))
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cos = (img_emb @ txt_emb.T).squeeze(-1).float().cpu().numpy()
        # CLIP cosines for unrelated pairs cluster around 0.1-0.3; rescale that
        # typical range to [0, 1].  This is a *similarity* hint, not a
        # calibrated probability that the frame answers the question.
        sims = np.clip((cos - 0.1) / 0.25, 0.0, 1.0)
        changes = _temporal_changes(frames)
        out = {}
        for f, s in zip(frames, sims, strict=True):
            scene, scene_unc = classify_scene(f.image)
            out[f.id] = ScoutSignals(similarity=float(s), scene_type=scene, visual_change=changes[f.id],
                                     quality=sharpness(f.image), uncertainty=float(scene_unc * 0.5))
        return out


def make_scout(backend: str, device: str = "cpu", **kwargs) -> Scout:
    if backend == "pixel_stats":
        return PixelStatsScout()
    if backend == "open_clip":
        return OpenClipScout(device=device, **kwargs)
    raise ValueError(f"unknown scout backend {backend!r}")
