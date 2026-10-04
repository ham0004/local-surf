"""Local (spatial) board features for completion candidates. Frozen models only.

Hypothesis under test (declared in docs/v2/novelty_check.md before results):
the global features of a frame (one MobileCLIP vector, flattened OCR) cannot
tell WHERE on the board the question-relevant content is, nor whether a
candidate shows board content that the three already-chosen frames do not.
These features add that, without any trained encoder:

  tile relevance   MobileCLIP similarity of the question to each of 8 square
                   tiles (2 rows x 4 columns, 180 px, overlapping) that
                   together cover the whole 640x360 frame. Unlike the global
                   scout, nothing is centre-cropped away.
  novel ink        chalk pixels (brighter than the tile median + 35) present in
                   the candidate but absent from EVERY anchor frame, per tile.
                   Pixel-aligned: valid when the camera is static between the
                   frames, which is checked (global alignment error) and given
                   to the model as a feature rather than assumed.
  relevant novelty novel ink weighted by the tile's question relevance: new
                   board content in the place the question points to.
  tile redundancy  best cosine between the candidate's tiles and any anchor
                   tile (local, not global, repetition).

Cost: 8 extra MobileCLIP image encodings per frame plus numpy ink maps; this
is charged in the report. Labels and gold answers are never inputs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

TILE_SIDE, ROWS, COLS = 180, 2, 4
INK_DELTA = 35.0
LOCAL_FEATURES = (
    "tile_rel_max", "tile_rel_top2_mean", "tile_rel_spread",
    "novel_ink_area", "novel_ink_relevant", "novel_ink_in_best_tile",
    "tile_redundancy_max", "camera_misalignment",
)


def tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Overlapping square tiles covering the full frame (left-to-right, top-to-bottom)."""
    s = min(TILE_SIDE, width, height)
    ys = np.linspace(0, height - s, ROWS).round().astype(int)
    xs = np.linspace(0, width - s, COLS).round().astype(int)
    return [(int(x), int(y), int(x) + s, int(y) + s) for y in ys for x in xs]


def tile_images(im: Image.Image) -> list[Image.Image]:
    return [im.crop(b) for b in tile_boxes(*im.size)]


def ink_map(im: Image.Image) -> np.ndarray:
    """Boolean chalk mask at full resolution, thresholded against each tile's own median."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    out = np.zeros(g.shape, dtype=bool)
    for x0, y0, x1, y1 in tile_boxes(g.shape[1], g.shape[0]):
        t = g[y0:y1, x0:x1]
        out[y0:y1, x0:x1] |= t > np.median(t) + INK_DELTA
    return out


def _tile_share(mask: np.ndarray, boxes) -> np.ndarray:
    return np.array([mask[y0:y1, x0:x1].mean() for x0, y0, x1, y1 in boxes])


def local_features(cand_img: Image.Image, cand_tiles_emb: np.ndarray, anchor_imgs: list[Image.Image],
                   anchor_tiles_emb: list[np.ndarray], question_emb: np.ndarray) -> list[float]:
    """One row of LOCAL_FEATURES for a candidate given the anchor frames."""
    boxes = tile_boxes(*cand_img.size)
    rel = cand_tiles_emb @ question_emb                               # (8,)
    top2 = np.sort(rel)[-2:]
    cand_ink = ink_map(cand_img)
    anchor_inks = [ink_map(a) for a in anchor_imgs]
    novel = cand_ink.copy()
    for m in anchor_inks:
        novel &= ~m
    novel_t = _tile_share(novel, boxes)
    w = np.exp((rel - rel.max()) / 0.01)                               # soft focus on the most relevant tiles
    w /= w.sum()
    grey = np.asarray(cand_img.convert("L"), dtype=np.float32)
    misalign = min((float(np.abs(grey - np.asarray(a.convert("L"), dtype=np.float32)).mean()) for a in anchor_imgs),
                   default=0.0)
    redundancy = max((float((cand_tiles_emb @ t.T).max()) for t in anchor_tiles_emb), default=0.0)
    return [float(rel.max()), float(top2.mean()), float(rel.max() - rel.min()),
            float(novel.mean()), float(w @ novel_t), float(novel_t[int(rel.argmax())]),
            redundancy, misalign]
