"""CPU-only experimental pairwise residual ranking on frozen cached features.

This is a prototype, not a novelty or performance claim. Scores are an anchored
MobileCLIP ranking plus a small L2-regularized linear residual. Retained-text
features describe lexical coverage; different numerals are NOT contradictions.
No labels, gold indices, encoders, images or torch are used at inference.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from .records import FrameCandidate, QuestionPool

BASE_FEATURES = (
    "clip", "head_a_relevance", "time_fraction", "path_a", "path_b", "path_both",
    "ocr_log_length", "ocr_question_overlap", "near_question_overlap",
    "ocr_option_max", "near_option_max", "ocr_question_cosine", "near_question_cosine",
    "image_ocr_cosine", "ocr_near_cosine", "ocr_log_tokens", "near_log_tokens",
)
CONTEXT_FEATURES = (
    "ocr_question_not_retained", "ocr_retained_overlap", "near_retained_overlap",
    "ocr_question_retained_overlap", "ocr_numerals_retained", "ocr_numerals_not_retained",
    "question_ocr_numerals_not_retained", "clip_x_question_missing",
    "ocr_cosine_x_question_missing", "near_cosine_x_question_missing",
)
RIDGES = (0.1, 1.0, 10.0, 100.0)
_STOP = set("the a an of to in is are was what which how does do and or for on at by with this that it as be".split())


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def _numbers(text):
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _overlap(left, right):
    return len(left & right) / max(1, len(right))


def _cosine(a, b):
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def feature_names(context=False):
    return BASE_FEATURES + (CONTEXT_FEATURES if context else ())


def raw_features(pool: QuestionPool, context=False):
    """Features use all options symmetrically, never the gold option/index.

    Base features never read pool.transcript. Nearby speech and cached Head A
    relevance remain available equally to both matched variants.
    """
    q = _tokens(pool.question)
    opts = [_tokens(o) for o in pool.options]
    retained = _tokens(pool.transcript_text) if context else set()
    retained_num = _numbers(pool.transcript_text) if context else set()
    missing = 1.0 - _overlap(retained, q)
    rows = []
    for c in pool.candidates:
        ocr, near = _tokens(c.ocr_text), _tokens(c.near_text)
        oc, nc = _cosine(c.ocr_emb, pool.question_emb), _cosine(c.near_emb, pool.question_emb)
        row = [c.clip_sim, c.head_a_visual, c.time_s / max(1.0, pool.duration_s),
               float("A" in c.paths), float("B" in c.paths), float("A" in c.paths and "B" in c.paths),
               np.log1p(len(c.ocr_text)), _overlap(ocr, q), _overlap(near, q),
               max((_overlap(ocr, o) for o in opts), default=0.0),
               max((_overlap(near, o) for o in opts), default=0.0), oc, nc,
               _cosine(c.emb, c.ocr_emb), _cosine(c.ocr_emb, c.near_emb),
               np.log1p(len(ocr)), np.log1p(len(near))]
        if context:
            onum, qnum = _numbers(c.ocr_text), _numbers(pool.question)
            row += [len((ocr & q) - retained) / max(1, len(q)), _overlap(retained, ocr),
                    _overlap(retained, near), len(ocr & q & retained) / max(1, len(q)),
                    _overlap(retained_num, onum), len(onum - retained_num) / max(1, len(onum)),
                    len((qnum & onum) - retained_num) / max(1, len(qnum)),
                    c.clip_sim * missing, oc * missing, nc * missing]
        rows.append(row)
    result = np.asarray(rows, dtype=float).reshape(len(rows), len(feature_names(context)))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite cached feature")
    return result


def design(pool: QuestionPool, context=False):
    """Label-free normalization over the whole question pool, not labeled rows.

    Context interaction scales use their base channel's scale, preserving the
    degree of missing retained coverage rather than normalizing it away.
    """
    x = raw_features(pool, context)
    if not len(x):
        return x, np.zeros(0)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    if context:
        for extra, base in ((-3, 0), (-2, 11), (-1, 12)):
            scale[extra] = scale[base]
    z = (x - center) / scale
    return z, z[:, 0].copy()


def ranked_indices(pool, scores, indices=None):
    indices = range(len(pool.candidates)) if indices is None else indices
    return sorted(indices, key=lambda i: (-float(scores[i]), pool.candidates[i].id))


@dataclasses.dataclass
class PairBlock:
    qa_id: str
    video_id: str
    x: np.ndarray
    target: np.ndarray


def pair_blocks(pools, rows, context=False):
    """Each question contributes a mean squared pairwise loss of equal weight.

    All unordered within-question pairs, including gain ties, fit the observed
    gain difference. Repeated (question, candidate) labels are rejected.
    """
    by = {}
    for row in rows:
        if row.scheme != "independent" or row.history:
            raise ValueError("Only randomly sampled independent single-frame rows are allowed")
        pool = pools[row.qa_id]
        if row.video_id != pool.video_id or not np.isclose(row.after - row.before, row.gain):
            raise ValueError("Inconsistent label metadata")
        by.setdefault(row.qa_id, []).append(row)
    blocks = []
    for q in sorted(by):
        rr, pool = by[q], pools[q]
        if len({r.candidate_id for r in rr}) != len(rr):
            raise ValueError("Duplicate candidate label")
        if len({r.before for r in rr}) != 1:
            raise ValueError("Independent rows must share a transcript-only baseline")
        x, anchor = design(pool, context)
        ix = {c.id: i for i, c in enumerate(pool.candidates)}
        dx, dy = [], []
        for i, a in enumerate(rr):
            for b in rr[i + 1:]:
                ai, bi = ix[a.candidate_id], ix[b.candidate_id]
                dx.append(x[ai] - x[bi])
                dy.append(a.gain - b.gain - (anchor[ai] - anchor[bi]))
        if dx:
            blocks.append(PairBlock(q, pool.video_id, np.asarray(dx), np.asarray(dy)))
    return blocks


@dataclasses.dataclass
class ResidualSelector:
    context: bool = False
    ridge: float | None = None
    weights: np.ndarray | None = None
    train_videos: tuple[str, ...] = ()
    name: str = "experimental_pairwise_residual"

    @classmethod
    def fit(cls, blocks, context=False, ridge=None):
        d = len(feature_names(context))
        w = np.zeros(d)
        if ridge is not None:
            if ridge <= 0:
                raise ValueError("Ridge must be positive")
            if blocks:
                # Mean across pairs within each question, then across questions.
                gram = sum(b.x.T @ b.x / len(b.target) for b in blocks) / len(blocks)
                rhs = sum(b.x.T @ b.target / len(b.target) for b in blocks) / len(blocks)
                w = np.linalg.solve(gram + ridge * np.eye(d), rhs)
        return cls(context, ridge, w, tuple(sorted({b.video_id for b in blocks})))

    def scores(self, pool):
        x, anchor = design(pool, self.context)
        return anchor + x @ (np.zeros(x.shape[1]) if self.weights is None else self.weights)

    def select(self, pool: QuestionPool, k: int) -> list[FrameCandidate]:
        return [pool.candidates[i] for i in ranked_indices(pool, self.scores(pool))[:max(0, k)]]

    def to_dict(self):
        return {"schema": 1, "context": self.context, "ridge": self.ridge,
                "weights": (np.zeros(len(feature_names(self.context))) if self.weights is None else self.weights).tolist(),
                "train_videos": list(self.train_videos), "feature_names": list(feature_names(self.context))}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, value):
        context = value["context"]
        if value["schema"] != 1 or value["feature_names"] != list(feature_names(context)):
            raise ValueError("Incompatible residual feature schema")
        w = np.asarray(value["weights"], dtype=float)
        if w.shape != (len(feature_names(context)),) or not np.isfinite(w).all():
            raise ValueError("Invalid residual coefficients")
        return cls(context, value["ridge"], w, tuple(value["train_videos"]))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def mmr_order(pool, indices=None, lam=0.7):
    """Existing raw-cosine MMR formula, deterministic candidate-ID tie break."""
    rest = list(range(len(pool.candidates))) if indices is None else list(indices)
    chosen = []
    while rest:
        def score(i):
            c = pool.candidates[i]
            redundancy = max((float(np.asarray(c.emb) @ np.asarray(pool.candidates[j].emb))
                              for j in chosen), default=0.0)
            return lam * c.clip_sim - (1 - lam) * redundancy
        best = min(rest, key=lambda i: (-score(i), pool.candidates[i].id))
        chosen.append(best)
        rest.remove(best)
    return chosen
