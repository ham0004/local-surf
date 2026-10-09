"""Head A (v3): question + timed speech -> moments where the visual evidence should be.

Inference side of the Head A cycle (docs/v3/heads_design.md; research_log steps 28-39). Two moment sources
plug into Path A of the frozen pipeline through ``moments(qa, segs, duration, n)``
(`videoqa.v2.candidates.build_pool`):

    ZeroShotMoments  frozen MiniLM relevance of every line (+1 neighbour each side), top line ends
    LearnedMoments   the light residual head (scripts/v3_head_a_light.py): score = frozen z + MLP(features),
                     moments = top line ends ("light") or peaks of the evidence density ("light_dens")

Records follow data/head_a: {"question", "options", "lines": [[start, end, text]], "duration_s"}.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

NMS_GAP_S = 8.0
GRID_S = 1.0
OFF_RANGE_S, OFF_BIN_S = 40.0, 4.0
N_BINS = int(2 * OFF_RANGE_S / OFF_BIN_S) + 1          # 21 bins, centre bin = offset 0
CUES = {
    "speech": ("say", "said", "says", "mention", "subtitle", "explain", "why", "talk", "according", "narrator"),
    "visual": ("color", "colour", "wear", "how many", "shape", "letters", "sign", "screen", "pattern", "logo",
               "written", "appear", "shown", "picture"),
    "order": ("after", "before", "when", "first", "last", "then", "finally"),
}
FEATURE_NAMES = ["z", "z_gap", "z_rank", "ctx1_max", "ctx1_mean", "ctx3_max", "ctx3_mean", "ctx7_max", "ctx7_mean",
                 "bm25_q", "bm25_qo", "opt_best", "opt_spread", "words", "digit", "position", "density",
                 "cue_speech", "cue_visual", "cue_order", "has_options"]
BGE_FEATURE_NAMES = ["bge_gap", "bge_rank", "bge_ctx3_max"]
_WORD = re.compile(r"[a-z0-9]+")


def _toks(t: str) -> set[str]:
    return {w for w in _WORD.findall(t.lower()) if len(w) > 2}


def line_times(rec) -> np.ndarray:
    """Frame time per line: 0.3 s before the line ends (v2's rule), inside the video."""
    return np.asarray([max(0.0, min(rec["duration_s"] - 0.05, b - 0.3)) for _, b, _ in rec["lines"]])


def contexts(rec) -> list[str]:
    lines = rec["lines"]
    return [" ".join(x[2] for x in lines[max(0, i - 1): i + 2]) for i in range(len(lines))]


def nms(times: list[float], gap: float = NMS_GAP_S) -> list[float]:
    out: list[float] = []
    for t in times:
        if all(abs(t - u) >= gap for u in out):
            out.append(t)
    return out


def density_moments(rec, rel: np.ndarray, off: np.ndarray | None, top_lines: int = 64,
                    temperature: float = 1.0, sigma_s: float = 4.0) -> list[float]:
    """Peaks of D(t) = sum_i softmax(rel/T)_i * q_i(t - t_i); q_i = learned offset bins, or a Gaussian at 0."""
    if len(rel) == 0:
        return []
    t = line_times(rec)
    keep = np.argsort(-rel)[:top_lines]
    w = np.exp((rel[keep] - rel[keep].max()) / temperature)
    w /= w.sum()
    grid = np.arange(0.0, rec["duration_s"], GRID_S)
    dens = np.zeros_like(grid)
    centres = (np.arange(N_BINS) - N_BINS // 2) * OFF_BIN_S
    for wi, i in zip(w, keep, strict=True):
        d = grid - t[i]
        if off is None:
            dens += wi * np.exp(-0.5 * (d / sigma_s) ** 2)
        else:
            # mixture of Gaussians at the bin centres (width = half a bin) weighted by the offset distribution
            dens += wi * (off[i][None, :] * np.exp(-0.5 * ((d[:, None] - centres[None, :]) / (OFF_BIN_S / 2)) ** 2)).sum(1)
    order = np.argsort(-dens, kind="stable")
    return nms([float(grid[j]) for j in order[:2000]], NMS_GAP_S)[:16]


def line_features(rec, z: np.ndarray) -> np.ndarray:
    """(lines, len(FEATURE_NAMES)) label-free features; z = frozen zero-shot relevance per line."""
    from ..retrieval import bm25_scores, tokenize  # noqa: PLC0415

    lines, n = rec["lines"], len(rec["lines"])
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    texts = [x[2] for x in lines]
    order = np.argsort(np.argsort(-z))
    ctx = []
    for w in (1, 3, 7):
        mx = np.array([z[max(0, i - w): i + w + 1].max() for i in range(n)])
        mn = np.array([z[max(0, i - w): i + w + 1].mean() for i in range(n)])
        ctx += [mx - z.max(), mn - z.max()]
    corpus = [tokenize(t) for t in texts]
    bq = np.asarray(bm25_scores(tokenize(rec["question"]), corpus))
    bqo = np.asarray(bm25_scores(tokenize(rec["question"] + " " + " ".join(rec["options"] or [])), corpus))
    opts = [_toks(o) for o in (rec["options"] or [])]
    ov = np.array([[len(_toks(t) & o) / (len(o) + 1) for o in opts] for t in texts]) if opts else np.zeros((n, 1))
    t = line_times(rec)
    dens = np.array([np.sum(np.abs(t - x) <= 15.0) for x in t])
    q = rec["question"].lower()
    cue = [float(any(c in q for c in CUES[k])) for k in ("speech", "visual", "order")]
    rows = np.column_stack([
        z, z - z.max(), order / max(n - 1, 1), *ctx,
        bq / (bq.max() + 1e-6), bqo / (bqo.max() + 1e-6), ov.max(1), ov.max(1) - ov.min(1),
        np.log1p([len(x.split()) for x in texts]), [float(bool(re.search(r"\d", x))) for x in texts],
        t / max(rec["duration_s"], 1.0), np.log1p(dens),
        np.tile(cue, (n, 1)), np.full(n, float(bool(rec["options"])))])
    return rows.astype(np.float32)


def bge_features(z2: np.ndarray) -> np.ndarray:
    n = len(z2)
    ctx3 = np.array([z2[max(0, i - 3): i + 4].max() for i in range(n)])
    order = np.argsort(np.argsort(-z2))
    return np.column_stack([z2 - z2.max(), order / max(n - 1, 1), ctx3 - z2.max()]).astype(np.float32)


def build_net(d_in: int, width: int, offsets: bool):
    import torch  # noqa: PLC0415

    net = torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.GELU(), torch.nn.Linear(width, width),
                              torch.nn.GELU(), torch.nn.Linear(width, 1 + (N_BINS if offsets else 0)))
    torch.nn.init.zeros_(net[-1].weight)          # untrained head = the zero-shot rule
    torch.nn.init.zeros_(net[-1].bias)
    return net


def record_of(qa, segs, duration: float) -> dict:
    return {"question": qa.question, "options": list(qa.options or []) or None, "duration_s": float(duration),
            "lines": [[s.start_s, s.end_s, s.text] for s in segs]}


class ZeroShotMoments:
    """Frozen MiniLM relevance over every line of the video; moments = the best line ends."""

    def __init__(self, scorer) -> None:
        self.scorer = scorer                       # videoqa.v2.head_a.HotMomentScorer

    def score(self, question: str, texts: list[str], options=None) -> np.ndarray:
        """The scorer's per-line scores (build_pool attaches them to candidates as features)."""
        return self.scorer.score(question, texts, options)

    def relevance(self, rec) -> np.ndarray:
        ctx = contexts(rec)
        if not ctx:
            return np.zeros(0)
        return np.concatenate([self.scorer.score(rec["question"], ctx[i:i + 256])[:, 1]
                               for i in range(0, len(ctx), 256)])

    def moments(self, qa, segs, duration: float, n: int) -> list[float]:
        rec = record_of(qa, segs, duration)
        rel = self.relevance(rec)
        return nms([float(x) for x in line_times(rec)[np.argsort(-rel, kind="stable")[:64]]], 1.0)[:n]


class LearnedMoments(ZeroShotMoments):
    """The trained light residual head (checkpoint from scripts/v3_head_a_light.py --save)."""

    def __init__(self, scorer, checkpoints, bge=None) -> None:
        """checkpoints: one path or several (same method/features; their scores are averaged, e.g. 3 seeds)."""
        import torch  # noqa: PLC0415

        super().__init__(scorer)
        paths = [checkpoints] if isinstance(checkpoints, (str, Path)) else list(checkpoints)
        self.heads = []
        for path in paths:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            net = build_net(len(ck["mu"]), ck["width"], ck["method"] == "light_off")
            net.load_state_dict(ck["state"])
            net.eval()
            self.heads.append((net, np.asarray(ck["mu"]), np.asarray(ck["sd"])))
            self.method, self.feature_set = ck["method"], ck["feature_set"]
        self.bge = bge                             # a reranker with .score(query, texts) for feature_set "ens"
        if self.feature_set == "ens" and bge is None:
            raise ValueError("this checkpoint needs the bge-reranker-base scorer")

    def scores(self, rec):
        import torch  # noqa: PLC0415

        z = self.relevance(rec)
        f = line_features(rec, z)
        if self.feature_set == "ens":
            q = rec["question"] + (" Options: " + " | ".join(rec["options"]) if rec["options"] else "")
            f = np.concatenate([f, bge_features(self.bge.score(q, contexts(rec)))], axis=1)
        with torch.no_grad():
            o = np.mean([net(torch.as_tensor((f - mu) / sd, dtype=torch.float32)).numpy()
                         for net, mu, sd in self.heads], axis=0)
        off = None
        if self.method == "light_off":
            e = np.exp(o[:, 1:] - o[:, 1:].max(1, keepdims=True))
            off = e / e.sum(1, keepdims=True)
        return z + o[:, 0], off

    def moments(self, qa, segs, duration: float, n: int) -> list[float]:
        rec = record_of(qa, segs, duration)
        rel, off = self.scores(rec)
        if self.method == "light":
            return nms([float(x) for x in line_times(rec)[np.argsort(-rel, kind="stable")[:64]]], 1.0)[:n]
        return density_moments(rec, rel, off)[:n]
