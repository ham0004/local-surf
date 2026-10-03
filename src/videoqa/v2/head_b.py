"""Head B: utility(candidate | question, retained transcript, selected history).

Predicts the signed change in answer correctness from ADDING one candidate frame
to the frames already selected:

    g(c | q, T, S) ~ R(q, T, S + {c}) - R(q, T, S)        in [-1, +1]

Inputs are only what is available at inference time (never the gold answer):

  content    MobileCLIP (frame x question), (OCR text x question), (nearby speech x question)
             element-wise products -> small learned projections
  scalars    similarities, OCR length/overlap, Head A text/visual scores, path flags,
             position in the video, distance to the retained transcript
  history    how many frames are selected, max/mean visual similarity to them,
             nearest time gap, OCR overlap with them  (all zero when S is empty)

Model: three 512->8 projections + scalars -> 64 -> 1 MLP (~13k parameters).
Loss: Huber regression on the gain + pairwise ranking between candidates that
share the SAME context and have DIFFERENT gains (ties are never ordered).
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from .records import FrameCandidate, QuestionPool

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in is are was what which how does do and or for on at by with this that it as be".split())
N_SCALAR = 20


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


# ---------------------------------------------------------------------------
# Feature construction (pure numpy; shared by training and inference)
# ---------------------------------------------------------------------------


def featurize(pool: QuestionPool, cand: FrameCandidate, history: list[FrameCandidate],
              use_ocr: bool = True) -> np.ndarray:
    """One feature vector for `cand` given the selected `history` (length 3*512 + N_SCALAR).

    ``use_ocr=False`` zeroes every OCR-derived feature, so a selector trained and run
    this way needs no OCR at inference (and pays no OCR time).
    """
    q = pool.question_emb
    q_tok = _tokens(pool.question + " " + " ".join(pool.options))
    ocr_tok, near_tok = _tokens(cand.ocr_text), _tokens(cand.near_text)
    retained_t = [0.5 * (s.start_s + s.end_s) for s in pool.transcript] or [0.0]
    dur = max(pool.duration_s, 1.0)

    ocr_emb = cand.ocr_emb if use_ocr else np.zeros_like(cand.ocr_emb)
    if not use_ocr:
        ocr_tok = set()
    content = np.concatenate([cand.emb * q, ocr_emb * q, cand.near_emb * q])
    scalars = [
        cand.clip_sim, float(ocr_emb @ q), float(cand.near_emb @ q),
        (np.log1p(len(cand.ocr_text)) / 6.0) if use_ocr else 0.0,
        len(ocr_tok & q_tok) / (len(q_tok) + 1), len(near_tok & q_tok) / (len(q_tok) + 1),
        cand.head_a_text / 10.0, cand.head_a_visual / 10.0,
        float("A" in cand.paths), float("B" in cand.paths), float(len(cand.paths) == 2),
        cand.time_s / dur,
        min(abs(cand.time_s - t) for t in retained_t) / dur,
    ]
    # -- history features (zero when nothing is selected yet) -----------------
    if history:
        sims = [float(cand.emb @ h.emb) for h in history]
        ocr_sims = [float(ocr_emb @ h.ocr_emb) if use_ocr else 0.0 for h in history]
        gaps = [abs(cand.time_s - h.time_s) for h in history]
        scalars += [len(history) / 8.0, max(sims), float(np.mean(sims)), min(gaps) / dur, max(ocr_sims),
                    max(h.clip_sim for h in history), float(any(set(h.paths) & set(cand.paths) for h in history))]
    else:
        scalars += [0.0] * 7
    assert len(scalars) == N_SCALAR
    return np.concatenate([content, np.asarray(scalars, dtype=np.float32)]).astype(np.float32)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HeadBConfig:
    proj_dim: int = 8
    hidden: int = 64
    epochs: int = 200
    lr: float = 3e-3
    weight_decay: float = 1e-3
    rank_weight: float = 0.5
    dropout: float = 0.1
    seed: int = 0


class HeadB:
    """Small MLP scorer. ``fit`` on (features, gain, context id); ``predict`` on features."""

    def __init__(self, cfg: HeadBConfig = HeadBConfig()) -> None:
        import torch  # noqa: PLC0415

        self.cfg, self.torch = cfg, torch
        torch.manual_seed(cfg.seed)
        d = 512
        nn = torch.nn
        self.proj = nn.ModuleList([nn.Linear(d, cfg.proj_dim) for _ in range(3)])
        self.mlp = nn.Sequential(nn.Linear(3 * cfg.proj_dim + N_SCALAR, cfg.hidden), nn.ReLU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 1))
        self.scalar_mean = self.scalar_std = None

    def _forward(self, X):
        torch = self.torch
        parts = [p(X[:, i * 512:(i + 1) * 512]) for i, p in enumerate(self.proj)]
        s = (X[:, 3 * 512:] - self.scalar_mean) / self.scalar_std
        return self.mlp(torch.cat(parts + [s], dim=1)).squeeze(-1)

    def parameters(self):
        return list(self.proj.parameters()) + list(self.mlp.parameters())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fit(self, X: np.ndarray, y: np.ndarray, contexts: np.ndarray) -> dict:
        torch, cfg = self.torch, self.cfg
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.scalar_mean = Xt[:, 3 * 512:].mean(0)
        self.scalar_std = Xt[:, 3 * 512:].std(0) + 1e-6
        I, J = _rank_pairs(contexts, y)
        It, Jt = torch.tensor(I, dtype=torch.long), torch.tensor(J, dtype=torch.long)
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for m in (self.proj, self.mlp):
            m.train()
        for _ in range(cfg.epochs):
            pred = self._forward(Xt)
            loss = torch.nn.functional.huber_loss(pred, yt, delta=0.5)
            if len(I):
                sign = torch.sign(yt[It] - yt[Jt])
                loss = loss + cfg.rank_weight * torch.nn.functional.softplus(-sign * (pred[It] - pred[Jt])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        for m in (self.proj, self.mlp):
            m.eval()
        return {"rows": len(y), "rank_pairs": len(I), "final_loss": float(loss.detach()), "params": self.n_params()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return self._forward(torch.tensor(X, dtype=torch.float32)).numpy()


def _rank_pairs(contexts: np.ndarray, y: np.ndarray):
    """Index pairs within the same context whose gains differ."""
    I, J = [], []
    for ctx in np.unique(contexts):
        idx = np.where(contexts == ctx)[0]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if y[idx[a]] != y[idx[b]]:
                    I.append(idx[a])
                    J.append(idx[b])
    return I, J
