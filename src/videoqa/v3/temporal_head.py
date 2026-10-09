"""Head A: question-conditioned temporal localisation over frozen frame features ("where to look").

Inputs (frozen, label-free at inference):
    emb   (N, 512)  MobileCLIP-S2 image embeddings of frames every ``step`` seconds (unit norm)
    times (N,)      their timestamps (s); duration (s)
    q     (512,)    MobileCLIP-S2 text embedding of the question (unit norm)

Model: a RESIDUAL over the MobileCLIP baseline score s_t = emb_t . q

    score_t = s_t / tau + r_t,     r_t = MLP(Transformer(FiLM_q(W emb_t) + pos(t)))

The residual's output layer starts at zero, so an untrained head ranks frames exactly like the
MobileCLIP baseline; training moves the ranking only where human evidence disagrees with it. The
transformer over time lets the score use context (neighbouring moments, position in the video).

Supervision (training videos only): CG-Bench human clue intervals. Target distribution over time bins
p_t proportional to 1 for bins inside an interval (padded by ``pad`` s); loss = cross-entropy between p
and softmax(score). Labels and gold answers are never inputs.

Selection: top-K peaks at least ``min_sep`` s apart (the same rule as the MobileCLIP baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def select_peaks(scores: np.ndarray, times: np.ndarray, k: int = 4, min_sep: float = 8.0) -> list[float]:
    """Greedy top-k by score with a minimum time separation; returns sorted times."""
    chosen: list[float] = []
    for i in np.argsort(-scores, kind="stable"):
        t = float(times[i])
        if all(abs(t - c) >= min_sep for c in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    return sorted(chosen)


def inside(t: float, intervals, pad: float = 0.0) -> bool:
    return any(a - pad <= t <= b + pad for a, b in intervals)


def hit_at_k(chosen: list[float], intervals, pad: float = 1.0) -> float:
    """1 if any chosen time lies inside a (padded) evidence interval."""
    return float(any(inside(t, intervals, pad) for t in chosen))


def target_distribution(times: np.ndarray, intervals, pad: float = 2.0) -> np.ndarray:
    m = np.array([inside(float(t), intervals, pad) for t in times], dtype=np.float32)
    if m.sum() == 0:                                   # interval between bins: nearest bin
        c = np.array([0.5 * (a + b) for a, b in intervals])
        m[int(np.argmin(np.abs(times[:, None] - c[None, :]).min(1)))] = 1.0
    return m / m.sum()


@dataclass
class HeadAConfig:
    hidden: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    tau: float = 0.02          # temperature of the MobileCLIP baseline term (its scores span ~0.1-0.35)
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 30
    pad: float = 2.0


def build_model(cfg: HeadAConfig, dim: int = 512):
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class TemporalHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = cfg.hidden
            self.frame = nn.Linear(dim + 1, h)          # frame embedding + its similarity to the question
            self.film = nn.Linear(dim, 2 * h)           # question -> per-channel scale and shift
            self.time = nn.Linear(8, h)                 # sinusoidal features of relative and absolute time
            layer = nn.TransformerEncoderLayer(h, cfg.heads, 2 * h, cfg.dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg.layers)
            self.out = nn.Linear(h, 1)
            nn.init.zeros_(self.out.weight)             # residual starts at 0: untrained head == MobileCLIP
            nn.init.zeros_(self.out.bias)

        @staticmethod
        def time_features(times, duration):
            rel = times / duration.clamp(min=1.0)
            minutes = times / 60.0
            feats = [torch.sin(2 * math.pi * rel), torch.cos(2 * math.pi * rel), rel,
                     torch.sin(2 * math.pi * rel * 4), torch.cos(2 * math.pi * rel * 4),
                     torch.log1p(minutes), torch.sin(minutes), torch.cos(minutes)]
            return torch.stack(feats, -1)

        def forward(self, emb, q, times, duration):
            """emb (B,N,D), q (B,D), times (B,N), duration (B,) -> scores (B,N)."""
            sim = (emb * q[:, None, :]).sum(-1)
            x = self.frame(torch.cat([emb, sim[..., None]], -1))
            gamma, beta = self.film(q).chunk(2, -1)
            x = x * (1 + gamma[:, None, :]) + beta[:, None, :] + self.time(self.time_features(times, duration[:, None]))
            r = self.out(self.encoder(x)).squeeze(-1)
            return sim / cfg.tau + r

    return TemporalHead()


def train_head(train_items: list[dict], val_items: list[dict], cfg: HeadAConfig, seed: int = 0, device: str = "cuda",
               log=print):
    """Train on items {emb, times, duration, q, intervals}; keep the epoch with the best val hit@4."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def tensors(it):
        return (torch.tensor(it["emb"], dtype=torch.float32, device=device)[None],
                torch.tensor(it["q"], dtype=torch.float32, device=device)[None],
                torch.tensor(it["times"], dtype=torch.float32, device=device)[None],
                torch.tensor([it["duration"]], dtype=torch.float32, device=device))

    targets = [torch.tensor(target_distribution(it["times"], it["intervals"], cfg.pad), device=device)
               for it in train_items]
    best, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for i in rng.permutation(len(train_items)):
            scores = model(*tensors(train_items[i]))[0]
            loss = -(targets[i] * torch.log_softmax(scores, -1)).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        val = evaluate(model, val_items, device) if val_items else {"hit@4": float("nan")}
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **val})
        if val_items and val["hit@4"] > best:
            best, best_state = val["hit@4"], {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"epoch {epoch:2d} loss {np.mean(losses):.3f} val hit@4 {val['hit@4']:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def score_items(model, items: list[dict], device: str = "cuda") -> list[np.ndarray]:
    import torch  # noqa: PLC0415

    model.eval()
    out = []
    with torch.no_grad():
        for it in items:
            s = model(torch.tensor(it["emb"], dtype=torch.float32, device=device)[None],
                      torch.tensor(it["q"], dtype=torch.float32, device=device)[None],
                      torch.tensor(it["times"], dtype=torch.float32, device=device)[None],
                      torch.tensor([it["duration"]], dtype=torch.float32, device=device))[0]
            out.append(s.float().cpu().numpy())
    return out


def evaluate(model, items: list[dict], device: str = "cuda", k: int = 4) -> dict:
    scores = score_items(model, items, device)
    hits = [hit_at_k(select_peaks(s, it["times"], k), it["intervals"]) for s, it in zip(scores, items, strict=True)]
    return {"hit@4": float(np.mean(hits))}
