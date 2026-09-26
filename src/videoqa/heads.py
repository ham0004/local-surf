"""Small learned utility head with the paired transcript-intervention loss.

Model
-----
u_hat(x) = w2 . tanh(W1 x + b1) + b2        one hidden layer, NumPy only

x is the action feature vector from features.py; u_hat predicts the answer-
quality GAIN of taking that action (measured by label generation).

Loss (per mini-batch)
---------------------
L = mean_i Huber(u_hat_i - gain_i)                                  [absolute utility]
  + lambda_pair * mean_(t,c) Huber((u_hat_t - u_hat_c) - (gain_t - gain_c))
                                                                     [paired term]

where (t, c) are rows for the SAME question and SAME candidate time, one under
TARGETED_DAMAGE and one under CONTROL_DAMAGE.  The paired term asks the head
to get the *difference* right: "this frame is worth more because answer-
relevant speech vanished, not because any speech vanished".

With perfect absolute labels the paired term adds no new information; the
hypothesis is that it helps with finite, noisy data.  Setting lambda_pair = 0
gives the "no pairing" ablation with identical rows and optimiser.

Kept in NumPy (manual gradients + Adam) so the CPU smoke path needs no torch,
and so the whole head is ~100 lines anyone can audit.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


def _huber_grad(r: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Huber loss value and d/dr, elementwise."""
    a = np.abs(r)
    loss = np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))
    grad = np.where(a <= delta, r, delta * np.sign(r))
    return loss, grad


@dataclasses.dataclass
class TrainConfig:
    hidden: int = 32
    lr: float = 3e-3
    epochs: int = 300
    lambda_pair: float = 1.0
    huber_delta: float = 0.5
    weight_decay: float = 1e-4
    seed: int = 0


class UtilityHead:
    """One-hidden-layer MLP predicting action utility."""

    def __init__(self, n_features: int, hidden: int = 32, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_features), (hidden, n_features))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden)
        self.b2 = np.zeros(1)
        # Feature standardisation, fit on the TRAIN rows only.
        self.mu = np.zeros(n_features)
        self.sigma = np.ones(n_features)

    # --- forward ---------------------------------------------------------
    def _norm(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = np.tanh(self._norm(np.atleast_2d(X)) @ self.W1.T + self.b1)
        return H @ self.w2 + self.b2[0]

    # --- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, pairs: np.ndarray, cfg: TrainConfig) -> list[float]:
        """Full-batch training.  ``pairs`` is an (P, 2) int array of row indices
        (targeted_row, control_row).  Returns the loss curve."""
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-6
        self.sigma[self.sigma < 1e-5] = 1.0   # constant features (e.g. bias) stay as-is
        Xn = self._norm(X)
        params = [self.W1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        curve = []

        for step in range(1, cfg.epochs + 1):
            # forward
            Z = Xn @ self.W1.T + self.b1
            H = np.tanh(Z)
            u = H @ self.w2 + self.b2[0]

            # absolute utility term
            loss_abs, g_abs = _huber_grad(u - y, cfg.huber_delta)
            du = g_abs / len(y)
            loss = loss_abs.mean()

            # paired term: d/du_t = +g, d/du_c = -g
            if cfg.lambda_pair > 0 and len(pairs):
                t, c = pairs[:, 0], pairs[:, 1]
                r = (u[t] - u[c]) - (y[t] - y[c])
                loss_pair, g_pair = _huber_grad(r, cfg.huber_delta)
                loss += cfg.lambda_pair * loss_pair.mean()
                scale = cfg.lambda_pair / len(pairs)
                np.add.at(du, t, scale * g_pair)
                np.add.at(du, c, -scale * g_pair)

            # backward through the MLP
            g_w2 = H.T @ du + cfg.weight_decay * self.w2
            g_b2 = np.array([du.sum()])
            dZ = np.outer(du, self.w2) * (1 - H**2)
            g_W1 = dZ.T @ Xn + cfg.weight_decay * self.W1
            g_b1 = dZ.sum(axis=0)

            # Adam update (in place so self.* stay bound to the same arrays)
            for p, g, mi, vi in zip(params, [g_W1, g_b1, g_w2, g_b2], m, v, strict=True):
                mi *= beta1
                mi += (1 - beta1) * g
                vi *= beta2
                vi += (1 - beta2) * g * g
                m_hat = mi / (1 - beta1**step)
                v_hat = vi / (1 - beta2**step)
                p -= cfg.lr * m_hat / (np.sqrt(v_hat) + eps)
            curve.append(float(loss))
        return curve

    # --- persistence -------------------------------------------------------
    def save(self, path: str | Path, **meta: float) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, w2=self.w2, b2=self.b2, mu=self.mu, sigma=self.sigma,
                 **{f"meta_{k}": np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path: str | Path) -> tuple[UtilityHead, dict[str, float]]:
        data = np.load(path)
        head = cls(n_features=data["W1"].shape[1], hidden=data["W1"].shape[0])
        for k in ("W1", "b1", "w2", "b2", "mu", "sigma"):
            setattr(head, k, data[k])
        meta = {k[5:]: float(data[k]) for k in data.files if k.startswith("meta_")}
        return head, meta
