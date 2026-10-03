"""Head A: transcript hot-moment scorer with two outputs.

For a question q and a transcript segment t_i (with one neighbour each side as
context), Head A predicts two separate quantities:

    out[0] = text utility   ~ R(q, T0 + {t_i}, S0) - R(q, T0, S0)   (add the LINE)
    out[1] = visual utility ~ R(q, T0, S0 + {f_i}) - R(q, T0, S0)   (add its FRAME)

The two targets come from SEPARATE interventions (labeling.head_a_labels); text
and frame are never added together, so neither gets credit for the other.

Backbone: cross-encoder/ms-marco-MiniLM-L6-v2 (22.7M, Apache-2.0), frozen. Before
training, both outputs fall back to its pretrained query-passage RELEVANCE logit
(zero-shot). Training fits a small linear head on frozen features.

Two feature modes:
  "full"   [relevance logit, 384-d CLS embedding]  (overfit in the pilot: 300 rows)
  "small"  six interpretable features: relevance logit, question-word overlap,
           best option-word overlap, spread of option overlaps (does the line
           separate the options?), contains a number, log length
"""

from __future__ import annotations

import re

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in is are was what which how does do and or for on at by with this that it as be".split())
SMALL_FEATURE_NAMES = ("relevance", "question_overlap", "option_overlap_max", "option_overlap_spread",
                       "has_number", "log_length")


def _toks(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}

BACKBONE = "cross-encoder/ms-marco-MiniLM-L6-v2"
BACKBONE_REV = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class HotMomentScorer:
    """score(question, segment_texts) -> (n, 2) array of [text utility, visual utility]."""

    def __init__(self, device: str = "cuda", cache_dir: str | None = None, feature_mode: str = "full") -> None:
        self.device, self.cache_dir, self.feature_mode = device, cache_dir, feature_mode
        self._tok = self._model = None
        self.head_w: np.ndarray | None = None    # (F, 2) linear head; None = zero-shot relevance
        self.head_b: np.ndarray | None = None
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None

    # -- frozen backbone ---------------------------------------------------
    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(BACKBONE, revision=BACKBONE_REV, cache_dir=self.cache_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            BACKBONE, revision=BACKBONE_REV, cache_dir=self.cache_dir).eval().to(self.device)

    def features(self, question: str, texts: list[str]) -> np.ndarray:
        """(n, 1 + H) frozen features: [relevance logit, CLS hidden state]."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self._model is None:
            self._load()
        torch = self._torch
        enc = self._tok([question] * len(texts), texts, padding=True, truncation=True, max_length=256,
                        return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        logit = out.logits[:, :1].float().cpu().numpy()                 # pretrained relevance
        cls = out.hidden_states[-1][:, 0].float().cpu().numpy()          # frozen sentence-pair embedding
        return np.concatenate([logit, cls], axis=1)

    def small_features(self, question: str, options: list[str] | None, texts: list[str]) -> np.ndarray:
        """(n, 6) interpretable features (see SMALL_FEATURE_NAMES)."""
        if not texts:
            return np.zeros((0, len(SMALL_FEATURE_NAMES)), dtype=np.float32)
        logit = self.features(question, texts)[:, 0]
        q, opts = _toks(question), [_toks(o) for o in (options or [])] or [set()]
        rows = []
        for lg, t in zip(logit, texts, strict=True):
            w = _toks(t)
            ov = [len(w & o) / (len(o) + 1) for o in opts]
            rows.append([lg, len(w & q) / (len(q) + 1), max(ov), max(ov) - min(ov),
                         float(bool(re.search(r"\d", t))), np.log1p(len(t.split()))])
        return np.asarray(rows, dtype=np.float32)

    def feats_for(self, question: str, texts: list[str], options: list[str] | None = None) -> np.ndarray:
        """Features in this scorer's mode."""
        if self.feature_mode == "small":
            return self.small_features(question, options, texts)
        return self.features(question, texts)

    # -- prediction ----------------------------------------------------------
    def score(self, question: str, texts: list[str], options: list[str] | None = None) -> np.ndarray:
        feats = self.feats_for(question, texts, options)
        if self.head_w is None:                                          # zero-shot: relevance for both
            return np.repeat(feats[:, :1], 2, axis=1)
        z = (feats - self.feat_mean) / self.feat_std
        return z @ self.head_w + self.head_b

    # -- training ------------------------------------------------------------
    def fit(self, feats: np.ndarray, targets: np.ndarray, groups: np.ndarray, epochs: int = 300,
            lr: float = 0.05, l2: float = 1e-3, rank_weight: float = 0.5, seed: int = 0) -> dict:
        """Fit the linear head on frozen features.

        targets: (n, 2) signed gains (text, visual). groups: question id per row;
        the ranking term only compares rows of the SAME question with DIFFERENT
        targets (ties are never forced into a preference).
        """
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        self.feat_mean = feats.mean(0)
        self.feat_std = feats.std(0) + 1e-6
        X = torch.tensor((feats - self.feat_mean) / self.feat_std, dtype=torch.float32)
        Y = torch.tensor(targets, dtype=torch.float32)
        W = torch.zeros(X.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr)
        pairs = _within_group_pairs(groups, targets)
        for _ in range(epochs):
            pred = X @ W + b
            loss = torch.nn.functional.huber_loss(pred, Y, delta=0.5) + l2 * (W ** 2).sum()
            for k in range(2):                                            # one ranking term per output
                i, j = pairs[k]
                if len(i):
                    s = torch.sign(Y[i, k] - Y[j, k])
                    loss = loss + rank_weight * torch.nn.functional.softplus(-s * (pred[i, k] - pred[j, k])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.head_w, self.head_b = W.detach().numpy(), b.detach().numpy()
        return {"final_loss": float(loss.detach()), "rank_pairs": [len(pairs[0][0]), len(pairs[1][0])]}


def _within_group_pairs(groups: np.ndarray, targets: np.ndarray):
    """For each output k: index pairs (i, j) in the same group with targets[i,k] != targets[j,k]."""
    out = []
    for k in range(targets.shape[1]):
        I, J = [], []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            for a in range(len(idx)):
                for c in range(a + 1, len(idx)):
                    if targets[idx[a], k] != targets[idx[c], k]:
                        I.append(idx[a])
                        J.append(idx[c])
        out.append((np.array(I, dtype=int), np.array(J, dtype=int)))
    return out
