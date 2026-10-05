"""Reference-based text metrics reported by EduVidQA (EMNLP 2025, Section 5.2.1), no LLM judge.

    bleu1       unigram BLEU with brevity penalty (whitespace/punctuation tokens, lower-cased)
    rouge_l     ROUGE-L F1 (longest common subsequence)
    meteor      NLTK METEOR (WordNet data in cache/nltk_data)
    entailment  p(entailment | premise = reference, hypothesis = generated) from an NLI
                cross-encoder; ERVQA (Ray et al., 2024) describes "roberta-base-nli trained on
                SNLI and MultiNLI"; the public checkpoint used here is cross-encoder/nli-roberta-base
                (pinned revision), whole answers, truncated to 512 tokens.

These measure overlap or entailment with the reference; FactQA (factqa.py) is the main
accuracy/coverage measure.
"""

from __future__ import annotations

import math
import re
from collections import Counter

NLI_MODEL = ("cross-encoder/nli-roberta-base", "1be0567456f0543475805e758725f151f283705a")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower())


def bleu1(reference: str, generated: str) -> float:
    ref, gen = tokens(reference), tokens(generated)
    if not gen:
        return 0.0
    overlap = sum((Counter(gen) & Counter(ref)).values())
    precision = overlap / len(gen)
    bp = 1.0 if len(gen) > len(ref) else math.exp(1 - len(ref) / len(gen))
    return bp * precision


def rouge_l(reference: str, generated: str) -> float:
    ref, gen = tokens(reference), tokens(generated)
    if not ref or not gen:
        return 0.0
    prev = [0] * (len(gen) + 1)
    for r in ref:                                     # LCS by dynamic programming, O(len(ref) * len(gen))
        cur = [0]
        for j, g in enumerate(gen, 1):
            cur.append(prev[j - 1] + 1 if r == g else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(gen), lcs / len(ref)
    return 2 * p * rc / (p + rc)


def meteor(reference: str, generated: str) -> float:
    import nltk  # noqa: PLC0415 - research dependency
    from nltk.translate.meteor_score import meteor_score  # noqa: PLC0415

    if "cache/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "cache/nltk_data")
    return float(meteor_score([tokens(reference)], tokens(generated)))


class Entailment:
    """Batched NLI entailment probability; the model loads lazily (CPU is fine for a few hundred pairs)."""

    def __init__(self, device: str = "cpu", cache_dir: str = "cache/hf/hub") -> None:
        self.device, self.cache_dir = device, cache_dir
        self._model = self._tok = None

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        name, rev = NLI_MODEL
        self._tok = AutoTokenizer.from_pretrained(name, revision=rev, cache_dir=self.cache_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev,
                                                                        cache_dir=self.cache_dir).eval().to(self.device)
        labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
        self._entail = labels["entailment"]
        self._torch = torch

    def __call__(self, pairs: list[tuple[str, str]], batch: int = 16) -> list[float]:
        """pairs: (reference, generated). Returns p(entailment) per pair."""
        if self._model is None:
            self._load()
        torch, out = self._torch, []
        for i in range(0, len(pairs), batch):
            ref, gen = zip(*pairs[i:i + batch], strict=True)
            enc = self._tok(list(ref), list(gen), truncation=True, max_length=512, padding=True,
                            return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1)
            out.extend(probs[:, self._entail].tolist())
        return out
