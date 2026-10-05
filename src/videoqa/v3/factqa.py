"""FactQA precision/recall (EduVidQA, EMNLP 2025, Appendix E.2.1; from SyllabusQA) with an API judge.

The judge lists the atomic claims of Answer 1 and counts how many are supported by Answer 2:

    precision = claims of the GENERATED answer supported by the reference   (accuracy)
    recall    = claims of the REFERENCE supported by the generated answer   (coverage)

Two judge calls per answer. The prompt text is the paper's, verbatim. The judge is the
strongest model on Google AI Studio's free tier (Gemini Flash at the time of writing);
its exact model version string is recorded with every verdict. The API key is read from the
GEMINI_API_KEY environment variable or ~/.gemini_api_key and is never written anywhere. Note: Google may use
free-tier request content to improve its products; only public dataset text and model
answers are sent. Verdicts are cached by (model, prompt version, direction, question,
answer 1, answer 2) and every real call is charged to a CallBudget.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

FACTQA_PROMPT_VERSION = "eduvidqa-E.2.1"
# Paid-tier prices, USD per token (ai.google.dev/gemini-api/docs/pricing, checked 2026-10-05; valid to 2026-12-31).
# Output includes thinking tokens.
PRICES = {"gemini-3.8-flash": {"input": 0.75e-6, "output": 3.75e-6}}


class CostCapReached(RuntimeError):
    """The next call could push measured spend past the declared dollar cap."""
FACTQA_PROMPT = """Your job is to evaluate the similarity of different answers to a single question. You will be given a question from a specific computer science college course. You will also be given two possible answers to that question, and will have to evaluate the claims in one answer against the other.
Steps:
1. List all of the atomic claims made by Answer 1. Note that an answer saying that there is no information counts as a single claim.
2. Tell me which of those claims are supported by Answer 2.
3. Summarize the results using the template: Score: <num supported claims>/<num total claims> Ensure that both numbers are integers.
Question: {question}
Answer 1: {answer_1}
Answer 2: {answer_2}"""


def api_key() -> str:
    """GEMINI_API_KEY from the environment, else ~/.gemini_api_key (outside the repository)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    home = Path.home() / ".gemini_api_key"
    if not key and home.exists():
        key = home.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("no Gemini key: set GEMINI_API_KEY or put the key in ~/.gemini_api_key")
    return key


def parse_score(text: str) -> tuple[int, int] | None:
    """Last 'Score: a/b' in the judge's reply, with 0 <= a <= b and b > 0."""
    hits = re.findall(r"Score:\s*\**\s*(\d+)\s*/\s*(\d+)", text)
    if not hits:
        return None
    a, b = map(int, hits[-1])
    return (a, b) if b > 0 and 0 <= a <= b else None


def verdict_key(model: str, direction: str, question: str, answer_1: str, answer_2: str) -> str:
    payload = {"model": model, "prompt": FACTQA_PROMPT_VERSION, "dir": direction, "q": question,
               "a1": answer_1, "a2": answer_2}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class GeminiFactQA:
    """FactQA with the Gemini API (REST, no SDK). Deterministic decoding (temperature 0)."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, cache_path: str | Path, budget=None, min_interval_s: float = 4.5,
                 max_usd: float | None = None, worst_case_usd_per_call: float = 0.05) -> None:
        self.model, self.budget, self.min_interval_s = model, budget, min_interval_s
        self.max_usd, self.worst_case = max_usd, worst_case_usd_per_call
        self.path = Path(cache_path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec
        self._last = 0.0
        self.calls, self.hits = 0, 0
        self.spent_usd = sum(r.get("usd", 0.0) or 0.0 for r in self.records.values())

    def _post(self, prompt: str) -> tuple[str, str]:
        key = api_key()
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"temperature": 0}}).encode()
        for attempt in range(6):
            wait = self.min_interval_s - (time.monotonic() - self._last)      # stay under the free-tier RPM
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(self.URL.format(model=self.model), data=body, method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out = json.loads(r.read())
                text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
                self._usage = out.get("usageMetadata", {})
                return text, out.get("modelVersion", self.model)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503) and attempt < 5:
                    time.sleep(min(60, 5 * 2 ** attempt))                     # rate limit / transient: back off
                    continue
                raise RuntimeError(f"Gemini API error {e.code}: {e.read()[:300]!r}") from e
        raise RuntimeError("Gemini API kept failing")

    def claims(self, qa_id: str, direction: str, question: str, answer_1: str, answer_2: str) -> dict:
        k = verdict_key(self.model, direction, question, answer_1, answer_2)
        if k in self.records:
            self.hits += 1
            return self.records[k]
        if self.budget is not None:
            self.budget.check_one()
        if self.max_usd is not None and self.spent_usd + self.worst_case > self.max_usd:
            raise CostCapReached(f"spent ${self.spent_usd:.3f} of the ${self.max_usd:.2f} cap")
        t0 = time.perf_counter()
        text, version = self._post(FACTQA_PROMPT.format(question=question, answer_1=answer_1, answer_2=answer_2))
        spent = time.perf_counter() - t0
        self.calls += 1
        if self.budget is not None:
            self.budget.charge(spent)
        s = parse_score(text)
        u = getattr(self, "_usage", {}) or {}
        tin = int(u.get("promptTokenCount", 0))
        tout = int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))
        price = PRICES.get(self.model)
        usd = tin * price["input"] + tout * price["output"] if price else None
        self.spent_usd += usd or 0.0
        rec = {"key": k, "qa_id": qa_id, "direction": direction, "model_version": version,
               "input_tokens": tin, "output_tokens": tout, "thought_tokens": int(u.get("thoughtsTokenCount", 0)),
               "usd": usd,
               "supported": s[0] if s else None, "total": s[1] if s else None,
               "score": s[0] / s[1] if s else None, "raw_tail": text[-400:]}
        self.records[k] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def score(self, qa_id: str, question: str, reference: str, generated: str) -> dict:
        """FactQA precision (generated vs reference) and recall (reference vs generated)."""
        p = self.claims(qa_id, "precision", question, generated, reference)
        r = self.claims(qa_id, "recall", question, reference, generated)
        return {"precision": p["score"], "recall": r["score"], "model_version": p["model_version"]}
