"""Local LLM judge for open-ended answers: consistency + coverage against a reference.

One call per answer (FactQA-style precision/recall would need several):

    consistency  2 = no contradiction with the reference and no clear factual error
                 1 = minor inaccuracy or partly unsupported claim
                 0 = contradicts the reference or is clearly wrong / evasive
    coverage     share of the reference's key points the answer states: 0, 25, 50, 75 or 100

quality = coverage/100 x {0: 0, 1: 0.5, 2: 1}[consistency]   (in [0, 1])

The judge sees the question, the reference and the candidate, never the video. It is
validated against a human audit before its scores are trusted (docs/v3/research_plan.md,
gate G2). Verdicts are cached by (judge identity, prompt version, question, reference,
candidate text). Two cached local judges are supported: Phi-4-mini-instruct (different
model family from the answerer) and Qwen3-VL-4B-Instruct used text-only (same family as
the answerer, so self-preference must be checked in the audit).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

JUDGE_PROMPT_VERSION = "v3-judge-1"
JUDGES = {                                   # name -> (model id, pinned revision)
    "phi4mini": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083"),
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
}

PROMPT = """You grade a student-facing answer to a lecture question against a reference answer.

QUESTION: {question}

REFERENCE ANSWER: {reference}

CANDIDATE ANSWER: {candidate}

Grade two things independently.
consistency: 2 if the candidate does not contradict the reference and contains no clear factual error;
1 if it has a minor inaccuracy or a partly unsupported claim; 0 if it contradicts the reference, is clearly
wrong, or does not answer.
coverage: the share of the reference's key points that the candidate states (0, 25, 50, 75 or 100).
A candidate may use different words; judge meaning, not wording. Do not reward length.

Reply with JSON only, for example {{"consistency": 2, "coverage": 50}}."""


def quality(consistency: int, coverage: int) -> float:
    return coverage / 100.0 * {0: 0.0, 1: 0.5, 2: 1.0}[consistency]


def parse_verdict(text: str) -> dict | None:
    """Extract {"consistency", "coverage"} from the judge's reply; None if malformed."""
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        c, v = int(d["consistency"]), int(d["coverage"])
    except (ValueError, KeyError, TypeError):
        return None
    if c not in (0, 1, 2):
        return None
    v = min((0, 25, 50, 75, 100), key=lambda x: abs(x - v))       # snap to the declared scale
    return {"consistency": c, "coverage": v, "quality": quality(c, v)}


def verdict_key(judge: str, question: str, reference: str, candidate: str) -> str:
    payload = {"judge": list(JUDGES[judge]), "prompt": JUDGE_PROMPT_VERSION, "q": question, "ref": reference,
               "cand": candidate}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LocalJudge:
    """Greedy, deterministic local judge with a persistent verdict cache and optional budget."""

    def __init__(self, name: str, cache_path: str | Path, cache_dir: str = "cache/hf/hub", budget=None) -> None:
        if name not in JUDGES:
            raise ValueError(f"unknown judge {name!r}; choose from {sorted(JUDGES)}")
        self.name, self.cache_dir, self.budget = name, cache_dir, budget
        self.model_id, self.revision = JUDGES[name]
        self.path = Path(cache_path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec
        self._model = self._tok = None
        self.calls, self.hits, self.seconds = 0, 0, 0.0

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer  # noqa: PLC0415

        if self.name == "qwen3vl4b":
            self._tok = AutoProcessor.from_pretrained(self.model_id, revision=self.revision, cache_dir=self.cache_dir)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, revision=self.revision, dtype=torch.bfloat16, device_map="cuda",
                cache_dir=self.cache_dir).eval()
        else:
            self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, cache_dir=self.cache_dir)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, dtype=torch.bfloat16, device_map="cuda",
                cache_dir=self.cache_dir).eval()
        self._torch = torch

    def _generate(self, prompt: str) -> str:
        if self._model is None:
            self._load()
        torch = self._torch
        if self.name == "qwen3vl4b":
            msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            msgs = [{"role": "user", "content": prompt}]
        enc = self._tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=40, do_sample=False)
        tok = self._tok.tokenizer if hasattr(self._tok, "tokenizer") else self._tok
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def judge(self, qa_id: str, question: str, reference: str, candidate: str) -> dict:
        key = verdict_key(self.name, question, reference, candidate)
        if key in self.records:
            self.hits += 1
            return self.records[key]
        if self.budget is not None:
            self.budget.check_one()
        t0 = time.perf_counter()
        raw = self._generate(PROMPT.format(question=question, reference=reference, candidate=candidate))
        spent = time.perf_counter() - t0
        self.calls += 1
        self.seconds += spent
        if self.budget is not None:
            self.budget.charge(spent)
        v = parse_verdict(raw)
        rec = {"key": key, "qa_id": qa_id, "judge": self.name, "raw": raw[:200], "seconds": spent,
               **(v or {"consistency": None, "coverage": None, "quality": None, "malformed": True})}
        self.records[key] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def unload(self) -> None:
        """Free GPU memory so the answerer and a judge never sit on the GPU together."""
        if self._model is not None:
            del self._model
            self._model = None
            self._torch.cuda.empty_cache()
