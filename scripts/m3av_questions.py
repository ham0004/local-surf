"""M3AV lecture questions (docs/v3/m3av_protocol.md, steps 1-4 F1-F3): generation from slide image + OCR only.

Nemotron sees one saved slide image and its OCR text (never the speech or the times) and writes one 4-option
question plus the exact slide words that support the answer. Slides are taken per lecture in the protocol's hash
order until the quota of questions passing F1 (format) and F2 (grounding) is met, at most 3 x quota slides per
lecture. F3 then marks every segment of the lecture that shows the same words as valid evidence. F4 (visibility)
needs the videos and is done by `scripts/m3av_visibility.py`.

Every reply is cached (runs/m3av/gen_cache.jsonl) and charged to runs/m3av/budget_gen.json.

    python scripts/m3av_questions.py --split dev          # data/m3av/questions_dev.jsonl
    python scripts/m3av_questions.py --split train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.answerer_api import APIVLMAnswerer, _jpeg  # noqa: E402
from videoqa.v2.budget import CallBudget  # noqa: E402

RAW = Path("data/m3av_raw/dataset_v1.0_noaudio")
SPLIT = Path("data/m3av_raw/set_split.json")
RUN, OUT = Path("runs/m3av"), Path("data/m3av")
MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
QUOTA = {"train": 4, "dev": 6, "test": 10}
N_TRAIN_LECTURES = 300
UNAVAILABLE = {"CHI-36F5D"}                         # dev video needs a YouTube sign-in (research log step 71)
GEN_LIMITS = (6000, 120 * 3600.0)        # seconds are summed over parallel workers, 429 waits included
PROMPT = """You write one multiple-choice question for a lecture-video benchmark from the lecture slide shown.

Rules:
- The question must be answerable from what is visibly written or drawn on THIS slide.
- Do not ask about the talk title, authors, affiliations, logos, dates or slide numbers.
- It must not be answerable from general knowledge alone; ask about a specific fact, number, label, item or
  relation shown on the slide.
- Write exactly 4 options; exactly one is correct; the wrong options must be plausible and of the same kind.
- evidence_text: copy the exact words from the slide that support the correct answer (a short span, 3-25 words).

OCR text of the slide (may contain recognition errors):
{ocr}

Reply with only a JSON object:
{{"question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "evidence_text": "..."}}"""


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def grounded(evidence: str, ocr: str, share: float = 0.8) -> bool:
    """F2 / F3: at least 80% of the evidence words occur in the OCR text."""
    ev, have = words(evidence), set(words(ocr))
    return bool(ev) and sum(w in have for w in ev) / len(ev) >= share


def order_key(lecture: str, start: float) -> str:
    return hashlib.sha256(f"m3av-protocol:{lecture}:{start}".encode()).hexdigest()


def lectures(split: str) -> list[str]:
    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in s.items() if v == split and k not in UNAVAILABLE)
    if split == "train":
        ids = sorted(ids, key=lambda x: hashlib.sha256(f"m3av-protocol:train:{x}".encode()).hexdigest())[:N_TRAIN_LECTURES]
    return ids


def segments(lecture: str) -> list[dict]:
    return json.loads((RAW / lecture / "ocr" / "seg_speech_ocr.json").read_text(encoding="utf-8"))


def eligible(segs: list[dict], lecture: str) -> list[int]:
    """Protocol step 1: at least 4 s, at least 8 distinct OCR words, not the first segment; in hash order."""
    ok = [i for i, s in enumerate(segs) if i > 0 and s["end"] - s["start"] >= 4.0 and len(set(words(s["ocr_text"]))) >= 8]
    return sorted(ok, key=lambda i: order_key(lecture, segs[i]["start"]))


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        q = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return q if isinstance(q, dict) else None


def f1(q: dict | None) -> bool:
    """Format: a question, 4 distinct non-empty options, key in range, evidence text present."""
    if not q or not isinstance(q.get("question"), str) or not isinstance(q.get("evidence_text"), str):
        return False
    opts = q.get("options")
    if not isinstance(opts, list) or len(opts) != 4 or not all(isinstance(o, str) and o.strip() for o in opts):
        return False
    return len({o.strip().lower() for o in opts}) == 4 and q.get("answer_index") in (0, 1, 2, 3)


class Generator:
    def __init__(self) -> None:
        RUN.mkdir(parents=True, exist_ok=True)
        self.api = APIVLMAnswerer(MODEL, max_tokens=4096)
        self.budget = CallBudget(RUN / "budget_gen.json", *GEN_LIMITS)
        self.cache_path = RUN / "gen_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r
        self.lock = threading.Lock()

    def ask(self, lecture: str, seg: dict) -> dict:
        key = f"{lecture}:{seg['start']}"
        if key in self.cache:
            return self.cache[key]
        from PIL import Image  # noqa: PLC0415

        with self.lock:
            if not self.budget.reserve(1):
                raise RuntimeError("generation ledger exhausted")
        img = Image.open(RAW / lecture / "ocr" / "seg_imgs" / seg["name"])
        content = [{"type": "image_url", "image_url": {"url": _jpeg(img, 1280)}},
                   {"type": "text", "text": PROMPT.format(ocr=seg["ocr_text"][:3000])}]
        t0 = time.perf_counter()
        out = self.api._post({"model": MODEL, "messages": [{"role": "user", "content": content}],
                              "temperature": 0, "max_tokens": self.api.max_tokens})
        text = (out["choices"][0]["message"].get("content") or "").strip()
        rec = {"key": key, "text": text, "seconds": time.perf_counter() - t0}
        with self.lock:
            self.budget.charge(rec["seconds"])
            self.cache[key] = rec
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def lecture(self, lecture: str, quota: int) -> tuple[list[dict], dict]:
        segs = segments(lecture)
        out, tried, fails = [], 0, {"F1": 0, "F2": 0}
        for i in eligible(segs, lecture):
            if len(out) >= quota or tried >= 3 * quota:
                break
            tried += 1
            q = parse(self.ask(lecture, segs[i])["text"])
            if not f1(q):
                fails["F1"] += 1
                continue
            if not grounded(q["evidence_text"], segs[i]["ocr_text"]):
                fails["F2"] += 1
                continue
            valid = [j for j, s in enumerate(segs) if j == i or grounded(q["evidence_text"], s["ocr_text"])]   # F3
            qa_id = f"m3av:{lecture}:{segs[i]['start']:g}"
            perm = sorted(range(4), key=lambda k: hashlib.sha256(f"m3av-options:{qa_id}:{k}".encode()).hexdigest())
            out.append({"qa_id": qa_id, "lecture": lecture,      # options in a fixed hash order (no position bias)
                        "question": q["question"].strip(), "options": [q["options"][k].strip() for k in perm],
                        "answer_index": perm.index(q["answer_index"]), "generated_answer_index": q["answer_index"], "evidence_text": q["evidence_text"].strip(),
                        "source_segment": [segs[i]["start"], segs[i]["end"]], "source_image": segs[i]["name"],
                        "valid_segments": [[segs[j]["start"], segs[j]["end"]] for j in valid]})
        return out, {"lecture": lecture, "eligible": len(eligible(segs, lecture)), "tried": tried, **fails, "kept": len(out)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("train", "dev", "test"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="first N lectures only (trial; output not written)")
    a = p.parse_args()
    gen = Generator()
    ids = lectures(a.split)[:a.limit or None]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda x: gen.lecture(x, QUOTA[a.split]), ids))
    qs = [q for r, _ in results for q in r]
    if a.limit:
        print(json.dumps(qs[:3], indent=1, ensure_ascii=False), [s for _, s in results])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    qs = [q for r, _ in results for q in r]
    (OUT / f"questions_{a.split}.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs),
                                                     encoding="utf-8")
    (OUT / f"attrition_{a.split}.json").write_text(json.dumps([s for _, s in results], indent=1), encoding="utf-8")
    tot = {k: sum(s[k] for _, s in results) for k in ("eligible", "tried", "F1", "F2", "kept")}
    print(f"{a.split}: {len(ids)} lectures, {tot}; ledger calls {gen.budget.calls}")


if __name__ == "__main__":
    main()
