"""Head B stage 2 data: the frozen answerer's own judgement of each pooled frame (CG-Bench train only).

For every training question and every candidate in its pool, the frozen Qwen3-VL-2B answers with that ONE
frame and no transcript (the frozen baseline's prompt). We keep the full softmax over the option letters, so
each frame gets a continuous utility: the probability of the gold option (and whether it is the argmax).
Head B then learns which frames make the answerer right, not only which lie inside the human interval.

Budget: own ledger, declared in research_log step 41 (5,000 calls / 3 GPU-hours). Cached by request key.

    python scripts/v3_head_b_labels.py                 # writes runs/v3_head_b/frame_utility.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN = Path("runs/v3_head_b")
LIMITS = (5000, 3 * 3600.0)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from v3_baseline_cycle import pool_dir, request_key  # noqa: PLC0415

    from videoqa.answerer import AnswerRequest, make_answerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.budget import CallBudget  # noqa: PLC0415
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.teacher import _as_frame, answerer_identity  # noqa: PLC0415

    cfg = load_config("configs/gpu_12gb.yaml")
    identity = answerer_identity(cfg)
    meta = json.loads((pool_dir("cgbench_train", "hybrid").parent / "pools.json").read_text())
    out_path = RUN / "frame_utility.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["key"] for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    budget = CallBudget(RUN / "labels_budget.json", *LIMITS)
    answerer = None
    RUN.mkdir(parents=True, exist_ok=True)
    for n, qid in enumerate(meta["qa_ids"]):
        p = load_pool(pool_dir("cgbench_train", "hybrid"), qid)
        for c in p.candidates:
            key = request_key(p, [c], [], identity)
            if key in done:
                continue
            if not budget.reserve(1):
                print("budget reached; stopping cleanly")
                return
            answerer = answerer or make_answerer(cfg)
            t0 = time.perf_counter()
            ans, _ = answerer.answer(AnswerRequest(p.question, p.options, [], [_as_frame(c)]))
            budget.charge(time.perf_counter() - t0)
            probs = ans.option_probs or []
            rec = {"key": key, "qa_id": qid, "cand": c.id, "option": ans.option_index,
                   "correct": ans.option_index == p.gold_option_index, "probs": probs,
                   "p_gold": probs[p.gold_option_index] if probs else None}
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            done.add(key)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(meta['qa_ids'])} questions; calls {budget.calls}", flush=True)


if __name__ == "__main__":
    main()
