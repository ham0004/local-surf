"""Measure the per-action cost model on this device (warm, repeated).

Measures, on real videos:
  * decode_ms   : decoding ONE frame at a sparse time (keyframe seek + decode),
                  the marginal cost of looking at one new moment
  * scout_ms    : scouting ONE frame (the frame + its fixed 1 s-earlier
                  neighbour for visual change), single-image batch
  * answer_ms   : final answer-model latency, fitted as
                  a + b * visual_tokens + c * text_tokens  (least squares)
                  from calls with 0, 1, 2 and 4 frames
Each measurement is repeated; mean and sd are reported, and run order is
rotated so warm-up effects do not favour one setting.

Writes a cost-model YAML that labels, training, threshold tuning and inference
all read (one definition everywhere), plus the raw timings.

Usage:
  uv run python scripts/profile_costs.py --data data/longvideobench_full --n-videos 8 \
      --out configs/cost_model_rtx5060ti.yaml --raw reports/costs/profile_rtx5060ti.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.answerer import AnswerRequest  # noqa: E402
from videoqa.answerer_hf import HFVLMAnswerer  # noqa: E402
from videoqa.datasets import load_local_dataset  # noqa: E402
from videoqa.frames import decode_at, probe  # noqa: E402
from videoqa.scout import OpenClipScout  # noqa: E402


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-videos", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", required=True)
    a = ap.parse_args()

    items, _ = load_local_dataset(a.data, splits=("train",))
    seen, picks = set(), []
    for it in items:
        if it.qa.video_id not in seen:
            seen.add(it.qa.video_id)
            picks.append(it)
        if len(picks) >= a.n_videos:
            break

    answerer = HFVLMAnswerer("Qwen/Qwen3-VL-2B-Instruct", revision="89644892e4d85e24eaac8bacfd4f463576704203",
                             cache_dir="cache/hf/hub")
    scout = OpenClipScout(device="cuda", cache_dir="cache/open_clip")
    answerer.ensure_loaded()
    scout.ensure_loaded()

    decode, scout_ms, answer_rows = [], [], []
    for rep in range(a.repeats):
        for it in (picks if rep % 2 == 0 else list(reversed(picks))):      # rotate order
            dur = probe(it.video_path).duration_s
            times = [dur * f for f in (0.2, 0.5, 0.8)]
            for t in times:
                res, ms = timed(lambda t=t: decode_at(it.video_path, [t], max_side=640))
                decode.append({"video": it.qa.video_id, "t": t, "ms": ms, "frames_visited": res.frames_visited})
                pair, ms2 = timed(lambda t=t: decode_at(it.video_path, [max(0.0, t - 1.0), t], max_side=640))
                _, ms3 = timed(lambda: scout.score(it.qa.question, pair.frames[1:]))
                scout_ms.append({"video": it.qa.video_id, "ms": ms3, "neighbour_decode_ms": ms2 - ms})
            frames = decode_at(it.video_path, times + [dur * 0.35], max_side=640).frames
            for k in (0, 1, 2, 4):
                req = AnswerRequest(it.qa.question, it.qa.options, it.transcript.segments[:12], frames[:k])
                pre_ms, vis, txt = answerer.prefill_ms(req)
                (ans, usage), ms = timed(lambda req=req: answerer.answer(req))
                answer_rows.append({"video": it.qa.video_id, "frames": k, "visual_tokens": vis,
                                    "prompt_text_tokens": txt, "generated_tokens": usage.generated_tokens,
                                    "prefill_ms": pre_ms, "total_ms": ms})

    # Prefill (grows with frames) and generation (grows with output length) are
    # fitted separately: total latency alone confounds the two (a first fit on
    # totals gave a NEGATIVE per-visual-token cost with a 645 ms residual).
    X = np.array([[1.0, r["visual_tokens"], r["prompt_text_tokens"]] for r in answer_rows])
    y = np.array([r["prefill_ms"] for r in answer_rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    G = np.array([[1.0, r["generated_tokens"]] for r in answer_rows])
    gy = np.array([r["total_ms"] - r["prefill_ms"] for r in answer_rows])
    gcoef, *_ = np.linalg.lstsq(G, gy, rcond=None)
    d = [r["ms"] for r in decode]
    sc = [r["ms"] for r in scout_ms]
    nb = [r["neighbour_decode_ms"] for r in scout_ms]
    model = {
        "device": "NVIDIA GeForce RTX 5060 Ti (16 GB), bf16",
        "measured": time.strftime("%Y-%m-%d"),
        "n_videos": len(picks), "repeats": a.repeats,
        "decode_ms_per_frame": round(statistics.mean(d), 1),
        "decode_ms_per_frame_sd": round(statistics.stdev(d), 1),
        "scout_ms_per_frame": round(statistics.mean(sc), 1),
        "scout_ms_per_frame_sd": round(statistics.stdev(sc), 1),
        "scout_neighbour_decode_ms": round(statistics.mean(nb), 1),
        "prefill_ms_intercept": round(float(coef[0]), 1),
        "prefill_ms_per_visual_token": round(float(coef[1]), 4),
        "prefill_ms_per_text_token": round(float(coef[2]), 4),
        "prefill_fit_residual_sd_ms": round(float(resid.std()), 1),
        "generation_ms_intercept": round(float(gcoef[0]), 1),
        "generation_ms_per_token": round(float(gcoef[1]), 2),
        "controller_ms_per_step": 1.0,
        "note": "decode/scout are single-frame marginal costs; answer cost is a linear fit, not a guarantee",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump({"cost_model": model}, sort_keys=False), encoding="utf-8")
    Path(a.raw).parent.mkdir(parents=True, exist_ok=True)
    Path(a.raw).write_text(json.dumps({"model": model, "decode": decode, "scout": scout_ms,
                                       "answer": answer_rows}, indent=1), encoding="utf-8")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
