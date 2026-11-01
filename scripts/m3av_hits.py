"""Primary measure of the lecture study for every system, without the answer model (docs/v3/m3av_protocol.md).

Source-slide interval hit@4: 1 when one of the 4 frames a system would show lies inside a visible span of a valid
evidence segment (1 s tolerance; the same test as `chosen_hits_evidence` in v3_baseline_cycle.py). Frames come
from the saved pools with the same `select` the answer stage uses. Paired bootstrap over lectures (10,000
resamples, seed 20261101); questions and lectures reported. Also: frames actually shown per system.

    python scripts/m3av_hits.py --split dev                     # reports/m3av/hits_dev.{json,md}
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_baseline_cycle import items, pool_dir, select  # noqa: E402

B, SEED = 10_000, 20261101
SYSTEMS = {  # name: (pool, selector); a missing pool or selection file is reported and skipped
    "S0 starting design": ("v2", "clip"),
    "S1 best framework (also S1t's frames)": ("hybrid", "mmropt"),
    "S2c Head A (CG-Bench)": ("hybrid_ahead", "mmropt"),
    "S2m Head A (M3AV train)": ("hybrid_ahead_m3av", "mmropt"),
    "S3 Head B": ("hybrid", "file:{ds}_hybrid_ev_set"),
    "R2 4 evenly spaced": ("uniform4", "clip"),
    "R3 dense MobileCLIP top-4": ("dense4", "clip"),
    "R4 dense option-aware top-4": ("dense4o", "clip"),
    "R5 VSI-style": ("vsi4", "clip"),
}
CONTRASTS = [("S2m Head A (M3AV train)", "R3 dense MobileCLIP top-4"),
             ("S2m Head A (M3AV train)", "S1 best framework (also S1t's frames)")]


def paired(a: dict, b: dict, lecture: dict) -> dict:
    qs = sorted(set(a) & set(b))
    byl = collections.defaultdict(list)
    for q in qs:
        byl[lecture[q]].append(float(a[q]) - float(b[q]))
    ls = sorted(byl)
    rng = np.random.default_rng(SEED)
    boots = np.array([np.mean(np.concatenate([byl[ls[i]] for i in rng.choice(len(ls), len(ls))])) for _ in range(B)])
    p = min(1.0, 2 * (min(np.sum(boots <= 0), np.sum(boots >= 0)) + 1) / (B + 1))
    return {"diff": round(100 * float(np.mean([a[q] - b[q] for q in qs])), 1),
            "lo": round(100 * float(np.percentile(boots, 2.5)), 1), "hi": round(100 * float(np.percentile(boots, 97.5)), 1),
            "p": round(float(p), 4), "n": len(qs), "lectures": len(ls)}


def main() -> None:
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    a = p.parse_args()
    ds = f"m3av_{a.split}"
    its = {it.qa.qa_id: it.qa for it in items(ds)}
    lecture = {q: v.video_id for q, v in its.items()}
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    qids = sorted(its)
    qo = dict(zip(qids, enc.embed_texts([its[q].question + " Options: " + ", ".join(its[q].options) for q in qids]),
                  strict=True))
    hits, shown, missing = {}, {}, []
    for name, (pool, rule) in SYSTEMS.items():
        d, rule = pool_dir(ds, pool), rule.format(ds=ds)
        if not d.exists() or (rule.startswith("file:") and not Path(f"runs/v3_head_b/selections/{rule[5:]}.json").exists()):
            missing.append(name)
            continue
        h, n = {}, []
        for q in qids:
            frames = select(load_pool(d, q), rule, 4, qo[q])
            h[q] = any(x - 1.0 <= c.time_s <= y + 1.0 for c in frames for x, y in its[q].evidence_intervals_s)
            n.append(len(frames))
        hits[name] = h
        shown[name] = {"mean": round(float(np.mean(n)), 2), "under_4": int(sum(x < 4 for x in n))}
    out = {"split": a.split, "questions": len(qids), "lectures": len(set(lecture.values())), "missing": missing,
           "systems": {k: {"hit@4": round(100 * float(np.mean(list(v.values()))), 1), **shown[k]} for k, v in hits.items()},
           "contrasts": {f"{x} - {y}": paired(hits[x], hits[y], lecture) for x, y in CONTRASTS if x in hits and y in hits},
           "vs_R3": {k: paired(v, hits["R3 dense MobileCLIP top-4"], lecture) for k, v in hits.items()
                     if "R3 dense MobileCLIP top-4" in hits and not k.startswith("R3")}}
    Path("reports/m3av").mkdir(parents=True, exist_ok=True)
    Path(f"reports/m3av/hits_{a.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [f"# Source-slide interval hit@4, M3AV {a.split} ({len(qids)} questions, {out['lectures']} lectures)", "",
             "| System | hit@4 (%) | frames shown (mean) | questions with < 4 frames | − R3 (95% interval over lectures) |",
             "|---|---|---|---|---|"]
    for k, v in out["systems"].items():
        c = out["vs_R3"].get(k)
        lines.append(f"| {k} | {v['hit@4']} | {v['mean']} | {v['under_4']} | "
                     + (f"{c['diff']:+.1f} ({c['lo']:+.1f}..{c['hi']:+.1f})" if c else "") + " |")
    if missing:
        lines += ["", "Not yet available: " + ", ".join(missing)]
    Path(f"reports/m3av/hits_{a.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
