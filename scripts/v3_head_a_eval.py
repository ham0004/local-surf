"""Head A cycle: one harness, one metric, every Path A method (zero-shot rules now, learned methods later).

Head A (docs/v3/heads_design.md) turns the question + timed speech into moments where the visual evidence
should be. Every method returns ranked times for a question; the metric is the same for all:

    recall@k   share of questions with at least one of the first k proposals inside a human clue
               interval (+/- 1 s, as in the baseline cycle); k = 1, 2, 4, 6 (v2 uses 6 Path A frames)
    dist@6     median distance (s) from the nearest of the 6 proposals to the evidence

Data: data/head_a/{train,dev}.jsonl from scripts/v3_head_a_data.py. Dev = CG-Bench dev-split videos with
English subtitles (164 questions, human intervals). Test-split videos are never read.

Zero-shot methods (no training):
    uniform      6 evenly spaced times (chance reference)
    bm25_window  centres of the v2 BM25 windows (retrieval only)
    bm25_line    BM25 over single lines of the whole transcript, frame at line end - 0.3 s
    v2           the v2 Path A rule exactly: lines inside the BM25 windows, MiniLM relevance of the line with
                 one neighbour each side, top lines, frame at line end - 0.3 s
    minilm_all   the same MiniLM relevance over every line of the video (no window restriction)
Each method also has a "+nms" variant: proposals at least 8 s apart, so k proposals are k distinct moments.

    python scripts/v3_head_a_eval.py zeroshot            # writes reports/v3_head_a/zeroshot.json
    python scripts/v3_head_a_eval.py offsets             # speech-to-evidence offsets on train (analysis)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path("data/head_a")
OUT = Path("reports/v3_head_a")
KS = (1, 2, 4, 6)
TOL_S = 1.0
NMS_GAP_S = 8.0


def load(split: str) -> list[dict]:
    return [json.loads(x) for x in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def segments(rec):
    from videoqa.schemas import TranscriptSegment  # noqa: PLC0415

    return [TranscriptSegment(id=f"s{i:05d}", start_s=a, end_s=b, text=t) for i, (a, b, t) in enumerate(rec["lines"])]


def inside(t: float, evidence) -> bool:
    return any(a - TOL_S <= t <= b + TOL_S for a, b in evidence)


def distance(t: float, evidence) -> float:
    return min(0.0 if a <= t <= b else min(abs(t - a), abs(t - b)) for a, b in evidence)


def nms(times: list[float], gap: float = NMS_GAP_S) -> list[float]:
    out: list[float] = []
    for t in times:
        if all(abs(t - u) >= gap for u in out):
            out.append(t)
    return out


def line_time(seg, duration: float) -> float:
    return max(0.0, min(duration - 0.05, seg.end_s - 0.3))


def score_metrics(recs, proposals) -> dict:
    hits = {k: [] for k in KS}
    dist = []
    for r, ts in zip(recs, proposals, strict=True):
        for k in KS:
            hits[k].append(any(inside(t, r["evidence"]) for t in ts[:k]))
        dist.append(min((distance(t, r["evidence"]) for t in ts[:6]), default=float("inf")))
    out = {f"recall@{k}": float(np.mean(v)) for k, v in hits.items()}
    out["dist@6_median_s"] = float(np.median(dist))
    out["n"] = len(recs)
    out["_hits6"] = [bool(x) for x in hits[6]]
    return out


class ZeroShot:
    """The zero-shot Path A rules (shared state: one MiniLM scorer)."""

    def __init__(self) -> None:
        from videoqa.v2.head_a import HotMomentScorer  # noqa: PLC0415

        self.scorer = HotMomentScorer(device="cuda", cache_dir="cache/hf/hub")
        self._cache: dict = {}

    def windows(self, rec):
        from videoqa.retrieval import bm25_rank, build_windows  # noqa: PLC0415
        from videoqa.transcript import build_units  # noqa: PLC0415

        segs = segments(rec)
        units = build_units(segs, 20.0, 5.0)
        query = rec["question"] + " " + " ".join(rec["options"] or [])
        return segs, build_windows(bm25_rank(query, units, 64), units, 4, 1, rec["duration_s"])

    def minilm(self, rec, segs):
        from videoqa.v2.candidates import _with_neighbours  # noqa: PLC0415

        key = rec["qa_id"]
        if key not in self._cache:
            ctx = [_with_neighbours(segs, s) for s in segs]
            self._cache[key] = np.concatenate([self.scorer.score(rec["question"], ctx[i:i + 256])[:, 1]
                                               for i in range(0, len(ctx), 256)]) if ctx else np.zeros(0)
        return self._cache[key]

    def propose(self, method: str, rec) -> list[float]:
        from videoqa.retrieval import bm25_scores, tokenize  # noqa: PLC0415

        dur = rec["duration_s"]
        if method == "uniform":
            return [dur * (i + 0.5) / 6 for i in range(6)]
        segs, wins = self.windows(rec)
        if method == "bm25_window":
            return [(w.start_s + w.end_s) / 2 for w in sorted(wins, key=lambda w: -w.score)]
        if method == "bm25_line":
            q = tokenize(rec["question"] + " " + " ".join(rec["options"] or []))
            s = bm25_scores(q, [tokenize(x.text) for x in segs])
            return [line_time(segs[i], dur) for i in np.argsort(-np.asarray(s), kind="stable")[:64]]
        rel = self.minilm(rec, segs)
        if method == "v2":
            idx = [i for i, s in enumerate(segs) if any(s.end_s > w.start_s and s.start_s < w.end_s for w in wins)]
            idx = sorted(idx, key=lambda i: -rel[i])
            return [line_time(segs[i], dur) for i in idx[:64]]
        if method == "minilm_all":
            return [line_time(segs[i], dur) for i in np.argsort(-rel, kind="stable")[:64]]
        raise ValueError(method)


def stage_zeroshot(a) -> None:
    recs = load(a.split)
    zs = ZeroShot()
    table = {}
    for method in ("uniform", "bm25_window", "bm25_line", "v2", "minilm_all"):
        props = [zs.propose(method, r) for r in recs]
        table[method] = score_metrics(recs, props)
        if method not in ("uniform", "bm25_window"):
            table[method + "+nms"] = score_metrics(recs, [nms(p) for p in props])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"zeroshot_{a.split}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for m, r in table.items():
        print(f"{m:16s} " + " ".join(f"R@{k}={r[f'recall@{k}']:.3f}" for k in KS)
              + f"  dist@6={r['dist@6_median_s']:.1f}s n={r['n']}")


def stage_offsets(a) -> None:
    """Where is the evidence relative to the best-matching speech line? (train, human intervals only)"""
    recs = [r for r in load("train") if not r["weak"]]
    zs = ZeroShot()
    offs, inside_line = [], []
    for r in recs:
        segs = segments(r)
        if not segs:
            continue
        rel = zs.minilm(r, segs)
        best = segs[int(np.argmax(rel))]
        centre = np.mean([(x + y) / 2 for x, y in r["evidence"]])
        offs.append(float(centre - best.end_s))
        inside_line.append(any(x - TOL_S <= best.end_s - 0.3 <= y + TOL_S for x, y in r["evidence"]))
    offs = np.asarray(offs)
    bins = [-1e9, -120, -40, -10, 10, 40, 120, 1e9]
    hist = np.histogram(offs, bins=bins)[0]
    out = {"n": len(offs), "median_offset_s": float(np.median(offs)), "abs_median_s": float(np.median(np.abs(offs))),
           "best_line_end_hits_evidence": float(np.mean(inside_line)),
           "histogram": {f"{lo:g}..{hi:g}": int(c) for lo, hi, c in zip(bins[:-1], bins[1:], hist, strict=True)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offsets_train.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("zeroshot", "offsets"))
    p.add_argument("--split", default="dev", choices=("dev", "train"))
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"zeroshot": stage_zeroshot, "offsets": stage_offsets}[a.stage](a)


if __name__ == "__main__":
    main()
