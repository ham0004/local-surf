"""Primary measure of the lecture study for every system, without the answer model (docs/v3/m3av_protocol.md).

Source-slide interval hit@4: 1 when one of the 4 frames a system would show lies inside a visible span of a valid
evidence segment (1 s tolerance; the same test as `chosen_hits_evidence` in v3_baseline_cycle.py). Frames come
from the saved pools with the same `select` the answer stage uses. Paired bootstrap over lectures (10,000
resamples, seed 20261101); raw p-values and Holm over exactly the two primary contrasts L1 and L2 (amendment 4).
Stops if any frozen system lacks a question of the cohort (no primary on a subset). Also: frames actually shown.

S-src hit@4 (second freeze supplement, auditor P83): a descriptive sensitivity analysis, never confirmatory. The
same frames and test, but each visible span is clipped to the question's source segment (positive-length parts
only) and other occurrences are ignored. Questions with no source-visible part stay in the cohort with no S-src
hit. Reported per system with L1/L2 differences and their paired bootstrap intervals; no p-values, no Holm.

    python scripts/m3av_hits.py --split dev                     # reports/m3av/hits_dev.{json,md}
    python scripts/m3av_hits.py --split test                    # frozen systems only
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

B, SEED, ALPHA = 10_000, 20261101, 0.05
SYSTEMS = {  # name: (pool, selector); the frozen systems (docs/v3/m3av_freeze.json), S1t shows S1's frames
    "S0 starting design": ("v2", "clip"),
    "S1 best framework (also S1t's frames)": ("hybrid", "mmropt"),
    "S2c Head A (CG-Bench)": ("hybrid_ahead", "mmropt"),
    "S2m Head A (M3AV train)": ("hybrid_ahead_m3av", "mmropt"),
    "S3 Head B": ("hybrid", "file:{ds}_hybrid_ev_set"),
    "R2 4 evenly spaced": ("uniform4d", "clip"),
    "R3 dense MobileCLIP top-4 (96 ranked)": ("dense4x", "clip"),
    "R4 dense option-aware top-4 (96 ranked)": ("dense4ox", "clip"),
    "R5 VSI-inspired (96 ranked)": ("vsi4x", "clip"),
}
DEV_ONLY = {  # dev diagnostics (research log step 86); never run on test
    "S2c, picker A_rank": ("hybrid_ahead", "A_rank"),
    "S2m, picker A_rank": ("hybrid_ahead_m3av", "A_rank"),
    "R3 dense MobileCLIP top-4, h0": ("dense4d0", "clip"),
    "R3 dense MobileCLIP top-4, h3": ("dense4d3", "clip"),
    "R4 dense option-aware top-4, h0": ("dense4od0", "clip"),
    "R4 dense option-aware top-4, h3": ("dense4od3", "clip"),
    "R5 VSI-inspired, h0": ("vsi4d0", "clip"),
    "R5 VSI-inspired, h3": ("vsi4d3", "clip"),
    # superseded: 4-candidate pool that merges repeated slides and shows fewer than 4 pictures
    "R3 (4-candidate pool, merged)": ("dense4", "clip"),
}
S2M, S1, R5 = "S2m Head A (M3AV train)", "S1 best framework (also S1t's frames)", "R5 VSI-inspired (96 ranked)"
PRIMARY = {"L1": (S2M, R5), "L2": (S2M, S1)}                       # amendment 4; Holm over exactly these two
SECONDARY = {"S2m - R3": (S2M, "R3 dense MobileCLIP top-4 (96 ranked)"),
             "S2m - R4": (S2M, "R4 dense option-aware top-4 (96 ranked)"), "S2m - S2c": (S2M, "S2c Head A (CG-Bench)")}


def paired(a: dict, b: dict, lecture: dict) -> dict:
    qs = sorted(a)
    byl = collections.defaultdict(list)
    for q in qs:
        byl[lecture[q]].append(float(a[q]) - float(b[q]))
    ls = sorted(byl)
    rng = np.random.default_rng(SEED)
    boots = np.array([np.mean(np.concatenate([byl[ls[i]] for i in rng.choice(len(ls), len(ls))])) for _ in range(B)])
    p = min(1.0, 2 * (min(np.sum(boots <= 0), np.sum(boots >= 0)) + 1) / (B + 1))
    return {"diff": round(100 * float(np.mean([a[q] - b[q] for q in qs])), 1),
            "lo": round(100 * float(np.percentile(boots, 2.5)), 1), "hi": round(100 * float(np.percentile(boots, 97.5)), 1),
            "p": float(p), "n": len(qs), "lectures": len(ls)}


def holm(ps: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    out, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def source_spans(split: str, its: dict) -> dict:
    """Visible spans clipped to the source segment (S-src); the cohort's visible spans are the evidence intervals."""
    out = {}
    for line in Path(f"data/m3av/questions_{split}_visible.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        if q["qa_id"] not in its:
            continue
        if [list(x) for x in its[q["qa_id"]].evidence_intervals_s] != q["visible_spans"]:
            raise SystemExit(f"visible spans differ from the evidence intervals: {q['qa_id']}")
        a0, b0 = q["source_segment"]
        out[q["qa_id"]] = [(max(a, a0), min(b, b0)) for a, b in q["visible_spans"] if min(b, b0) > max(a, a0)]
    if set(out) != set(its):
        raise SystemExit("cohort file and qa.jsonl hold different questions")
    return out


def missing_ids(ds: str, pool: str, rule: str, qids: list) -> list:
    from videoqa.v2.candidates import _safe  # noqa: PLC0415

    d = pool_dir(ds, pool)
    gone = {q for q in qids if not (d / _safe(q) / "pool.json").exists()}
    if rule.startswith("file:"):
        f = Path(f"runs/v3_head_b/selections/{rule[5:]}.json")
        sel = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        gone |= {q for q in qids if q not in sel}
    return sorted(gone)


def main() -> None:
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    a = p.parse_args()
    ds = f"m3av_{a.split}"
    its = {it.qa.qa_id: it.qa for it in items(ds)}
    lecture = {q: v.video_id for q, v in its.items()}
    qids = sorted(its)
    systems = dict(SYSTEMS, **(DEV_ONLY if a.split == "dev" else {}))
    # complete-cohort check: every frozen system covers every question; only dev diagnostics may be absent
    incomplete = {k: g for k, (pool, rule) in systems.items() if (g := missing_ids(ds, pool, rule.format(ds=ds), qids))}
    bad = {k: len(g) for k, g in incomplete.items() if k in SYSTEMS}
    if bad:
        raise SystemExit(f"incomplete frozen systems on {ds} ({len(qids)} questions), questions missing: {bad}")
    enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
    qo = dict(zip(qids, enc.embed_texts([its[q].question + " Options: " + ", ".join(its[q].options) for q in qids]),
                  strict=True))
    src = source_spans(a.split, its)
    hits, src_hits, shown = {}, {}, {}
    for name, (pool, rule) in systems.items():
        if name in incomplete:
            continue
        d, rule = pool_dir(ds, pool), rule.format(ds=ds)
        h, hs, n = {}, {}, []
        for q in qids:
            frames = select(load_pool(d, q), rule, 4, qo[q])
            h[q] = any(x - 1.0 <= c.time_s <= y + 1.0 for c in frames for x, y in its[q].evidence_intervals_s)
            hs[q] = any(x - 1.0 <= c.time_s <= y + 1.0 for c in frames for x, y in src[q])
            if hs[q] and not h[q]:
                raise SystemExit(f"S-src hit without a primary hit: {name}, {q}")
            n.append(len(frames))
        hits[name], src_hits[name] = h, hs
        shown[name] = {"mean": round(float(np.mean(n)), 2), "under_4": int(sum(x < 4 for x in n))}
    prim = {k: paired(hits[x], hits[y], lecture) for k, (x, y) in PRIMARY.items()}
    for k, v in holm({k: v["p"] for k, v in prim.items()}).items():
        prim[k].update(p_holm=v, significant_at_0_05=bool(v <= ALPHA))
    out = {"split": a.split, "questions": len(qids), "lectures": len(set(lecture.values())),
           "absent_dev_diagnostics": sorted(incomplete),
           "systems": {k: {"hit@4": round(100 * float(np.mean(list(v.values()))), 1), **shown[k]} for k, v in hits.items()},
           "primary": {k: {"contrast": f"{x} - {y}", **prim[k]} for k, (x, y) in PRIMARY.items()},
           "secondary": {k: paired(hits[x], hits[y], lecture) for k, (x, y) in SECONDARY.items()},
           "per_question": {k: {q: int(v[q]) for q in qids} for k, v in hits.items() if k in SYSTEMS}}
    no_src = sorted(q for q in qids if not src[q])
    src_l = {k: {kk: vv for kk, vv in paired(src_hits[x], src_hits[y], lecture).items() if kk != "p"}
             for k, (x, y) in PRIMARY.items()}
    out["s_src"] = {"definition": "sensitivity analysis (second freeze supplement, auditor P83), descriptive only: "
                                  "visible spans clipped to the source segment, other occurrences ignored, same frames "
                                  "and 1 s tolerance; no p-values, no Holm",
                    "questions_without_source_visible_part": len(no_src), "ids_without_source_visible_part": no_src,
                    "systems": {k: round(100 * float(np.mean(list(v.values()))), 1) for k, v in src_hits.items()},
                    "contrasts": {k: {"contrast": f"{x} - {y}", **src_l[k]} for k, (x, y) in PRIMARY.items()},
                    "per_question": {k: {q: int(v[q]) for q in qids} for k, v in src_hits.items() if k in SYSTEMS}}
    Path("reports/m3av").mkdir(parents=True, exist_ok=True)
    Path(f"reports/m3av/hits_{a.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    def row(name: str, c: dict) -> str:
        return f"| {name} | {c['diff']:+.1f} ({c['lo']:+.1f}..{c['hi']:+.1f}) | {c['p']:.4g} |"

    lines = [f"# Source-slide interval hit@4, M3AV {a.split} ({len(qids)} questions, {out['lectures']} lectures)", "",
             "Paired bootstrap over lectures (10,000 resamples, seed 20261101); every frozen system covers every question."
             + (" Dev results were used to choose methods; they are not a confirmation." if a.split == "dev" else ""),
             "", "## Primary contrasts (amendment 4; Holm over L1 and L2)", "",
             "| Contrast | Difference, points (95% interval) | raw p | Holm p | significant at 0.05 |", "|---|---|---|---|---|"]
    lines += [row(f"{k}: {x} - {y}", prim[k]) + f" {prim[k]['p_holm']:.4g} | "
              f"{'yes' if prim[k]['significant_at_0_05'] else 'no'} |" for k, (x, y) in PRIMARY.items()]
    lines += ["", "## Secondary contrasts (no correction; descriptive)", "",
              "| Contrast | Difference, points (95% interval) | raw p |", "|---|---|---|"]
    lines += [row(f"{x} - {y}", out["secondary"][k]) for k, (x, y) in SECONDARY.items()]
    lines += ["", "## Systems", "", "| System | hit@4 (%) | frames shown (mean) | questions with < 4 frames |",
              "|---|---|---|---|"]
    lines += [f"| {k} | {v['hit@4']} | {v['mean']} | {v['under_4']} |" for k, v in out["systems"].items()]
    lines += ["", "## Sensitivity: S-src hit@4 (descriptive, not confirmatory)", "",
              "Visible spans clipped to the question's source segment; other occurrences ignored; same frames. "
              f"{len(no_src)} of {len(qids)} questions have no source-visible part and count as misses for every "
              "system. A lower S-src score is not worse retrieval in itself: genuine repeats are excluded.", "",
              "| Contrast | Difference, points (95% interval) |", "|---|---|"]
    lines += [f"| {k}: {x} - {y} | {src_l[k]['diff']:+.1f} ({src_l[k]['lo']:+.1f}..{src_l[k]['hi']:+.1f}) |"
              for k, (x, y) in PRIMARY.items()]
    lines += ["", "| System | S-src hit@4 (%) | primary hit@4 (%) |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {out['systems'][k]['hit@4']} |" for k, v in out["s_src"]["systems"].items()]
    if incomplete:
        lines += ["", "Dev diagnostics not available: " + ", ".join(sorted(incomplete))]
    Path(f"reports/m3av/hits_{a.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
