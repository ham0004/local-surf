"""Does the retained speech change WHICH frames are useful?  (no extra teacher calls)

The main run measures the same Path A frames under two transcript contexts:

  without speech   Head A frame labels:  R({}, {f}) - R({}, {})          (T0 = empty)
  with speech      Head B labels:        R(T, {f}) - R(T, {})            (T = retained excerpt)

Disagreement shows that positive answer-utility labels change with context.
It does not identify why: a lost positive gain can reflect a correct transcript
baseline (a ceiling effect), or speech disrupting a frame-only correct answer.
Joint correctness outcomes below distinguish these cases. Cohen's kappa is
chance-adjusted agreement, not a statistical independence test.

Reports the gain table, joint correctness outcomes and lecture-bootstrap intervals.

    python scripts/v2_transcript_conditioning.py --run runs/v2_main \
        --out reports/v2_review/transcript_conditioning_review.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def summarize_conditioning(no_speech_rows, with_speech_rows, bootstrap_samples: int = 5000) -> dict:
    """Compare matched labels without making new teacher calls.

    Head A stores frame correctness as baseline + visual gain; Head B stores
    transcript-only and combined correctness directly. All are binary outcomes.
    """
    no_speech = {(r["qa_id"], r["candidate_id"]): r for r in no_speech_rows}
    with_speech = {(r["qa_id"], r["candidate_id"]): r for r in with_speech_rows}
    keys = sorted(set(no_speech) & set(with_speech))
    if not keys:
        raise ValueError("No matching question/frame labels were found")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    u0 = np.array([no_speech[k]["visual_gain"] > 0 for k in keys])
    u1 = np.array([with_speech[k]["gain"] > 0 for k in keys])
    vids = np.array([no_speech[k]["video_id"] for k in keys])
    speech_alone_correct = np.array([with_speech[k]["before"] == 1.0 for k in keys])
    frame_alone_correct = np.array([no_speech[k]["base"] + no_speech[k]["visual_gain"] == 1.0 for k in keys])
    combined_correct = np.array([with_speech[k]["after"] == 1.0 for k in keys])
    empty_correct = np.array([no_speech[k]["base"] == 1.0 for k in keys])
    lost = u0 & ~u1
    redundant = frame_alone_correct & speech_alone_correct & combined_correct
    destructive = frame_alone_correct & speech_alone_correct & ~combined_correct
    speech_interference = frame_alone_correct & ~speech_alone_correct & ~combined_correct

    rng = np.random.default_rng(0)
    uv = sorted(set(vids))
    boot_diff, boot_kappa = [], []
    for _ in range(bootstrap_samples):
        pick = np.concatenate([np.where(vids == v)[0] for v in rng.choice(uv, len(uv))])
        boot_diff.append(u0[pick].mean() - u1[pick].mean())
        boot_kappa.append(kappa(u0[pick], u1[pick]))
    out = {
        "matched_frames": len(keys), "questions": len({k[0] for k in keys}), "lectures": len(uv),
        "useful_without_speech": int(u0.sum()), "useful_with_speech": int(u1.sum()),
        "table": {"useful_both": int((u0 & u1).sum()), "useful_only_without_speech": int((u0 & ~u1).sum()),
                  "useful_only_with_speech": int((~u0 & u1).sum()), "useful_neither": int((~u0 & ~u1).sum())},
        "frames_where_speech_alone_already_answered": int(speech_alone_correct.sum()),
        "frames_where_no_evidence_already_answered": int(empty_correct.sum()),
        "joint_correctness_outcomes": [
            {"frame_alone_correct": f, "speech_alone_correct": s, "combined_correct": c,
             "count": int(((frame_alone_correct == f) & (speech_alone_correct == s)
                           & (combined_correct == c)).sum())}
            for f in (False, True) for s in (False, True) for c in (False, True)
        ],
        "useful_only_without_speech_decomposition": {
            "redundant_success_both_sources_and_combination_correct": int((lost & redundant).sum()),
            "destructive_combination_both_sources_correct_alone": int((lost & destructive).sum()),
            "speech_interference_frame_correct_speech_and_combination_wrong": int((lost & speech_interference).sum()),
            "other": int((lost & ~(redundant | destructive | speech_interference)).sum()),
        },
        "joint_only_success_neither_source_correct_alone": int(
            (~frame_alone_correct & ~speech_alone_correct & combined_correct).sum()),
        "rate_difference_without_minus_with": float(u0.mean() - u1.mean()),
        "rate_difference_ci95_lecture_bootstrap": [float(np.percentile(boot_diff, 2.5)),
                                                    float(np.percentile(boot_diff, 97.5))],
        "cohen_kappa": kappa(u0, u1),
        "kappa_ci95_lecture_bootstrap": [float(np.nanpercentile(boot_kappa, 2.5)),
                                         float(np.nanpercentile(boot_kappa, 97.5))],
        "reading": ("Kappa measures chance-adjusted agreement of positive-gain labels, not independence. "
                    "Lost positive gain can reflect redundant success or negative interference; see the joint "
                    "correctness decomposition. Counts are frame/question pairs clustered within questions "
                    "and lectures, not independent frame observations."),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2_main")
    ap.add_argument("--out", default="reports/v2_review/transcript_conditioning_review.json",
                    help="new derived report path; historical run outputs are left intact")
    a = ap.parse_args()
    run = Path(a.run)
    read_rows = lambda name: [json.loads(x) for x in (run / name).read_text(encoding="utf-8").splitlines()
                             if x.strip()]
    out = summarize_conditioning(read_rows("head_a_labels.jsonl"), read_rows("rows_independent.jsonl"))
    out["source_run"] = str(run)
    destination = Path(a.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
