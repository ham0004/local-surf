"""Does the retained speech change WHICH frames are useful?  (no extra teacher calls)

The main run measures the same Path A frames under two transcript contexts:

  without speech   Head A frame labels:  R({}, {f}) - R({}, {})          (T0 = empty)
  with speech      Head B labels:        R(T, {f}) - R(T, {})            (T = retained excerpt)

If frame usefulness were independent of the transcript, the two labels would
agree. Disagreement is direct evidence that transcript conditioning matters:

  useful only WITHOUT speech   the speech already carried that information (redundant frame)
  useful only WITH speech      the frame helps only in combination with the speech (complementary)

Reports the 2x2 table, Cohen's kappa, and lecture-bootstrap intervals.

    python scripts/v2_transcript_conditioning.py --run runs/v2_main
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2_main")
    a = ap.parse_args()
    run = Path(a.run)
    no_speech = {(r["qa_id"], r["candidate_id"]): r for r in map(json.loads, (run / "head_a_labels.jsonl")
                 .read_text(encoding="utf-8").splitlines()) if r}
    with_speech = {(r["qa_id"], r["candidate_id"]): r for r in map(json.loads, (run / "rows_independent.jsonl")
                   .read_text(encoding="utf-8").splitlines()) if r}
    keys = sorted(set(no_speech) & set(with_speech))
    u0 = np.array([no_speech[k]["visual_gain"] > 0 for k in keys])
    u1 = np.array([with_speech[k]["gain"] > 0 for k in keys])
    vids = np.array([no_speech[k]["video_id"] for k in keys])
    speech_alone_correct = np.array([with_speech[k]["before"] == 1.0 for k in keys])

    rng = np.random.default_rng(0)
    uv = sorted(set(vids))
    boot_diff, boot_kappa = [], []
    for _ in range(5000):
        pick = np.concatenate([np.where(vids == v)[0] for v in rng.choice(uv, len(uv))])
        boot_diff.append(u0[pick].mean() - u1[pick].mean())
        boot_kappa.append(kappa(u0[pick], u1[pick]))
    out = {
        "matched_frames": len(keys), "questions": len({k[0] for k in keys}), "lectures": len(uv),
        "useful_without_speech": int(u0.sum()), "useful_with_speech": int(u1.sum()),
        "table": {"useful_both": int((u0 & u1).sum()), "useful_only_without_speech": int((u0 & ~u1).sum()),
                  "useful_only_with_speech": int((~u0 & u1).sum()), "useful_neither": int((~u0 & ~u1).sum())},
        "frames_where_speech_alone_already_answered": int(speech_alone_correct.sum()),
        "rate_difference_without_minus_with": float(u0.mean() - u1.mean()),
        "rate_difference_ci95_lecture_bootstrap": [float(np.percentile(boot_diff, 2.5)),
                                                    float(np.percentile(boot_diff, 97.5))],
        "cohen_kappa": kappa(u0, u1),
        "kappa_ci95_lecture_bootstrap": [float(np.nanpercentile(boot_kappa, 2.5)),
                                         float(np.nanpercentile(boot_kappa, 97.5))],
        "reading": ("kappa near 1 = speech does not change which frames help; kappa near 0 = the useful frames "
                    "under speech are largely different ones"),
    }
    (run / "transcript_conditioning.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
