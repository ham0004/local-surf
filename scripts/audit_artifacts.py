"""Reproduce the protocol audit of saved label artifacts (read-only).

Recomputes, from a label directory written by `videoqa build-labels`, the
counts behind docs/corrections/2026-09-28.md:

  * questions, distinct source videos, rows, matched pairs
  * triples whose targeted and control damage affected different word counts
    (any difference, > 25 % and > 50 % of targeted words, worst case)
  * matched pairs whose two rows differ in requested time, in scout features,
    or in candidate-source features (these should be identical if transcript
    state were the only difference)
  * EXPAND_TRANSCRIPT pairs (same action name, possibly different text)
  * pairs with byte-identical input features but different target gains
  * sha256 of the artifact files, so the audited inputs are pinned

Usage:
  uv run python scripts/audit_artifacts.py --labels runs/lvb_full/labels_train \
      --out reports/audit_2026-09-28/labels_train_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.features import CANDIDATE_FEATURES  # noqa: E402
from videoqa.labels import read_rows  # noqa: E402

IDX = {n: i for i, n in enumerate(CANDIDATE_FEATURES)}
SCOUT = [IDX[n] for n in CANDIDATE_FEATURES if n.startswith(("scout_", "scene_"))]
SOURCE = [IDX[n] for n in CANDIDATE_FEATURES if n.startswith("src_")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.labels)
    rows, pairs = read_rows(d)

    # --- damage matching, from the spec-schema examples (one per question x condition x step)
    words: dict[str, dict[str, int]] = {}
    for line in (d / "examples.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        cond = ex["transcript_condition"]
        if cond in ("targeted_damage", "control_damage"):
            words.setdefault(ex["qa_id"], {})[cond] = ex["damage"]["words_affected"]
    unequal = [(q, w["targeted_damage"], w["control_damage"]) for q, w in words.items()
               if len(w) == 2 and w["targeted_damage"] != w["control_damage"]]
    rel = lambda t, c: abs(t - c) / max(1, t)  # noqa: E731
    worst = max(unequal, key=lambda x: rel(x[1], x[2]), default=None)

    # --- pair identity
    times = scout = source = expand = identical = identical_diff_gain = 0
    for i, j in pairs:
        r, c = rows[i], rows[j]
        if r.candidate_time_s is None or c.candidate_time_s is None:
            expand += 1
        elif r.candidate_time_s != c.candidate_time_s:
            times += 1
        if [r.features[k] for k in SCOUT] != [c.features[k] for k in SCOUT]:
            scout += 1
        if [r.features[k] for k in SOURCE] != [c.features[k] for k in SOURCE]:
            source += 1
        if r.features == c.features:
            identical += 1
            identical_diff_gain += r.gain != c.gain

    report = {
        "labels": str(d),
        "questions": len({r.qa_id for r in rows}),
        "source_videos": len({r.video_id for r in rows}),
        "rows": len(rows),
        "pairs": len(pairs),
        "damage_matching": {
            "triples": len(words),
            "unequal_affected_words": len(unequal),
            "unequal_over_25pct": sum(rel(t, c) > 0.25 for _, t, c in unequal),
            "unequal_over_50pct": sum(rel(t, c) > 0.50 for _, t, c in unequal),
            "worst": {"qa_id": worst[0], "targeted_words": worst[1], "control_words": worst[2]} if worst else None,
        },
        "pair_identity": {
            "unequal_requested_time": times,
            "unequal_scout_features": scout,
            "unequal_source_features": source,
            "expand_transcript_pairs": expand,
            "identical_features": identical,
            "identical_features_but_different_gain": identical_diff_gain,
        },
        "sha256": {p.name: sha256(p) for p in sorted(d.iterdir()) if p.is_file()},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
