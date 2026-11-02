"""Second supplement to the lecture-study method freeze (auditor P83): adds the S-src sensitivity analysis to the
reporting before any system runs on test. Additive: docs/v3/m3av_freeze.json and docs/v3/m3av_freeze_supplement.json
stay unchanged and are verified by hash; this record chains to both and holds the new hashes of the files the
addition changed (scripts/m3av_hits.py, scripts/run_m3av_test.sh) and of this verifier. No model, setting, pool,
selection rule, cohort, primary measure or contrast changes.

    python scripts/m3av_freeze_supplement2.py write                                  # docs/v3/m3av_freeze_supplement2.json
    python scripts/m3av_freeze_supplement2.py verify [--split test --stage inputs]   # chained: freeze, supplement 1, this
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m3av_freeze_supplement as s1  # noqa: E402

OUT = Path("docs/v3/m3av_freeze_supplement2.json")
S1_SHA, S1_COMMIT = "60ddc55ccb17b3fedd3bd26c4900e8c8b34a3adb5e02d797fcbec6318fa56347", "a07ba93"
CHANGED = ["scripts/m3av_hits.py", "scripts/run_m3av_test.sh"]
NEW_FILES = ["scripts/m3av_freeze_supplement2.py"]
S_SRC = ("S-src hit@4, a descriptive sensitivity analysis (auditor P83): for each question, each frozen visible span "
         "is intersected with the question's source_segment and only positive-length intersections are kept; other "
         "occurrences are ignored; the 1 s hit tolerance applies to the clipped intervals. Same 853 test questions, "
         "112 lectures, the same 4 selected frames and the same frozen systems; questions without a source-visible "
         "part stay in the denominator with no S-src hit for every system (test 3: m3av:CHI-167A9:153, "
         "m3av:CHI-218EE:22.46, m3av:CHI-DED34:241.45; dev 1: m3av:NIH-9A5A0:539.45). Reported: every system's S-src "
         "score and the L1/L2 differences with paired lecture-bootstrap 95% intervals (10,000 draws, seed 20261101); "
         "no p-values, no Holm, no significance claim. The primary measure, cohort, L1/L2, Holm family and all primary "
         "outputs are unchanged. The scorer stops if any question has an S-src hit without a primary hit.")


def chain() -> list[str]:
    bad = []
    if s1.sha(s1.FREEZE) != s1.FREEZE_SHA["lf"]:
        bad.append("the original freeze record changed")
    if s1.sha(s1.OUT) != S1_SHA:
        bad.append("freeze supplement 1 changed")
    return bad


def stage_write(_a) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    if bad := chain():
        raise SystemExit("\n".join(bad))
    if subprocess.run(["git", "status", "--porcelain", "--", *CHANGED, *NEW_FILES, "reports/m3av/hits_dev.json"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("commit the code first")
    rec1 = json.loads(s1.OUT.read_text(encoding="utf-8"))["code_after_freeze"]
    before = {**rec1["changed"], **rec1["new"]}
    dev = json.loads(Path("reports/m3av/hits_dev.json").read_text(encoding="utf-8"))
    rec = {"supplements": {"freeze": {"record": str(s1.FREEZE.as_posix()), "sha256": s1.FREEZE_SHA},
                           "supplement_1": {"record": str(s1.OUT.as_posix()), "sha256_lf": S1_SHA, "commit": S1_COMMIT}},
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "why": "auditor P83: accepted pre-outcome S-src sensitivity analysis, recorded before the first test run",
           "s_src": S_SRC,
           "code_after_supplement_1": {"changed": {f: {"supplement_1": before[f], "now": s1.sha(Path(f))} for f in CHANGED},
                                       "new": {f: s1.sha(Path(f)) for f in NEW_FILES}},
           "dev_check": {"report": "reports/m3av/hits_dev.json", "sha256": s1.sha(Path("reports/m3av/hits_dev.json")),
                         "result": "primary per-question hits, system scores, L1/L2 with Holm and secondary contrasts "
                                   "identical to the report before the addition; no S-src hit without a primary hit",
                         "s_src_dev_l1_l2": dev["s_src"]["contrasts"]},
           "primary_test_commands": {"file": "scripts/run_m3av_test.sh",
                                     "text": Path("scripts/run_m3av_test.sh").read_text(encoding="utf-8")}}
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{OUT.as_posix()}: sha256 {s1.sha(OUT)}")


def stage_verify(a) -> None:
    bad = chain()
    rec1 = json.loads(s1.OUT.read_text(encoding="utf-8"))
    rec = json.loads(OUT.read_text(encoding="utf-8"))
    bad += [f"model {n}" for n, v in s1.models().items() if v != rec1["models"][n]]
    want = {kind: dict(rec1["code_after_freeze"][kind]) for kind in ("changed", "new")}
    for f, v in rec["code_after_supplement_1"]["changed"].items():  # files this supplement changed: its hashes
        kind = "changed" if f in want["changed"] else "new"
        if want[kind].get(f) != v["supplement_1"]:
            bad.append(f"code {f} does not chain to supplement 1")
        want[kind][f] = v["now"]
    now = s1.code_changes()
    bad += [f"code {f}" for kind in ("changed", "new") for f, h in want[kind].items() if now[kind].get(f) != h]
    bad += [f"code {f} changed again" for f in now["changed"] if f not in want["changed"]]
    bad += [f"code {f}" for f, h in rec["code_after_supplement_1"]["new"].items() if s1.sha(Path(f)) != h]
    if a.split:
        man = json.loads(Path(f"docs/v3/m3av_manifest_{a.split}_{a.part}.json").read_text(encoding="utf-8"))
        have = s1.manifest(a.split, a.part)
        bad += [f"{a.part} {k}" for k in man if have.get(k) != man[k]]
    if bad:
        raise SystemExit("freeze verification failed:\n" + "\n".join(bad))
    print("freeze verified (freeze, supplement 1, supplement 2): models, code" + (f", {a.split} {a.part}" if a.split else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("write", "verify"))
    p.add_argument("--split", choices=("dev", "test"), default=None)
    p.add_argument("--stage", dest="part", choices=("inputs", "artifacts"), default=None)
    a = p.parse_args()
    if bool(a.split) != bool(a.part):
        raise SystemExit("give both --split and --stage, or neither")
    {"write": stage_write, "verify": stage_verify}[a.action](a)


if __name__ == "__main__":
    main()
