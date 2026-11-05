"""Blind human review of the 100 lecture test packets: the analyses declared in docs/v3/m3av_protocol.md
("Question review", amendment 3), fixed in research log steps 98-99 before any human verdict was seen (auditor
P142-P146). Reads saved files only, runs no model. Everything below is fixed before the human answers are opened.

Inputs, all under one private folder (default runs/m3av/review_human/, not in git):
  - raw/: the exports exactly as received. The review page's shared store saved by the owner (JSON; each reviewer's
    document holds an `answers` map, one record per item with an ISO time `t`), or a reviewer's "Copy my answers"
    text (CSV: `reviewer,<name>`, then item, qa_id, verdict, key, one, slide, legible, shortcut, reason, note), or a
    filled form.csv. The script hashes every raw file it reads.
  - identities.json: the private provenance map. It lists every source and gives each a stable private reviewer
    key; names never leave this file. A store source maps each document id; a CSV source names its reviewer and
    the time it was received. Unlisted sources or documents stop the run.
        {"sources": [{"file": "raw/store_cutoff.json", "kind": "store", "role": "cutoff", "docs": {"<doc id>": "r1"}},
                     {"file": "raw/bob.csv", "kind": "csv", "role": "cutoff", "reviewer": "r2",
                      "received": "2026-11-08T21:00:00+06:00"}],
         "resolutions": {"r2": {"<qa_id>": "raw/bob.csv"}}}

Rules (fixed now):
  - Eligibility: reviewers are the owner's group members, blind to systems, results and both model reviews. The
    builder, the auditor and the owner are not reviewers.
  - Cutoff: an answer counts only if made (store: its `t`) or received (CSV: `received`) by CUTOFF, 8 November 2026,
    23:59 Dhaka. Later answers are late, reported and not used.
  - One final answer per reviewer and question. Identical copies (another export, another snapshot) count once.
    Different store versions: the one with the latest `t` (a revision). Any other difference (a CSV against the
    store or another CSV, two rows in one file, equal times) is a conflict. It stops the run unless identities.json
    names the source the reviewer confirms as final; import order never decides.
  - Inputs: every file in raw/ must be listed in identities.json and every listed file must exist; one snapshot
    holds one answer per reviewer and question (a different second answer stops the run). The frozen inputs must
    match their recorded hashes (EXPECTED); each model review must give exactly one valid verdict per sampled
    question. Output source labels are pseudonymous ("raw source 1 (store, cutoff)").
  - Snapshots: the owner saves the store at the cutoff (role "cutoff") before any clarification is asked, and a
    separate later snapshot for the clarifications (role "clarification"); neither is edited.
  - Validity: a field left empty makes the answer incomplete (missing for that reviewer, not a vote). A non-empty
    value outside the vocabulary (verdict accept/reject/uncertain; checks y/n; shortcut none/knowledge/transcript;
    reason OK, KEY, AMB, UNSUP, ILLEG, SPAN, UI, OTHER) or a question outside the sample stops the run. Reject and
    uncertain need a note. The shortcut never affects validity.
  - Contradiction: accept with a check answered n or a reason other than OK; reject or uncertain with reason OK;
    reason KEY with key y, AMB with one y, UNSUP with slide y, ILLEG with legible y. Nothing is inferred from the
    fields. A contradictory answer is sent back to its reviewer, blind, to recheck. A revision of that item made by
    CLARIFICATION_CUTOFF (9 November, 23:59 Dhaka) in a source with role "clarification" replaces it. Otherwise it
    stays unresolved and is not a vote. Clarification sources are read for flagged items only.
  - Combination: the verdict that has a strict majority of an item's valid votes; no strict majority, uncertain
    (reported as "no majority"). Items with no valid vote are unreviewed, which is distinct from uncertain.

Analyses:
  1. Rejection rate (and reject-or-uncertain rate): lecture-balanced and the cohort's question-weighted
     sum(n_l r_l) / sum(n_l), with 95% lecture-bootstrap intervals (10,000 draws, seed 20261101). These are the
     planned estimates only when all 100 items are reviewed; otherwise they are observed-review rates conditional on
     the responders, with the missing items and page blocks listed. The intervals cover sampling only, not
     annotator error or non-response.
  2. Coverage per item and per reviewer; counts of the checks, shortcuts and reason codes over complete human
     answers (which can exceed 100 when reviewers overlap); agreement of the combined verdict with each model review
     and between humans (percent, Cohen's kappa, 3 x 3 tables); per-item votes under pseudonyms H1, H2, ...
  3. Secondary 1 (declared): every system's hit@4 and S-src hit@4 and every condition's answer accuracy, on the
     cohort without the human-rejected sample questions, with one mask for all systems. Uncertain and unreviewed
     questions are kept, which does not validate them. L1 and L2 use paired lecture-bootstrap intervals, with no
     p-values or Holm. Also, labelled separately: the conservative mask without rejected-or-uncertain items, and
     hit@4 on the accepted sample alone (descriptive, not a cohort estimate). Each mask lists its dropped ids and
     reasons, its retained and removed questions and lectures, and its per-lecture attrition.

    python scripts/m3av_human_review.py                     # reports/m3av/human_review_test.{json,md}
    python scripts/m3av_human_review.py --dir <folder> --out <folder>
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

B, SEED = 10_000, 20261101
DHAKA = datetime.timezone(datetime.timedelta(hours=6))
CUTOFF = datetime.datetime(2026, 11, 8, 23, 59, 59, tzinfo=DHAKA)
CLARIFICATION_CUTOFF = datetime.datetime(2026, 11, 9, 23, 59, 59, tzinfo=DHAKA)
VERDICTS = ("accept", "reject", "uncertain")
CHECKS = ("key", "one", "slide", "legible")
VOCAB = {"verdict": VERDICTS, "key": ("y", "n"), "one": ("y", "n"), "slide": ("y", "n"), "legible": ("y", "n"),
         "shortcut": ("none", "knowledge", "transcript"),
         "reason": ("OK", "KEY", "AMB", "UNSUP", "ILLEG", "SPAN", "UI", "OTHER")}
FIELDS = ("verdict", *CHECKS, "shortcut", "reason", "note")
FORM_COLS = {"verdict": "verdict", "key correct": "key", "one best answer": "one", "supported by the slide": "slide",
             "legible at 360p": "legible", "shortcut": "shortcut", "reason code": "reason", "note": "note",
             "qa_id": "qa_id", "n": "item", "item": "item"}
REASON_CHECK = {"KEY": "key", "AMB": "one", "UNSUP": "slide", "ILLEG": "legible"}
BLOCKS = {"h1": (1, 50), "h2": (51, 100), "t1": (1, 34), "t2": (35, 67), "t3": (68, 100),
          "q1": (1, 25), "q2": (26, 50), "q3": (51, 75), "q4": (76, 100)}   # the page's blocks, in page order
MODEL_REVIEWS = {"builder": Path("collab/review_builder_test.csv"), "auditor": Path("collab/review_auditor_test.csv")}
PAGE_FILES = (Path("collab/review_page/index.html"), Path("collab/review_page/template.html"),
              Path("collab/review_page/items.json"))
# The frozen inputs and the hashes recorded for them before any human verdict (research log steps 88-90, 95, 99):
# raw bytes, except the QA records, sealed with line ends normalised (docs/v3/m3av_qa_test_outputs.json, sha256_lf).
EXPECTED = {
    "reports/m3av/review_sample_test.json": ("bytes", "32f36bf9dfe6aa3e88ad74409c0f783436c9d195c4933a7e7647285b330e5057"),
    "reports/m3av/hits_test.json": ("bytes", "6ef21809895cab37194152733ba31a9ae4b68ed2f4aef6dafeac980716088ef1"),
    "runs/m3av/qa/test/records.jsonl": ("lf", "78760d3248ccd6e7ba69dbdf420ad88b037340d44c4c60b57b8cd721f65cc51f"),
    "collab/review_builder_test.csv": ("bytes", "8e77f77745f2ad65715662d7a112b215a35ccbee509aacdedaf6245426f699f9"),
    "collab/review_auditor_test.csv": ("bytes", "b60a0bf85e0fba92ef2e2ecc2b12a7736c7092f4cd0b5a77c29cec92befe2cb4"),
    "collab/review_page/index.html": ("bytes", "0f2a69f4c2120a28d7e7b1e4554026dd9b98a0924f0dd4fcd00c37088f19d597"),
    "collab/review_page/template.html": ("bytes", "1ebb157d3508d050c84829fca6bdee48c4bb9167500324a7d069e97a6c5fc7b3"),
    "collab/review_page/items.json": ("bytes", "dc258efe612668436a2c766c80062efe369c7606c5eb12b0b2aeab1a4886031d"),
}


class InputError(SystemExit):
    pass


def sha(path: Path, lf: bool = False) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n") if lf else b).hexdigest()


def verify_expected(expected: dict, root: Path = Path(".")) -> dict:
    """Every frozen input must exist and match its recorded hash; otherwise the run stops."""
    for rel, (kind, h) in expected.items():
        p = root / rel
        if not p.exists():
            raise InputError(f"missing frozen input: {rel}")
        if sha(p, kind == "lf") != h:
            raise InputError(f"{rel} does not match its recorded SHA-256 ({kind})")
    return {rel: h for rel, (_k, h) in expected.items()}


def read_model(path: Path, sample: list) -> dict:
    """One valid verdict per sampled question from a model review; duplicates or other values stop the run."""
    out = {}
    for r in read_csv(path):
        q = r.get("qa_id", "")
        if q in out:
            raise InputError(f"{path.name}: {q} appears twice")
        if r.get("verdict") not in VERDICTS:
            raise InputError(f"{path.name}: {q} has verdict {r.get('verdict')!r}")
        out[q] = r["verdict"]
    if set(out) != set(sample):
        raise InputError(f"{path.name} does not cover exactly the sample")
    return out


def when(s: str) -> datetime.datetime:
    t = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise InputError(f"time without a zone: {s}")
    return t


def norm(row: dict) -> dict:
    """Page field names (or the packet form's column names) -> the page's fields, values stripped."""
    page, out = set(FORM_COLS.values()), {}
    for k, v in row.items():
        k = (k or "").strip().lower()
        key = k if k in page else next((FORM_COLS[c] for c in FORM_COLS if k.startswith(c)), None)
        if key:
            out[key] = str("" if v is None else v).strip()
    for c in ("verdict", *CHECKS, "shortcut"):
        out[c] = out.get(c, "").lower()
    out["reason"] = out.get("reason", "").upper()
    return out


def answer_of(a: dict) -> tuple:
    return tuple(a.get(f, "") for f in FIELDS)


def read_csv(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("reviewer,"):
        text = "\n".join(lines[1:])
    return [norm(r) for r in csv.DictReader(io.StringIO(text))]


def read_store(path: Path) -> dict:
    """Document id -> document, for every object holding an `answers` map; the id is the object's own id field, or
    else the key it sits under."""
    docs: dict = {}

    def walk(x, key=None):
        if isinstance(x, dict):
            if isinstance(x.get("answers"), dict):
                did = str(x.get("id") or x.get("doc_id") or x.get("path") or key)
                if did in docs:
                    raise InputError(f"{path.name}: document {did} appears twice")
                docs[did] = x
                return
            inner = x.get("id") or x.get("doc_id") or x.get("path")
            for k, v in x.items():
                walk(v, inner or k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)

    walk(json.loads(path.read_text(encoding="utf-8")))
    return docs


def records(ident: dict, base: Path) -> tuple[list, dict]:
    """Every answer record (reviewer, qa_id, answer, time, source, role, kind) and the raw-file hashes."""
    out, hashes = [], {}
    listed = {str(Path(s["file"]).as_posix()) for s in ident["sources"]}
    raw = base / "raw"
    present = {p.relative_to(base).as_posix() for p in raw.rglob("*") if p.is_file()} if raw.exists() else set()
    if present - listed or listed - present:
        raise InputError(f"raw/ and identities.json disagree: unlisted {sorted(present - listed)}, "
                         f"missing {sorted(listed - present)}")
    for i, s in enumerate(ident["sources"], 1):
        f = base / s["file"]
        hashes[f"raw source {i} ({s['kind']}, {s.get('role', 'cutoff')})"] = sha(f)
        role = s.get("role", "cutoff")
        if role not in ("cutoff", "clarification"):
            raise InputError(f"{s['file']}: role must be cutoff or clarification")
        if s["kind"] == "csv":
            t = when(s["received"])
            for a in read_csv(f):
                out.append((s["reviewer"], a.get("qa_id", ""), a, t, s["file"], role, "csv"))
        elif s["kind"] == "store":
            docs = read_store(f)
            unknown = sorted(set(docs) - set(s["docs"]))
            if unknown:
                raise InputError(f"{s['file']}: documents not in identities.json: {unknown}")
            for did, doc in docs.items():
                for k, v in doc["answers"].items():
                    if not isinstance(v, dict) or not v.get("t"):
                        raise InputError(f"{s['file']}: {did} {k} has no answer time")
                    a = norm(v)
                    if v.get("restored"):          # restored by its reviewer from an old shared browser draft (P158)
                        a["restored"] = str(v["restored"])
                    out.append((s["docs"][did], str(v.get("qa_id", "")), a, when(v["t"]), s["file"], role, "store"))
        else:
            raise InputError(f"{s['file']}: kind must be store or csv")
    seen: dict = {}
    for rev, q, a, _t, src, _role, _kind in out:            # one answer per reviewer and question in one snapshot
        if seen.setdefault((src, rev, q), answer_of(a)) != answer_of(a):
            raise InputError(f"{src}: two different answers to {q} from one reviewer in one snapshot")
    return out, hashes


def check_values(a: dict, qa_id: str, sample: set, where: str) -> None:
    if qa_id not in sample:
        raise InputError(f"{where}: question not in the review sample: {qa_id!r}")
    for f, ok in VOCAB.items():
        if a.get(f) and a[f] not in ok:
            raise InputError(f"{where} {qa_id}: {f} = {a[f]!r} is not one of {ok}")


def complete(a: dict) -> bool:
    return all(a.get(f) for f in VOCAB) and (a["verdict"] == "accept" or bool(a.get("note")))


def contradictory(a: dict) -> bool:
    v, r = a["verdict"], a["reason"]
    if v == "accept":
        return r != "OK" or any(a[c] == "n" for c in CHECKS)
    return r == "OK" or (r in REASON_CHECK and a[REASON_CHECK[r]] == "y")


def pick(cands: list, rev: str, q: str, resolutions: dict) -> tuple:
    """One final answer from a reviewer's candidate records for one question (see the module rules)."""
    distinct = {answer_of(c[2]) for c in cands}
    if len(distinct) == 1:
        return max(cands, key=lambda c: c[3])
    chosen = resolutions.get(rev, {}).get(q)
    if chosen:
        mine = [c for c in cands if c[4] == chosen]
        if not mine or len({answer_of(c[2]) for c in mine}) != 1:
            raise InputError(f"resolution for {q} names {chosen}, which holds no single answer for it")
        return mine[0]
    if all(c[6] == "store" for c in cands):
        latest = max(c[3] for c in cands)
        top = {answer_of(c[2]) for c in cands if c[3] == latest}
        if len(top) == 1:
            return next(c for c in cands if c[3] == latest)
    raise InputError(f"conflicting answers for one reviewer and question {q} in {sorted({c[4] for c in cands})}; "
                     "name the confirmed source in identities.json resolutions")


def collect(ident: dict, base: Path, sample: set) -> dict:
    """Final answers per reviewer and question, with the reasons any answer was not used."""
    recs, hashes = records(ident, base)
    for rev, q, a, _t, src, _role, _kind in recs:
        check_values(a, q, sample, src)
    res = ident.get("resolutions", {})
    late, cut, clar = collections.Counter(), collections.defaultdict(list), collections.defaultdict(list)
    for r in recs:
        rev, q, _a, t, _src, role, _kind = r
        if role == "cutoff":
            if t <= CUTOFF:
                cut[(rev, q)].append(r)
            else:
                late[rev] += 1
        elif CUTOFF < t <= CLARIFICATION_CUTOFF:
            clar[(rev, q)].append(r)
    final, status = {}, {}
    for key, cands in cut.items():
        a = pick(cands, *key, res)[2]
        if not complete(a):
            status[key] = "incomplete"
            continue
        if contradictory(a):
            if key in clar:
                b = pick(clar[key], *key, res)[2]
                if complete(b) and not contradictory(b):
                    final[key], status[key] = b, "clarified"
                    continue
            status[key] = "contradictory, unresolved"
            continue
        final[key], status[key] = a, "valid"
    unused = sum(len(v) for k, v in clar.items() if status.get(k) != "clarified")
    return {"final": final, "status": status, "late": dict(late), "clarification_records_unused": unused,
            "hashes": hashes}


def combine(votes: list) -> tuple:
    """(verdict, no_majority): the strict-majority verdict of the valid votes, else uncertain."""
    c = collections.Counter(votes).most_common()
    if c[0][1] * 2 > len(votes):
        return c[0][0], False
    return "uncertain", True


def kappa(a: list, b: list) -> float | None:
    if not a:
        return None
    po = np.mean([x == y for x, y in zip(a, b, strict=True)])
    pe = sum((a.count(v) / len(a)) * (b.count(v) / len(b)) for v in VERDICTS)
    return None if pe == 1 else round(float((po - pe) / (1 - pe)), 3)


def agreement(a: dict, b: dict) -> dict:
    qs = sorted(set(a) & set(b))
    x, y = [a[q] for q in qs], [b[q] for q in qs]
    return {"n": len(qs), "agree": int(sum(u == v for u, v in zip(x, y, strict=True))),
            "percent": round(100 * float(np.mean([u == v for u, v in zip(x, y, strict=True)])), 1) if qs else None,
            "kappa": kappa(x, y),
            "table": {u: {v: int(sum(1 for p, s in zip(x, y, strict=True) if p == u and s == v)) for v in VERDICTS}
                      for u in VERDICTS}}


def boot_rate(r: dict, w: dict) -> tuple:
    ls = sorted(r)
    rv, wv = np.array([r[x] for x in ls], float), np.array([w[x] for x in ls], float)
    idx = np.random.default_rng(SEED).integers(0, len(ls), (B, len(ls)))
    boots = (rv[idx] * wv[idx]).sum(1) / wv[idx].sum(1)
    return (round(100 * float((rv * wv).sum() / wv.sum()), 1), round(100 * float(np.percentile(boots, 2.5)), 1),
            round(100 * float(np.percentile(boots, 97.5)), 1))


def rates(flag: dict, lecture: dict, n_l: dict) -> dict:
    r = {lecture[q]: float(v) for q, v in flag.items()}
    return {"lecture_balanced": boot_rate(r, {x: 1.0 for x in r}),
            "question_weighted": boot_rate(r, {x: n_l[x] for x in r}), "count": int(sum(flag.values())),
            "of": len(flag)}


def cohort_inputs() -> dict:
    """The frozen inputs, each checked fail-closed."""
    from m3av_hits import SYSTEMS  # noqa: PLC0415
    from m3av_qa import CONDITIONS  # noqa: PLC0415

    verified = verify_expected(EXPECTED)
    sample_doc = json.loads(Path("reports/m3av/review_sample_test.json").read_text(encoding="utf-8"))
    sample = sample_doc["qa_ids"]
    hits = json.loads(Path("reports/m3av/hits_test.json").read_text(encoding="utf-8"))
    prim, ssrc = hits["per_question"], hits["s_src"]["per_question"]
    if set(prim) != set(SYSTEMS) or set(ssrc) != set(SYSTEMS):
        raise InputError("the hit report's systems are not the frozen roster")
    cohort = sorted(next(iter(prim.values())))
    if len(cohort) != 853 or any(sorted(v) != cohort for d in (prim, ssrc) for v in d.values()):
        raise InputError("the hit report does not cover the 853-question cohort for every system")
    lecture = {q: q.split(":")[1] for q in cohort}
    if len(sample) != 100 or len(set(sample)) != 100 or len({lecture.get(q) for q in sample}) != 100 \
            or set(sample) - set(cohort):
        raise InputError("the review sample is not 100 distinct cohort questions from 100 lectures")
    qa = collections.defaultdict(dict)
    for line in Path("runs/m3av/qa/test/records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["qa_id"] in qa[r["condition"]]:
                raise InputError(f"repeated QA record: {r['condition']} {r['qa_id']}")
            qa[r["condition"]][r["qa_id"]] = int(r["correct"])
    if set(qa) != set(CONDITIONS) or any(sorted(v) != cohort for v in qa.values()):
        raise InputError("the QA records do not cover every condition and question exactly once")
    model = {k: read_model(f, sample) for k, f in MODEL_REVIEWS.items()}
    order = sorted(sample, key=lambda q: hashlib.sha256(f"human-review:{q}".encode()).hexdigest())
    files = [Path("docs/v3/m3av_qa_test_outputs.json"), Path("scripts/m3av_human_review.py")]
    return {"sample": sample, "cohort": cohort, "lecture": lecture, "prim": prim, "ssrc": ssrc, "qa": dict(qa),
            "model": model, "page_item": {q: i + 1 for i, q in enumerate(order)},
            "codes": {s: s.split()[0] for s in SYSTEMS},
            "hashes": {**verified, **{p.as_posix(): sha(p) for p in files}}}


def analyse(c: dict, got: dict) -> dict:  # noqa: PLR0915
    from m3av_hits import PRIMARY, paired  # noqa: PLC0415

    sample, cohort, lecture = c["sample"], c["cohort"], c["lecture"]
    n_l = collections.Counter(lecture.values())
    revs = sorted({k[0] for k in got["status"]} | set(got["late"]))
    alias = {r: f"H{i}" for i, r in enumerate(revs, 1)}
    votes = collections.defaultdict(dict)
    for (rev, q), a in got["final"].items():
        votes[q][alias[rev]] = a["verdict"]
    combined, no_major = {}, []
    for q in sample:
        if votes.get(q):
            combined[q], nm = combine(list(votes[q].values()))
            if nm:
                no_major.append(q)
    reviewed = [q for q in sample if q in combined]
    missing = [q for q in sample if q not in combined]
    full = not missing
    st = collections.Counter(got["status"].values())
    rep = {"what": "blind human review of the 100 test packets (protocol 'Question review'; research log steps 98-99)",
           "rules": {"cutoff": CUTOFF.isoformat(), "clarification_cutoff": CLARIFICATION_CUTOFF.isoformat(),
                     "combination": "strict majority of valid votes, else uncertain"},
           "input_hashes": {**c["hashes"], **got["hashes"]},
           "reviewers": {alias[r]: {"valid": sum(1 for k, s in got["status"].items() if k[0] == r and s in
                                                 ("valid", "clarified")),
                                    **{s: sum(1 for k, x in got["status"].items() if k[0] == r and x == s)
                                       for s in ("incomplete", "contradictory, unresolved", "clarified")},
                                    "late": got["late"].get(r, 0),
                                    "restored_from_browser_draft": sum(1 for (x, _q), a in got["final"].items()
                                                                       if x == r and a.get("restored"))} for r in revs},
           "answer_status": dict(st), "clarification_records_unused": got["clarification_records_unused"],
           "coverage": {"items_reviewed": len(reviewed), "items_unreviewed": len(missing),
                        "votes_per_item": dict(collections.Counter(len(votes.get(q, {})) for q in sample)),
                        "complete": full, "unreviewed": [{"qa_id": q, "page_item": c["page_item"][q],
                                                          "blocks": [b for b, (x, y) in BLOCKS.items()
                                                                     if x <= c["page_item"][q] <= y]}
                                                         for q in missing]},
           "combined_counts": {v: sum(combined[q] == v for q in reviewed) for v in VERDICTS},
           "no_majority": len(no_major)}
    if not reviewed:
        raise InputError("no valid human vote for any sampled question")
    label = ("planned estimate (all 100 sampled items reviewed)" if full else
             f"observed reviews only: {len(reviewed)} of 100 items, conditional on the responders; not the planned "
             "cohort estimate")
    rep["rejection_rate"] = {"label": label, **rates({q: combined[q] == "reject" for q in reviewed}, lecture, n_l)}
    rep["reject_or_uncertain_rate"] = {"label": label,
                                       **rates({q: combined[q] != "accept" for q in reviewed}, lecture, n_l)}
    counts = {f: collections.Counter(a[f] for a in got["final"].values()) for f in (*CHECKS, "shortcut", "reason")}
    rep["counts_over_valid_human_answers"] = {"answers": len(got["final"]), **{f: dict(v) for f, v in counts.items()}}
    rep["agreement"] = {f"human vs {k}": agreement(combined, m) for k, m in c["model"].items()}
    rep["agreement"]["builder vs auditor"] = agreement(c["model"]["builder"], c["model"]["auditor"])
    hs = sorted(alias.values(), key=lambda h: int(h[1:]))
    for i, x in enumerate(hs):
        for y in hs[i + 1:]:
            g = agreement({q: v[x] for q, v in votes.items() if x in v}, {q: v[y] for q, v in votes.items() if y in v})
            if g["n"]:
                rep["agreement"][f"{x} vs {y}"] = g
    masks = {"secondary 1: without human-rejected": {q for q in reviewed if combined[q] == "reject"},
             "conservative: without human-rejected or uncertain": {q for q in reviewed if combined[q] != "accept"}}
    prim, ssrc, qa, codes = c["prim"], c["ssrc"], c["qa"], c["codes"]

    def scores(keep):
        sub = {"questions": len(keep), "lectures": len({lecture[q] for q in keep}),
               "hit@4": {codes[s]: round(100 * float(np.mean([prim[s][q] for q in keep])), 1) for s in prim},
               "s_src_hit@4": {codes[s]: round(100 * float(np.mean([ssrc[s][q] for q in keep])), 1) for s in ssrc},
               "qa_accuracy": {k: round(100 * float(np.mean([v[q] for q in keep])), 1) for k, v in qa.items()},
               "contrasts": {}}
        for name, (x, y) in PRIMARY.items():
            d = paired({q: prim[x][q] for q in keep}, {q: prim[y][q] for q in keep}, lecture)
            s = paired({q: ssrc[x][q] for q in keep}, {q: ssrc[y][q] for q in keep}, lecture)
            sub["contrasts"][name] = {"hit@4": {k: d[k] for k in ("diff", "lo", "hi")},
                                      "s_src": {k: s[k] for k in ("diff", "lo", "hi")}}
        return sub

    rep["full_cohort"] = scores(cohort)
    rep["masks"] = {}
    for mk, drop in masks.items():
        sub = scores([q for q in cohort if q not in drop])
        kept_l = {lecture[q] for q in cohort if q not in drop}
        sub.update({"removed_questions": len(drop), "removed_lectures": len({lecture[q] for q in drop} - kept_l),
                    "dropped": [{"qa_id": q, "combined": combined[q], "votes": votes[q],
                                 "reasons": sorted({got["final"][(r, q)]["reason"] for r in revs if (r, q) in got["final"]})}
                                for q in sorted(drop)],
                    "per_lecture_attrition": dict(collections.Counter(lecture[q] for q in drop)),
                    "note": "uncertain and unreviewed questions kept" if mk.startswith("secondary") else
                    "unreviewed questions kept"})
        rep["masks"][mk] = sub
    acc = [q for q in reviewed if combined[q] == "accept"]
    rep["accepted_sample_only"] = {"label": "descriptive; the accepted sampled cases, not a cohort estimate",
                                   "questions": len(acc), "lectures": len({lecture[q] for q in acc}),
                                   "hit@4": {codes[s]: round(100 * float(np.mean([prim[s][q] for q in acc])), 1)
                                             if acc else None for s in prim}}
    rep["per_item"] = {q: {"page_item": c["page_item"][q], "combined": combined.get(q, "unreviewed"),
                           "votes": votes.get(q, {}), "no_majority": q in no_major,
                           "builder": c["model"]["builder"][q], "auditor": c["model"]["auditor"][q]} for q in sample}
    return rep


def markdown(rep: dict) -> str:
    def ci(t):
        return f"{t[0]} ({t[1]}..{t[2]})"

    cov = rep["coverage"]
    md = ["# Lecture study: blind human review of the 100 test packets", "",
          "Analyses declared in the protocol (Question review, amendment 3) and fixed in research log steps 98-99 before "
          "any human verdict was seen. The cohort's labels are otherwise automatic. Sensitivity results are descriptive "
          "and never replace the primary result. Intervals cover sampling only, not annotator error or non-response.", "",
          f"Reviewers {len(rep['reviewers'])}: " + "; ".join(
              f"{k} {v['valid']} valid, {v['incomplete']} incomplete, {v['contradictory, unresolved']} unresolved, "
              f"{v['late']} late" for k, v in rep["reviewers"].items()) + ".",
          f"Items reviewed {cov['items_reviewed']} of 100 (votes per item: {cov['votes_per_item']}); "
          f"no strict majority: {rep['no_majority']}."]
    if cov["unreviewed"]:
        md.append("Unreviewed items (page number): " + ", ".join(str(u["page_item"]) for u in cov["unreviewed"]) + ".")
    md += ["", "| Combined human verdict | accept | reject | uncertain |", "|---|---|---|---|",
           "| items | " + " | ".join(str(rep["combined_counts"][v]) for v in VERDICTS) + " |", "",
           f"Rates: {rep['rejection_rate']['label']}.", "",
           "| Rate, percent (95% lecture-bootstrap interval) | lecture-balanced | cohort question-weighted |",
           "|---|---|---|",
           f"| rejected | {ci(rep['rejection_rate']['lecture_balanced'])} | {ci(rep['rejection_rate']['question_weighted'])} |",
           f"| rejected or uncertain | {ci(rep['reject_or_uncertain_rate']['lecture_balanced'])} | "
           f"{ci(rep['reject_or_uncertain_rate']['question_weighted'])} |", "",
           "## Agreement", "", "| Pair | items | agree | percent | Cohen's kappa |", "|---|---|---|---|---|"]
    for k, g in rep["agreement"].items():
        md.append(f"| {k} | {g['n']} | {g['agree']} | {g['percent']} | {g['kappa']} |")
    cnt = rep["counts_over_valid_human_answers"]
    md += ["", f"## Counts over the {cnt['answers']} valid human answers (reviewers can overlap)", ""]
    for k, v in cnt.items():
        if k != "answers":
            md.append(f"- {k}: " + ", ".join(f"{x} {n}" for x, n in sorted(v.items())))
    md += ["", "## Hit@4 with the human-rejected sample questions removed (descriptive)", "",
           "| Cohort | questions | S1 | S2m | R5 | L1 S2m − R5 (95% interval) | L2 S2m − S1 (95% interval) |",
           "|---|---|---|---|---|---|---|"]
    rows = {"full cohort (primary)": rep["full_cohort"], **rep["masks"]}

    def d(x):
        return f"{x['diff']:+.1f} ({x['lo']:+.1f}..{x['hi']:+.1f})"

    for mk, sub in rows.items():
        h, c = sub["hit@4"], sub["contrasts"]
        md.append(f"| {mk} | {sub['questions']} | {h['S1']} | {h['S2m']} | {h['R5']} | {d(c['L1']['hit@4'])} | "
                  f"{d(c['L2']['hit@4'])} |")
    md += ["", "S-src hit@4:", "", "| Cohort | S1 | S2m | R5 | L1 | L2 |", "|---|---|---|---|---|---|"]
    for mk, sub in rows.items():
        h, c = sub["s_src_hit@4"], sub["contrasts"]
        md.append(f"| {mk} | {h['S1']} | {h['S2m']} | {h['R5']} | {d(c['L1']['s_src'])} | {d(c['L2']['s_src'])} |")
    md += ["", "Answer-model accuracy (descriptive):", "", "| Cohort | S1 | S2m | R5 | q_only | t_only |",
           "|---|---|---|---|---|---|"]
    for mk, sub in rows.items():
        q = sub["qa_accuracy"]
        md.append(f"| {mk} | {q['S1']} | {q['S2m']} | {q['R5']} | {q['q_only']} | {q['t_only']} |")
    s = rep["accepted_sample_only"]
    md += ["", f"Accepted sample questions alone ({s['questions']} questions, {s['lectures']} lectures; {s['label']}): "
           "hit@4 " + ", ".join(f"{k} {v}" for k, v in s["hit@4"].items()) + ".",
           "", "All systems and conditions, dropped ids and reasons, and per-lecture attrition: human_review_test.json."]
    return "\n".join(md) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("runs/m3av/review_human"), help="raw/ and identities.json")
    p.add_argument("--out", type=Path, default=Path("reports/m3av"), help="folder for human_review_test.{json,md}")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    c = cohort_inputs()
    ident = json.loads((a.dir / "identities.json").read_text(encoding="utf-8"))
    got = collect(ident, a.dir, set(c["sample"]))
    rep = analyse(c, got)
    rep["input_hashes"]["identities.json (private)"] = sha(a.dir / "identities.json")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "human_review_test.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = markdown(rep)
    (a.out / "human_review_test.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
