"""Synthetic checks of the human-review rules (scripts/m3av_human_review.py; auditor P143-P146, P157). No real answers."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "m3av_human_review", Path(__file__).resolve().parents[1] / "scripts/m3av_human_review.py")
hr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr)

SAMPLE = [f"m3av:L{i:03d}:1" for i in range(1, 6)]
OK = dict(verdict="accept", key="y", one="y", slide="y", legible="y", shortcut="none", reason="OK", note="")
REJ = dict(verdict="reject", key="y", one="y", slide="y", legible="n", shortcut="none", reason="SPAN", note="off")
T0, T1 = "2026-11-08T10:00:00+06:00", "2026-11-08T12:00:00+06:00"
LATE, CLAR, TOO_LATE = "2026-11-09T08:00:00+06:00", "2026-11-09T20:00:00+06:00", "2026-11-10T08:00:00+06:00"


def raw(tmp):
    (tmp / "raw").mkdir(exist_ok=True)
    return tmp / "raw"


def store(tmp, name, docs):
    """docs: {doc id: {qa_id: (answer, t)}} in the page's shape."""
    body = [{"id": d, "data": {"name": "x", "answers": {f"p{i:03d}": dict(a, qa_id=q, item=i, t=t)
                                                        for i, (q, (a, t)) in enumerate(ans.items(), 1)}}}
            for d, ans in docs.items()]
    (raw(tmp) / name).write_text(json.dumps(body), encoding="utf-8")


def csvfile(tmp, name, rows):
    lines = ['reviewer,"x"', "item,qa_id,verdict,key,one,slide,legible,shortcut,reason,note"]
    lines += [",".join([str(i), q, *(a[f] for f in hr.FIELDS)]) for i, (q, a) in enumerate(rows, 1)]
    (raw(tmp) / name).write_text("\n".join(lines), encoding="utf-8")


def run(tmp, sources, resolutions=None):
    return hr.collect({"sources": sources, "resolutions": resolutions or {}}, tmp, set(SAMPLE))


def src_store(name, docs, role="cutoff"):
    return {"file": f"raw/{name}", "kind": "store", "role": role, "docs": docs}


def src_csv(name, rev, received=T1):
    return {"file": f"raw/{name}", "kind": "csv", "role": "cutoff", "reviewer": rev, "received": received}


def test_strict_majority():
    assert hr.combine(["accept"]) == ("accept", False)
    assert hr.combine(["accept", "accept", "reject"]) == ("accept", False)
    assert hr.combine(["accept", "accept", "reject", "uncertain"]) == ("uncertain", True)   # plurality is not enough
    assert hr.combine(["accept", "reject"]) == ("uncertain", True)
    assert hr.combine(["reject", "reject"]) == ("reject", False)


def test_identical_exports_count_once(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (OK, T0)}})
    csvfile(tmp_path, "a.csv", [(SAMPLE[0], OK)])
    got = run(tmp_path, [src_store("s.json", {"people/a": "r1"}), src_csv("a.csv", "r1")])
    assert list(got["final"]) == [("r1", SAMPLE[0])]


def test_store_revision_latest_wins(tmp_path):
    store(tmp_path, "s1.json", {"people/a": {SAMPLE[0]: (OK, T0)}})
    store(tmp_path, "s2.json", {"people/a": {SAMPLE[0]: (REJ, T1)}})
    got = run(tmp_path, [src_store("s1.json", {"people/a": "r1"}), src_store("s2.json", {"people/a": "r1"})])
    assert got["final"][("r1", SAMPLE[0])]["verdict"] == "reject"


def test_csv_store_conflict_stops_unless_resolved(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (OK, T0)}})
    csvfile(tmp_path, "a.csv", [(SAMPLE[0], REJ)])
    srcs = [src_store("s.json", {"people/a": "r1"}), src_csv("a.csv", "r1")]
    with pytest.raises(hr.InputError):
        run(tmp_path, srcs)
    got = run(tmp_path, srcs, {"r1": {SAMPLE[0]: "raw/a.csv"}})
    assert got["final"][("r1", SAMPLE[0])]["verdict"] == "reject"


def test_conflicting_rows_in_one_file_stop(tmp_path):
    csvfile(tmp_path, "a.csv", [(SAMPLE[0], OK), (SAMPLE[0], REJ)])
    with pytest.raises(hr.InputError):
        run(tmp_path, [src_csv("a.csv", "r1")])


def test_conflict_inside_one_store_snapshot_stops(tmp_path):
    body = [{"id": "people/a", "data": {"answers": {"p001": dict(OK, qa_id=SAMPLE[0], t=T0),
                                                    "p002": dict(REJ, qa_id=SAMPLE[0], t=T1)}}}]
    (raw(tmp_path) / "s.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(hr.InputError):
        run(tmp_path, [src_store("s.json", {"people/a": "r1"})])


def test_invalid_values_and_ids_stop(tmp_path):
    csvfile(tmp_path, "a.csv", [(SAMPLE[0], dict(OK, reason="GARBAGE"))])
    with pytest.raises(hr.InputError):
        run(tmp_path, [src_csv("a.csv", "r1")])
    csvfile(tmp_path, "a.csv", [("m3av:NOPE:1", OK)])
    with pytest.raises(hr.InputError):
        run(tmp_path, [src_csv("a.csv", "r1")])


def test_unlisted_document_stops(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (OK, T0)}, "people/b": {SAMPLE[1]: (OK, T0)}})
    with pytest.raises(hr.InputError):
        run(tmp_path, [src_store("s.json", {"people/a": "r1"})])


def test_raw_inventory_must_match(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (OK, T0)}})
    csvfile(tmp_path, "forgotten.csv", [(SAMPLE[1], OK)])
    with pytest.raises(hr.InputError):                        # a raw export not listed
        run(tmp_path, [src_store("s.json", {"people/a": "r1"})])
    with pytest.raises(hr.InputError):                        # a listed export that does not exist
        run(tmp_path, [src_store("s.json", {"people/a": "r1"}), src_csv("forgotten.csv", "r2"),
                       src_csv("gone.csv", "r3")])


def test_output_labels_are_pseudonymous(tmp_path):
    csvfile(tmp_path, "alice_smith.csv", [(SAMPLE[0], OK)])
    got = run(tmp_path, [src_csv("alice_smith.csv", "r1")])
    assert not any("alice" in k for k in got["hashes"])


def test_incomplete_and_late(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (dict(REJ, note=""), T0), SAMPLE[1]: (OK, LATE)}})
    got = run(tmp_path, [src_store("s.json", {"people/a": "r1"})])
    assert got["status"][("r1", SAMPLE[0])] == "incomplete"
    assert got["late"] == {"r1": 1} and not got["final"]


def test_contradiction_needs_clarification(tmp_path):
    bad = dict(OK, legible="n")                               # accept although the fact is not legible
    assert hr.contradictory(bad) and hr.contradictory(dict(REJ, reason="OK"))
    assert hr.contradictory(dict(REJ, reason="KEY")) and not hr.contradictory(REJ)
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (bad, T0), SAMPLE[1]: (bad, T0)}})
    store(tmp_path, "c.json", {"people/a": {SAMPLE[0]: (REJ, CLAR), SAMPLE[1]: (REJ, TOO_LATE), SAMPLE[2]: (OK, CLAR)}})
    got = run(tmp_path, [src_store("s.json", {"people/a": "r1"}),
                         src_store("c.json", {"people/a": "r1"}, "clarification")])
    assert got["status"][("r1", SAMPLE[0])] == "clarified"
    assert got["status"][("r1", SAMPLE[1])] == "contradictory, unresolved"
    assert ("r1", SAMPLE[2]) not in got["final"]             # clarification sources only answer flagged items
    assert got["clarification_records_unused"] == 1


def test_model_review_must_be_one_valid_verdict_each(tmp_path):
    head = "n,qa_id,verdict (accept/reject/uncertain),key correct (y/n)\n"
    (tmp_path / "ok.csv").write_text(head + "".join(f"{i},{q},accept,y\n" for i, q in enumerate(SAMPLE, 1)),
                                     encoding="utf-8")
    assert set(hr.read_model(tmp_path / "ok.csv", SAMPLE).values()) == {"accept"}
    (tmp_path / "dup.csv").write_text(head + f"1,{SAMPLE[0]},accept,y\n" * 2, encoding="utf-8")
    (tmp_path / "bad.csv").write_text(head + "".join(f"{i},{q},maybe,y\n" for i, q in enumerate(SAMPLE, 1)),
                                      encoding="utf-8")
    for name in ("dup.csv", "bad.csv"):
        with pytest.raises(hr.InputError):
            hr.read_model(tmp_path / name, SAMPLE)


def test_frozen_hash_mismatch_stops(tmp_path):
    (tmp_path / "a.jsonl").write_bytes(b"x\r\n")
    lf = hashlib.sha256(b"x\n").hexdigest()
    assert hr.verify_expected({"a.jsonl": ("lf", lf)}, tmp_path)
    with pytest.raises(hr.InputError):
        hr.verify_expected({"a.jsonl": ("bytes", lf)}, tmp_path)
    with pytest.raises(hr.InputError):
        hr.verify_expected({"missing.json": ("bytes", lf)}, tmp_path)


def fake_cohort():
    from m3av_hits import SYSTEMS  # noqa: PLC0415  (importable once the script module is loaded)

    cohort = sorted(SAMPLE + [f"m3av:L{i:03d}:2" for i in range(1, 6)])
    prim = {s: {q: (i + j) % 2 for j, q in enumerate(cohort)} for i, s in enumerate(SYSTEMS)}
    return {"sample": SAMPLE, "cohort": cohort, "lecture": {q: q.split(":")[1] for q in cohort}, "prim": prim,
            "ssrc": prim, "qa": {c: {q: 1 for q in cohort} for c in ("S1", "S2m", "R5", "q_only", "t_only")},
            "codes": {s: s.split()[0] for s in SYSTEMS},
            "model": {"builder": {q: "accept" for q in SAMPLE}, "auditor": {q: "accept" for q in SAMPLE}},
            "page_item": {q: i + 1 for i, q in enumerate(SAMPLE)}, "hashes": {}}


def test_missing_items_and_no_accepts(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (REJ, T0), SAMPLE[1]: (REJ, T0)}})
    got = run(tmp_path, [src_store("s.json", {"people/a": "r1"})])
    rep = hr.analyse(fake_cohort(), got)
    assert rep["coverage"]["items_unreviewed"] == 3 and not rep["coverage"]["complete"]
    assert rep["rejection_rate"]["label"].startswith("observed reviews only")
    assert rep["rejection_rate"]["count"] == 2 and rep["rejection_rate"]["of"] == 2
    m = rep["masks"]["secondary 1: without human-rejected"]
    assert m["questions"] == 8 and m["removed_questions"] == 2 and m["removed_lectures"] == 0
    assert rep["accepted_sample_only"]["questions"] == 0 and set(rep["accepted_sample_only"]["hit@4"].values()) == {None}
    assert rep["per_item"][SAMPLE[4]]["combined"] == "unreviewed"
    hr.markdown(rep)


def test_no_valid_vote_stops(tmp_path):
    store(tmp_path, "s.json", {"people/a": {SAMPLE[0]: (dict(REJ, note=""), T0)}})
    with pytest.raises(hr.InputError):
        hr.analyse(fake_cohort(), run(tmp_path, [src_store("s.json", {"people/a": "r1"})]))
