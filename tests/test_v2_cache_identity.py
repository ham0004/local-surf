"""Cache schema v2: every part of the rendered request is in the identity; correctness follows current gold."""
import copy
import json

import numpy as np
from PIL import Image

from videoqa.schemas import Answer, TranscriptSegment
from videoqa.v2.records import FrameCandidate, QuestionPool
from videoqa.v2.teacher import KEY_SCHEMA, CachedTeacher, evidence_key, request_key

IDENTITY = {"model_id": "fake", "max_new_tokens": 64, "decoding": "greedy"}


def _pool():
    cands = [FrameCandidate(f"q:f{i:02d}", 10.0 * i, ("B",), f"d{i}", "0" * 16, image=Image.new("RGB", (4, 4)),
                            emb=np.ones(2)) for i in range(3)]
    return QuestionPool("q", "v", "what value?", ["a", "b"], 1,
                        [TranscriptSegment("s00001", 0.0, 5.0, "some speech")], cands)


class Fake:
    def __init__(self):
        self.n = 0

    def answer(self, req):
        self.n += 1
        return Answer(text="B", option_index=1), None


def test_key_changes_with_frame_time_label_transcript_metadata_and_decoding():
    p = _pool()
    base = request_key(p, p.candidates[:1], IDENTITY)
    moved = copy.deepcopy(p)
    moved.candidates[0].time_s = 11.0                      # same pixels, different "Frame at" label
    assert request_key(moved, moved.candidates[:1], IDENTITY) != base
    for field, value in (("id", "s00009"), ("end_s", 6.0)):
        t = copy.deepcopy(p)
        setattr(t.transcript[0], field, value)              # rendered as [id start-end] in the prompt
        assert request_key(t, t.candidates[:1], IDENTITY) != base
    assert request_key(p, p.candidates[:1], {**IDENTITY, "max_new_tokens": 128}) != base
    # the legacy key was blind to all of these (the defect being fixed)
    assert evidence_key(moved, moved.candidates[:1], "fake") == evidence_key(p, p.candidates[:1], "fake")


def test_correctness_is_recomputed_from_the_cached_prediction_when_gold_changes(tmp_path):
    p, fake = _pool(), Fake()
    t = CachedTeacher(fake, "fake", tmp_path / "c.jsonl", identity=IDENTITY)
    assert t.quality(p, p.candidates[:1]) == 1.0
    p.gold_option_index = 0
    assert t.quality(p, p.candidates[:1]) == 0.0 and fake.n == 1
    rec = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])
    assert rec["schema"] == KEY_SCHEMA and "quality" not in rec and rec["option"] == 1


def test_legacy_records_are_reused_only_for_the_same_question_and_frame_ids(tmp_path):
    p = _pool()
    legacy = tmp_path / "old.jsonl"
    rec = {"key": evidence_key(p, p.candidates[:1], "fake"), "qa_id": "q", "frames": ["q:f00"],
           "quality": 1.0, "option": 1, "text": "B"}
    legacy.write_text(json.dumps(rec) + "\n")
    fake = Fake()
    t = CachedTeacher(fake, "fake", tmp_path / "new.jsonl", identity=IDENTITY, legacy_paths=(legacy,))
    assert t.quality(p, p.candidates[:1]) == 1.0 and fake.n == 0 and t.legacy_hits == 1
    # same pixels and v1 key, but a different candidate id (e.g. another pool): not trusted, re-asked
    other = copy.deepcopy(p)
    other.candidates[0].id = "q:f07"
    t.quality(other, other.candidates[:1])
    assert fake.n == 1
    assert not legacy.read_text().count(KEY_SCHEMA)        # the legacy file is never written


def test_cache_only_reader_raises_instead_of_guessing(tmp_path):
    p = _pool()
    t = CachedTeacher(None, "fake", tmp_path / "c.jsonl", identity=IDENTITY)
    try:
        t.quality(p, p.candidates[:1])
    except LookupError:
        return
    raise AssertionError("an uncached request must not return a value")


def test_call_budget_enforces_both_limits_and_persists(tmp_path):
    import pytest

    from videoqa.v2.budget import BudgetExceeded, CallBudget

    p, fake = _pool(), Fake()
    b = CallBudget(tmp_path / "b.json", max_calls=2, max_seconds=3600)
    t = CachedTeacher(fake, "fake", tmp_path / "c.jsonl", identity=IDENTITY, budget=b)
    assert b.reserve(2) and not b.reserve(3)
    t.quality(p, p.candidates[:1])
    t.quality(p, p.candidates[:1])                          # cache hit: not charged
    t.quality(p, p.candidates[1:2])
    assert b.calls == 2 and fake.n == 2
    with pytest.raises(BudgetExceeded):
        t.quality(p, p.candidates[2:3])
    assert CallBudget(tmp_path / "b.json", 2, 3600).calls == 2      # persisted across processes
    with pytest.raises(ValueError, match="different limits"):
        CallBudget(tmp_path / "b.json", 5, 3600)                     # raising a limit needs a new ledger
    tb = CallBudget(tmp_path / "t.json", max_calls=100, max_seconds=1e-9)
    tb.charge(1.0)
    assert not tb.reserve(1)                                         # time limit alone also stops the run
