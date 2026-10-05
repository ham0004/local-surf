"""CPU tests for v3 open-ended answering and judging (no models loaded)."""
import json

import pytest

from videoqa.schemas import Answer, TranscriptSegment
from videoqa.v2.budget import BudgetExceeded, CallBudget
from videoqa.v3.judge import parse_verdict, quality, verdict_key
from videoqa.v3.openended import AnswerCache, build_open_prompt, request_key

ID = {"model_id": "fake", "max_new_tokens": 256}


class F:
    def __init__(self, t, d):
        self.decoded_pts_s, self.digest = t, d


class FakeAnswerer:
    def __init__(self):
        self.n = 0

    def answer(self, req):
        self.n += 1

        class U:
            visual_tokens, generated_tokens = 220 * len(req.frames), 12
        return Answer(text=f"answer with {len(req.frames)} frames", option_index=None), U()


SEG = [TranscriptSegment("s00001", 10.0, 14.0, "a pointer stores an address")]


def test_open_prompt_has_question_and_timestamped_transcript_but_no_options():
    p = build_open_prompt("What does %s do?", SEG)
    assert "QUESTION: What does %s do?" in p and "[s00001 10.0-14.0s]" in p and "OPTIONS" not in p


def test_request_key_covers_frame_time_label_and_transcript_metadata():
    base = request_key("q", SEG, [F(1.0, "d")], ID)
    assert request_key("q", SEG, [F(2.0, "d")], ID) != base                        # same pixels, new time label
    assert request_key("q", [TranscriptSegment("s00002", 10.0, 14.0, SEG[0].text)], [F(1.0, "d")], ID) != base
    assert request_key("q", SEG, [F(1.0, "d")], {**ID, "max_new_tokens": 64}) != base
    assert request_key("q", SEG, [F(1.0, "d"), F(3.0, "e")], ID) == request_key("q", SEG, [F(3.0, "e"), F(1.0, "d")], ID)


def test_answer_cache_reuses_and_charges_budget(tmp_path):
    fake = FakeAnswerer()
    b = CallBudget(tmp_path / "b.json", max_calls=1, max_seconds=60)
    c = AnswerCache(tmp_path / "a.jsonl", ID, fake, b)
    c.answer("q1", "q", SEG, [F(1.0, "d")])
    c.answer("q1", "q", SEG, [F(1.0, "d")])
    assert fake.n == 1 and c.hits == 1 and b.calls == 1
    with pytest.raises(BudgetExceeded):
        c.answer("q1", "q", SEG, [])
    assert AnswerCache(tmp_path / "a.jsonl", ID).get("q", SEG, [F(1.0, "d")])["text"] == "answer with 1 frames"


def test_parse_verdict_snaps_and_rejects_malformed():
    assert parse_verdict('ok {"consistency": 2, "coverage": 60} done') == {"consistency": 2, "coverage": 50,
                                                                          "quality": 0.5}
    assert parse_verdict('{"consistency": 3, "coverage": 50}') is None
    assert parse_verdict("no json here") is None
    assert quality(1, 100) == 0.5 and quality(0, 100) == 0.0


def test_verdict_key_depends_on_judge_and_candidate():
    a = verdict_key("phi4mini", "q", "r", "c")
    assert a != verdict_key("qwen3vl4b", "q", "r", "c") and a != verdict_key("phi4mini", "q", "r", "c2")
    json.dumps(a)


def test_factqa_score_parsing_and_key():
    from videoqa.v3.factqa import FACTQA_PROMPT, parse_score, verdict_key

    assert parse_score("claims...\nScore: 3/5") == (3, 5)
    assert parse_score("Score: 1/4 ... final **Score: 2/4**") == (2, 4)
    assert parse_score("Score: 5/3") is None and parse_score("no score") is None
    assert verdict_key("m", "precision", "q", "a", "b") != verdict_key("m", "recall", "q", "b", "a")
    assert "{answer_1}" in FACTQA_PROMPT and "Score: <num supported claims>/<num total claims>" in FACTQA_PROMPT
