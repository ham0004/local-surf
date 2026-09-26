"""Answer-quality scoring and the fixture answerer's documented behaviour."""

from videoqa import fixtures
from videoqa.answerer import (
    AnswerRequest,
    FixtureAnswerer,
    answer_quality,
    build_prompt_text,
    parse_option_letter,
    token_f1,
)
from videoqa.frames import decode_at
from videoqa.schemas import Answer, QAItem


def _qa(name):
    return {q.qa_id.split("/")[-1]: q for q in fixtures.qa_items()}[name]


def test_quality_multiple_choice_is_exact_option_match():
    qa = _qa("q_lr")
    assert answer_quality(Answer(text="0.3", option_index=2), qa) == 1.0
    assert answer_quality(Answer(text="0.1", option_index=1), qa) == 0.0


def test_quality_open_answer_uses_normalised_match_then_f1():
    qa = QAItem(qa_id="x", video_id="v", question="q", gold_answer="The green light")
    assert answer_quality(Answer(text="green light"), qa) == 1.0
    assert 0 < token_f1("green lamp", "green light") < 1


def test_parse_option_letter():
    assert parse_option_letter("C. because at 16s ...", 4) == 2
    assert parse_option_letter("no letter here", 4) is None


def test_prompt_marks_transcript_as_data_and_lists_options():
    text = build_prompt_text(AnswerRequest("q?", ["a", "b"], fixtures.transcript().segments[:1], []))
    assert "OPTIONS:\nA. a\nB. b" in text and "[s00000 1.0-6.0s]" in text


def test_fixture_answerer_needs_the_right_frame_for_visual_only_fact(lecture_video):
    qa, ans = _qa("q_acc"), FixtureAnswerer()
    # transcript alone: accuracy never spoken -> wrong default guess
    no_frame, _ = ans.answer(AnswerRequest(qa.question, qa.options, fixtures.transcript().segments, []))
    assert answer_quality(no_frame, qa) == 0.0
    # with the results-slide frame -> correct
    frame = decode_at(lecture_video, [35.0]).frames
    with_frame, usage = ans.answer(AnswerRequest(qa.question, qa.options, [], frame))
    assert answer_quality(with_frame, qa) == 1.0 and usage.visual_tokens > 0


def test_fixture_answerer_requires_ordered_pair_for_transition(lecture_video):
    qa, ans = _qa("q_switch"), FixtureAnswerer()
    only_after = decode_at(lecture_video, [60.0]).frames
    pair = decode_at(lecture_video, [50.0, 60.0]).frames
    assert answer_quality(ans.answer(AnswerRequest(qa.question, qa.options, [], only_after))[0], qa) == 0.0
    assert answer_quality(ans.answer(AnswerRequest(qa.question, qa.options, [], pair))[0], qa) == 1.0
