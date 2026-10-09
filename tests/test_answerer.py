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


# --- citations and abstention (2026-09-28 repairs) --------------------------
from videoqa.answerer import finalize_answer, is_abstention, parse_citations  # noqa: E402
from videoqa.schemas import Frame, TranscriptSegment as Seg  # noqa: E402


def _frame(t):
    return Frame(id=f"f{t}", requested_s=t, decoded_pts_s=t, width=1, height=1)


def test_citations_are_what_the_model_cited_not_every_supplied_frame():
    # Regression: citations_s used to be a copy of all supplied frame times.
    excerpt = [Seg("s00002", 16.0, 22.0, "the learning rate was set to 0.3")]
    a = finalize_answer("C. 0.3, shown at 35.00s and said at 16.0-22.0s", ["a", "b", "c"], [0.1, 0.1, 0.8],
                        [_frame(5.0), _frame(35.0)], excerpt)
    assert a.supplied_frame_times_s == [5.0, 35.0]
    assert a.citations_s == [35.0]                 # only the frame it named
    assert a.cited_segment_ids == ["s00002"]        # 16.0 falls inside that line
    assert a.option_index == 2 and not a.forced_choice


def test_citation_to_nothing_supplied_is_recorded_as_invalid():
    fr, segs, bad = parse_citations("see 99.0s and s00077", [5.0], [Seg("s00001", 0, 2, "x")])
    assert fr == [] and segs == [] and set(bad) == {"99.0s", "s00077"}


def test_abstention_is_kept_not_turned_into_a_forced_guess():
    a = finalize_answer("The evidence is insufficient to determine this.", ["a", "b"], [0.9, 0.1], [], [])
    assert a.abstained and a.option_index is None and a.confidence is None
    assert is_abstention("I cannot determine the colour from these frames")


def test_letterless_answer_uses_models_own_top_letter_and_is_flagged():
    a = finalize_answer("green, because the light changed", ["red", "green"], [0.2, 0.8], [], [])
    assert a.option_index == 1 and a.forced_choice and a.confidence == 0.8


def test_parse_option_letter_beyond_h_and_pronoun_i():
    from videoqa.answerer import parse_option_letter

    assert parse_option_letter("J. the tenth option", 10) == 9
    assert parse_option_letter("(I) because the slide says so", 10) == 8
    assert parse_option_letter("I think the answer is B", 10) == 1
    assert parse_option_letter("Answer: G", 10) == 6
    assert parse_option_letter("B", 4) == 1
