"""Synthetic lecture generator: variety, determinism and on-disk dataset layout."""

import json

from videoqa import fixtures
from videoqa.answerer import AnswerRequest, FixtureAnswerer, answer_quality
from videoqa.frames import decode_at


def test_generated_specs_are_deterministic_and_varied():
    assert fixtures.generate_spec(3) == fixtures.generate_spec(3)
    specs = [fixtures.generate_spec(s) for s in range(30)]
    assert len({s.acc for s in specs}) > 1 and len({(s.before, s.after) for s in specs}) > 1
    # some lectures speak the accuracy aloud, some do not -> vision sometimes needed
    spoken = ["final accuracy was" in " ".join(t for _, _, t in s.speech) for s in specs]
    assert any(spoken) and not all(spoken)


def test_write_dataset_layout_and_frame_lookup(tmp_path):
    qa_path = fixtures.write_dataset(tmp_path, n_videos=2, seed0=100)
    rows = [json.loads(line) for line in qa_path.read_text().splitlines()]
    assert len(rows) == 6 and rows[0]["video_id"] == "synth_0100"
    assert (tmp_path / "videos" / "synth_0100.mp4").exists()
    assert (tmp_path / "transcripts" / "synth_0101.json").exists()

    # the fixture answerer reads frames of THIS lecture, not the default one
    spec = fixtures.get_spec("synth_0100")
    qa = fixtures.qa_items("synth_0100")[1]
    frame = decode_at(tmp_path / "videos" / "synth_0100.mp4", [35.0]).frames
    assert frame[0].video_id == "synth_0100"
    ans, _ = FixtureAnswerer().answer(AnswerRequest(qa.question, qa.options, [], frame))
    assert ans.text == spec.acc and answer_quality(ans, qa) == 1.0


def test_get_spec_rebuilds_synthetic_ids_and_rejects_unknown_ids():
    import pytest
    fixtures._REGISTRY.pop("synth_0777", None)      # simulate a fresh process
    assert fixtures.get_spec("synth_0777") == fixtures.generate_spec(777)
    with pytest.raises(KeyError):
        fixtures.get_spec("some_real_video")


def test_fixture_answerer_matches_whole_options_only():
    from videoqa.answerer import AnswerRequest, FixtureAnswerer
    from videoqa.schemas import TranscriptSegment
    seg = [TranscriptSegment("s0", 0, 1, "we covered the settings; rate 0.01")]
    ans, _ = FixtureAnswerer().answer(AnswerRequest("q?", ["red", "0.1", "0.01"], seg, []))
    assert ans.text == "0.01"      # not "red" (in 'covered') and not "0.1" (prefix of 0.01)
