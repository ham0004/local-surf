"""Scout behaviour and the no-answer-leak restriction."""

import dataclasses

from videoqa import fixtures
from videoqa.frames import decode_at
from videoqa.schemas import ALLOWED_SCOUT_FIELDS, SceneType, ScoutSignals
from videoqa.scout import PixelStatsScout


def test_pixel_scout_outputs_only_allowed_signals(lecture_video):
    frames = decode_at(lecture_video, [5.0, 35.0]).frames
    out = PixelStatsScout().score("What final accuracy is shown?", frames)
    for sig in out.values():
        assert isinstance(sig, ScoutSignals)
        assert set(dataclasses.asdict(sig)) == ALLOWED_SCOUT_FIELDS
        # no string anywhere in the output except the enum value
        assert all(not isinstance(v, str) or isinstance(v, SceneType) for v in dataclasses.asdict(sig).values())


def test_pixel_scout_never_contains_answer_text(lecture_video):
    frames = decode_at(lecture_video, [35.0]).frames
    out = PixelStatsScout().score("accuracy?", frames)
    assert "87" not in repr(out)


def test_pixel_scout_is_honest_about_missing_question_awareness(lecture_video):
    frames = decode_at(lecture_video, [5.0]).frames
    sig = next(iter(PixelStatsScout().score("anything", frames).values()))
    assert sig.similarity == 0.5 and sig.uncertainty == 1.0


def test_slide_frames_classified_as_slides(lecture_video):
    frames = decode_at(lecture_video, [5.0, 20.0]).frames
    out = PixelStatsScout().score("q", frames)
    assert all(s.scene_type == SceneType.SLIDE for s in out.values())


def test_indicator_switch_registers_as_visual_change(lecture_video):
    # Same layout, only the indicator colour/label changes: the hash misses it
    # (see test_frames) but the pixel difference must not.
    frames = decode_at(lecture_video, [50.0, 60.0]).frames
    out = PixelStatsScout().score("q", frames)
    later = max(frames, key=lambda f: f.decoded_pts_s)
    assert out[later.id].visual_change > 0.05


def test_fixture_slides_render():
    assert fixtures.render_slide(fixtures.SLIDES[2]).size == (fixtures.WIDTH, fixtures.HEIGHT)
