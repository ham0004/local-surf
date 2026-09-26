"""Frame decoding: real PTS, in-bounds times, order preservation, visited counts."""

from videoqa import fixtures
from videoqa.frames import average_hash, decode_at, hamming_hex, probe


def test_probe_reads_duration(lecture_video):
    info = probe(lecture_video)
    assert abs(info.duration_s - fixtures.DURATION_S) < 0.5
    assert (info.width, info.height) == (fixtures.WIDTH, fixtures.HEIGHT)


def test_decode_returns_true_pts_close_to_request_in_request_order(lecture_video):
    times = [40.0, 5.0, 20.1]      # deliberately unsorted
    res = decode_at(lecture_video, times)
    assert [f.requested_s for f in res.frames] == times
    for f in res.frames:
        # 5 fps -> the served frame is at most one frame (0.2 s) after the request
        assert 0 <= f.decoded_pts_s - f.requested_s <= 0.2 + 1e-6


def test_request_past_end_is_clamped_to_last_frame(lecture_video):
    res = decode_at(lecture_video, [500.0])
    assert res.frames[0].decoded_pts_s <= fixtures.DURATION_S


def test_visited_counts_every_decoded_frame(lecture_video):
    res = decode_at(lecture_video, [0.0, 2.0])
    # frames 0..10 at 5 fps must all be decoded to reach t=2.0
    assert res.frames_visited >= 11


def test_max_side_resizes(lecture_video):
    res = decode_at(lecture_video, [5.0], max_side=320)
    assert max(res.frames[0].width, res.frames[0].height) == 320


def test_average_hash_is_layout_level_only(lecture_video):
    # Documents a real limitation: the 8x8 hash cannot tell "Switch OFF / red"
    # from "Switch ON / green" (same layout, different answer!).  This is why
    # hashes are never used to drop frames the controller asked for, and why
    # the scout's visual_change uses a finer pixel difference.
    res = decode_at(lecture_video, [50.0, 60.0])
    assert hamming_hex(res.frames[0].phash, res.frames[1].phash) <= 2


def test_average_hash_is_stable():
    img = fixtures.render_slide(fixtures.SLIDES[0])
    assert average_hash(img) == average_hash(img.copy())


def test_seeking_reduces_decoded_frames_on_short_gop_video(tmp_path):
    """With keyframes every 2 s, sparse requests should not decode the whole span."""
    from fractions import Fraction

    import av
    import numpy as np

    path = tmp_path / "short_gop.mp4"
    with av.open(str(path), mode="w") as c:
        s = c.add_stream("libx264", rate=Fraction(10, 1))
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.options = {"g": "20", "keyint_min": "20", "sc_threshold": "0"}   # keyframe every 2 s
        for i in range(600):                                               # 60 s
            img = np.full((48, 64, 3), i % 255, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    times = [5.0, 30.0, 55.0]
    seek = decode_at(path, times)
    linear = decode_at(path, times, seek_gap_s=float("inf"))
    assert [f.decoded_pts_s for f in seek.frames] == [f.decoded_pts_s for f in linear.frames]
    assert seek.frames_visited < linear.frames_visited / 3
