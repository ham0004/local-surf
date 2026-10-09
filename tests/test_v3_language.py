"""Transcript language check (mislabelled caption tracks)."""
from videoqa.schemas import TranscriptSegment
from videoqa.v3.language import is_english_transcript, latin_share


def seg(t):
    return TranscriptSegment("s00001", 0.0, 1.0, t)


def test_tamil_track_labelled_english_is_rejected():
    assert not is_english_transcript([seg("\u0bb8\u0bcd\u0b9f\u0bc7\u0bb7\u0ba9\u0bcd B (PIGGYBACK)")])
    assert is_english_transcript([seg("the sender resends frames 3, 4, 5 and 6")])
    assert latin_share("123 ...") == 0.0
