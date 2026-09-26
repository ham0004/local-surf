"""Shared fixtures: the synthetic lecture video is encoded once per test session."""

import pytest

from videoqa import fixtures


@pytest.fixture(scope="session")
def lecture_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "fixture_lecture.mp4"
    return fixtures.write_video(path)
