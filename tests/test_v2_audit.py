"""Regression tests for the result audit; no models or GPU calls."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from videoqa.schemas import TranscriptSegment
from videoqa.v2.head_b import featurize
from videoqa.v2.records import FrameCandidate, QuestionPool


def _conditioning_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "v2_transcript_conditioning.py"
    spec = importlib.util.spec_from_file_location("conditioning_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_correctness_separates_redundancy_from_negative_interference():
    no_speech, with_speech = [], []
    # Exercise all eight joint outcomes, plus a nonzero no-evidence baseline:
    # gain-only-with-speech must not automatically be called joint-only success.
    outcomes = [(0, f, s, c) for f in (0, 1) for s in (0, 1) for c in (0, 1)] + [(1, 1, 0, 1)]
    for i, (base, frame, speech, combined) in enumerate(outcomes):
        identity = {"qa_id": "q1", "candidate_id": f"f{i}", "video_id": "v1"}
        no_speech.append({**identity, "base": base, "visual_gain": frame - base})
        with_speech.append({**identity, "before": speech, "after": combined, "gain": combined - speech})
    summary = _conditioning_module().summarize_conditioning(no_speech, with_speech, bootstrap_samples=20)
    assert summary["matched_frames"] == 9
    assert summary["table"] == {"useful_both": 1, "useful_only_without_speech": 3,
                                "useful_only_with_speech": 2, "useful_neither": 3}
    assert summary["useful_only_without_speech_decomposition"] == {
        "redundant_success_both_sources_and_combination_correct": 1,
        "destructive_combination_both_sources_correct_alone": 1,
        "speech_interference_frame_correct_speech_and_combination_wrong": 1,
        "other": 0,
    }
    assert summary["joint_only_success_neither_source_correct_alone"] == 1
    assert summary["frames_where_no_evidence_already_answered"] == 1
    assert sum(row["count"] for row in summary["joint_correctness_outcomes"]) == 9
    recovered_frame_correct = sum(row["count"] for row in summary["joint_correctness_outcomes"]
                                  if row["frame_alone_correct"])
    assert recovered_frame_correct == 5


@pytest.mark.parametrize("use_ocr", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
def test_speech_blind_features_ignore_speech_scores_paths_and_history_paths(use_ocr, with_history):
    rng = np.random.default_rng(20)
    candidate = FrameCandidate("f1", 20.0, ("A", "B"), "d1", "0000", 0.4,
                               emb=rng.normal(size=512), ocr_emb=rng.normal(size=512),
                               near_emb=rng.normal(size=512), ocr_text="matrix 42", near_text="matrix 42",
                               head_a_text=1.5, head_a_visual=2.5)
    previous = copy.deepcopy(candidate)
    previous.id, previous.time_s, previous.paths = "f0", 5.0, ("A",)
    history = [previous] if with_history else []
    pool = QuestionPool("q1", "v1", "What matrix value?", ["42", "10"], 0,
                        [TranscriptSegment("s0", 0.0, 10.0, "matrix forty two")],
                        [candidate, previous], duration_s=100.0, question_emb=rng.normal(size=512))
    blind_before = featurize(pool, candidate, history, use_ocr, use_transcript=False)
    full_before = featurize(pool, candidate, history, use_ocr, use_transcript=True)
    candidate.head_a_text, candidate.head_a_visual = 99.0, -99.0
    candidate.near_text, candidate.near_emb = "unrelated utterance", rng.normal(size=512)
    candidate.paths = ("B",)
    previous.paths = ("A", "different")
    pool.transcript = [TranscriptSegment("s9", 70.0, 90.0, "different speech")]
    blind_after = featurize(pool, candidate, history, use_ocr, use_transcript=False)
    full_after = featurize(pool, candidate, history, use_ocr, use_transcript=True)
    np.testing.assert_array_equal(blind_before, blind_after)
    assert not np.array_equal(full_before, full_after)
    # A blind selector still uses the visual signal on the fixed candidate pool.
    candidate.emb *= -1
    assert not np.array_equal(blind_after, featurize(pool, candidate, history, use_ocr, use_transcript=False))
