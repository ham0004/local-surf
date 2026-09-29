"""Evaluation harness: conditions, fair charging, hypothesis metrics, bootstrap."""

import dataclasses

from videoqa import fixtures
from videoqa.answerer import FixtureAnswerer
from videoqa.config import budget_from_config, load_config
from videoqa.controller import TranscriptOnlyPolicy, UniformPolicy
from videoqa.evaluate import EvalRecord, cluster_bootstrap_diff, evaluate, summarise, to_markdown
from videoqa.scout import PixelStatsScout

CFG = load_config("configs/cpu.yaml")
BUDGET = dataclasses.replace(budget_from_config(CFG), max_frames=8)


def test_evaluate_runs_all_conditions_and_charges_text_only_fairly(lecture_video):
    qa = [q for q in fixtures.qa_items() if q.qa_id.endswith("q_lr")][0]
    recs, skipped = evaluate([(qa, str(lecture_video), fixtures.transcript())],
                             [TranscriptOnlyPolicy(), UniformPolicy()], CFG, BUDGET, PixelStatsScout(),
                             FixtureAnswerer())
    assert skipped == 0
    assert {r.condition for r in recs} == {"clean", "targeted_damage", "control_damage", "asr_noise"}
    text_only = [r for r in recs if r.policy == "transcript_only"]
    assert all(r.decoded_frames == 0 and r.frames == 0 for r in text_only)   # never charged decoding
    s = summarise(recs)
    # transcript-only fails exactly when answer-relevant speech is removed
    assert s["transcript_only"]["clean"]["quality"] == 1.0
    assert s["transcript_only"]["targeted_damage"]["quality"] == 0.0
    assert s["transcript_only"]["control_damage"]["quality"] == 1.0
    assert "selectivity" in s["uniform"]["hypothesis"]
    assert "| uniform | clean |" in to_markdown(s)


def test_clean_only_benchmark_mode_runs_every_question_once_per_policy(lecture_video):
    items = [(q, str(lecture_video), fixtures.transcript()) for q in fixtures.qa_items()]
    recs, skipped = evaluate(items, [TranscriptOnlyPolicy(), UniformPolicy()], CFG, BUDGET, PixelStatsScout(),
                             FixtureAnswerer(), clean_only=True)
    assert skipped == 0                                   # no fair-triple requirement in this mode
    assert {r.condition for r in recs} == {"clean"}
    assert len(recs) == 2 * len(items)


def _rec(q, v, pol, quality):
    return EvalRecord(q, v, pol, "clean", quality, 0, 0, 0, 0.0, [])


def test_cluster_bootstrap_resamples_videos():
    recs = []
    for v in range(10):
        for k in range(3):
            recs += [_rec(f"v{v}q{k}", f"v{v}", "a", 1.0), _rec(f"v{v}q{k}", f"v{v}", "b", 0.0)]
    out = cluster_bootstrap_diff(recs, "a", "b", "clean", n_boot=200)
    assert out["diff"] == 1.0 and out["ci95_low"] == 1.0 and out["n_videos"] == 10


def test_selectivity_bootstrap_per_question_and_paired_difference():
    from videoqa.evaluate import selectivity_bootstrap

    def rec(q, v, pol, cond, frames):
        return EvalRecord(q, v, pol, cond, 1.0, frames, 0, 0, 0.0, [])
    recs = []
    for v in range(6):
        q = f"v{v}q"
        # policy "sel" looks 2 more frames under targeted than control; "flat" never differs
        recs += [rec(q, f"v{v}", "sel", "targeted_damage", 3), rec(q, f"v{v}", "sel", "control_damage", 1),
                 rec(q, f"v{v}", "flat", "targeted_damage", 2), rec(q, f"v{v}", "flat", "control_damage", 2)]
    one = selectivity_bootstrap(recs, "sel", n_boot=200)
    assert one["selectivity"] == 2.0 and one["ci95_low"] == 2.0 and one["n_videos"] == 6
    diff = selectivity_bootstrap(recs, "sel", "flat", n_boot=200)
    assert diff["selectivity_diff"] == 2.0
    assert selectivity_bootstrap(recs, "missing") is None


def test_recovery_table_uses_the_speech_failure_stratum():
    from videoqa.evaluate import recovery_table, speech_failure_stratum

    def rec(q, v, pol, cond, quality):
        return EvalRecord(q, v, pol, cond, quality, 1, 0, 0, 0.0, [], warm_ms=100.0)
    recs = []
    for i in range(4):
        q, v = f"q{i}", f"v{i}"
        broke = i < 2                       # speech mattered for q0, q1 only
        recs += [rec(q, v, "transcript_only", "clean", 1.0),
                 rec(q, v, "transcript_only", "targeted_damage", 0.0 if broke else 1.0),
                 rec(q, v, "looker", "targeted_damage", 1.0 if i == 0 else 0.0),
                 rec(q, v, "looker", "control_damage", 1.0)]
    assert speech_failure_stratum(recs) == {"q0", "q1"}
    t = recovery_table(recs, n_boot=100)
    assert t["stratum_questions"] == 2 and t["policies"]["looker"]["recovery"] == 0.5
    assert t["policies"]["transcript_only"]["recovery"] == 0.0
