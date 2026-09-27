"""Lazy selective search: each policy pays only for the visual work it does."""

import dataclasses

from videoqa import fixtures
from videoqa.acquisition import VisualCache, prepare_text, run_lazy, seed_candidates
from videoqa.answerer import FixtureAnswerer, answer_quality
from videoqa.config import budget_from_config, load_config
from videoqa.controller import (
    HeuristicController,
    RetrievalPolicy,
    ScoutSimilarityPolicy,
    TranscriptOnlyPolicy,
    UniformPolicy,
)
from videoqa.costs import CostMeter
from videoqa.scout import PixelStatsScout

CFG = load_config("configs/cpu.yaml")
BUDGET8 = dataclasses.replace(budget_from_config(CFG), max_frames=8)


class CountingScout(PixelStatsScout):
    def __init__(self):
        self.calls = 0

    def score(self, question, frames):
        self.calls += len(frames)
        return super().score(question, frames)


def _qa(name):
    return {q.qa_id.split("/")[-1]: q for q in fixtures.qa_items()}[name]


def _run(video, policy, qa_name="q_acc", budget=None):
    qa, meter, scout = _qa(qa_name), CostMeter(), CountingScout()
    prep = prepare_text(video, fixtures.transcript(), qa.question, qa.options, CFG, meter)
    cache = VisualCache(prep, scout, meter, CFG["answerer"].get("frame_max_side"))
    res = run_lazy(prep, cache, policy, FixtureAnswerer(), budget or budget_from_config(CFG), meter)
    return prep, res, meter, scout, qa


def _stage(meter, name):
    return [r for r in meter.records if r.stage == name]


def test_transcript_only_decodes_and_scouts_nothing(lecture_video):
    prep, res, meter, scout, _ = _run(lecture_video, TranscriptOnlyPolicy())
    assert not prep.frames and not prep.scout and scout.calls == 0
    assert not _stage(meter, "decode") and not _stage(meter, "scout")
    assert meter.total().decoded_frames == 0 and res.frames == []


def test_text_preparation_alone_decodes_nothing(lecture_video):
    meter = CostMeter()
    qa = _qa("q_acc")
    prep = prepare_text(lecture_video, fixtures.transcript(), qa.question, qa.options, CFG, meter)
    assert prep.candidates and meter.total().decoded_frames == 0 and prep.frame_size[0] > 0


def test_uniform_and_retrieval_never_pay_for_the_scout(lecture_video):
    for policy in (UniformPolicy(), RetrievalPolicy()):
        prep, res, meter, scout, _ = _run(lecture_video, policy, budget=BUDGET8)
        assert scout.calls == 0 and not _stage(meter, "scout")
        # only what was looked at was decoded
        assert set(prep.frames) == set(res.state.looked_at)


def test_scout_similarity_pays_for_every_non_rescue_candidate(lecture_video):
    prep, res, meter, scout, _ = _run(lecture_video, ScoutSimilarityPolicy())
    assert scout.calls == len(seed_candidates(prep, "all")) > 0


def test_learned_style_controller_scouts_only_its_seed(lecture_video):
    prep, res, meter, scout, _ = _run(lecture_video, HeuristicController())
    assert scout.calls == min(4, len(seed_candidates(prep, "all")))


def test_lazy_path_still_answers_a_visual_only_question(lecture_video):
    _, res, _, _, qa = _run(lecture_video, UniformPolicy(), budget=BUDGET8)
    assert answer_quality(res.answer, qa) == 1.0 and not res.cap_violations


def test_scout_signals_do_not_depend_on_the_candidate_pool(lecture_video):
    """visual_change uses a fixed neighbour, so the same moment scores the same
    whether or not other candidates exist."""
    qa, meter = _qa("q_acc"), CostMeter()
    full = prepare_text(lecture_video, fixtures.transcript(), qa.question, qa.options, CFG, meter)
    only = prepare_text(lecture_video, fixtures.transcript(), qa.question, qa.options, CFG, meter,
                        candidates=full.candidates[2:3])
    cid = full.candidates[2].id
    a = VisualCache(full, PixelStatsScout(), meter, None)
    for c in full.candidates:
        a.ensure_scout(c.id)
    b = VisualCache(only, PixelStatsScout(), meter, None).ensure_scout(cid)
    assert a.ensure_scout(cid) == b
