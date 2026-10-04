"""Small CPU tests for leakage boundaries, pair weighting and inference."""
import copy
import importlib.util
from pathlib import Path

import numpy as np

from videoqa.schemas import TranscriptSegment
from videoqa.v2.records import FrameCandidate, QuestionPool, UtilityRow
from videoqa.v2.residual_selector import ResidualSelector, design, pair_blocks, raw_features


def fixture():
    pools, rows = {}, []
    for v in range(4):
        q = f"q{v}"
        candidates = [FrameCandidate(f"{q}:f{i}", i * 5.0, ("A", "B") if i % 2 else ("B",),
                                     f"d{i}", "0", clip_sim=i / 5, head_a_visual=4 - i,
                                     ocr_text=("matrix rank 2" if i % 2 else "vector 9"),
                                     near_text="matrix vector", emb=np.array([1., i + 1.]),
                                     ocr_emb=np.array([i + 1., 1.]), near_emb=np.array([1., 1.]))
                      for i in range(6)]
        pools[q] = QuestionPool(q, f"v{v}", "what is matrix rank 2?", ["2", "9"], 0,
                                [TranscriptSegment("s", 0., 20., "matrix rank 2")], candidates,
                                duration_s=30., question_emb=np.array([1., 0.]))
        rows += [UtilityRow(q, f"v{v}", c.id, [], 0., float(i % 2), float(i % 2), "independent")
                 for i, c in enumerate(candidates)]
    return pools, rows


def probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "v2_residual_probe.py"
    spec = importlib.util.spec_from_file_location("residual_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_text_only_changes_context_and_gold_never_changes_features():
    pools, _ = fixture()
    a = pools["q0"]
    b = copy.deepcopy(a)
    b.transcript[0].text = "unrelated cat 9"
    np.testing.assert_array_equal(raw_features(a), raw_features(b))
    np.testing.assert_array_equal(design(a)[0], design(b)[0])
    assert not np.allclose(design(a, True)[0], design(b, True)[0])
    b = copy.deepcopy(a)
    b.gold_option_index = 1
    for context in (False, True):
        np.testing.assert_array_equal(design(a, context)[0], design(b, context)[0])


def test_held_out_labels_do_not_affect_fit_or_inner_ridge():
    pools, rows = fixture()
    module = probe_module()
    modified = copy.deepcopy(rows)
    for r in modified:
        if r.video_id == "v0":
            r.after, r.gain = 1 - r.after, 1 - r.gain
    for context in (False, True):
        a, audit_a = module.fit_outer(pools, rows, "v0", context, k=2)
        b, audit_b = module.fit_outer(pools, modified, "v0", context, k=2)
        np.testing.assert_array_equal(a.weights, b.weights)
        assert audit_a == audit_b and "v0" not in a.train_videos
        assert "v0" not in audit_a["inner_cv"]["videos"]


def test_roundtrip_valid_k_and_fallback_ties(tmp_path):
    pools, rows = fixture()
    p = pools["q0"]
    model = ResidualSelector.fit(pair_blocks(pools, rows, True), True, 1.)
    path = tmp_path / "model.json"
    model.save(path)
    restored = ResidualSelector.load(path)
    np.testing.assert_allclose(model.scores(p), restored.scores(p))
    assert len({c.id for c in restored.select(p, 4)}) == 4
    assert restored.select(p, 0) == [] and restored.select(p, -2) == []
    assert len(restored.select(p, 99)) == 6
    for c in p.candidates:
        c.clip_sim = 0.
    assert [c.id for c in ResidualSelector().select(p, 3)] == sorted(c.id for c in p.candidates)[:3]


def test_question_pair_weighting_and_proxy_scope():
    pools, rows = fixture()
    blocks = pair_blocks(pools, rows)
    model = ResidualSelector.fit(blocks, ridge=1.)
    # Repeating every pair within one question must not increase its weight.
    altered = copy.deepcopy(blocks)
    altered[0].x = np.repeat(altered[0].x, 3, axis=0)
    altered[0].target = np.repeat(altered[0].target, 3)
    np.testing.assert_allclose(model.weights, ResidualSelector.fit(altered, ridge=1.).weights, atol=1e-12)
    module = probe_module()
    p = pools["q0"]
    subset = [r for r in rows if r.qa_id == p.qa_id][:3]
    metric = module.subset_metric(p, subset, design(p)[1], k=4)
    assert metric["selected_count"] == 3 and metric["labeled_candidates"] == 3
    assert set(metric["selected_ids"]) == {r.candidate_id for r in subset}


def test_empty_pool_inference():
    pools, _ = fixture()
    p = pools["q0"]
    p.candidates = []
    assert ResidualSelector(context=True).select(p, 4) == []
