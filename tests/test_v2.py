"""Framework 2 units: teacher cache, label budgets, selectors and Head B, with fakes (no GPU)."""

import numpy as np
from PIL import Image

from videoqa.schemas import Answer, TranscriptSegment
from videoqa.v2.candidates import _stable_representatives
from videoqa.v2.head_b import HeadB, HeadBConfig, featurize
from videoqa.v2.labeling import head_a_labels, independent_labels, prefix_labels
from videoqa.v2.records import FrameCandidate, QuestionPool
from videoqa.v2.selectors import GreedyUtility, MMRSelector, SimilarityTopK, UnaryUtility
from videoqa.v2.teacher import CachedTeacher

RNG = np.random.default_rng(0)


def _unit(v):
    return v / np.linalg.norm(v)


def _pool(n=6, good=("f02",)):
    """A pool whose candidates carry random embeddings; `good` ids answer the question."""
    q = _unit(RNG.normal(size=512))
    cands = []
    for i in range(n):
        img = Image.new("RGB", (8, 8), (i * 30, 0, 0))
        cands.append(FrameCandidate(id=f"f{i:02d}", time_s=10.0 * i, paths=("B",), digest=f"d{i}", phash="0" * 16,
                                    clip_sim=float(i) / n, emb=_unit(RNG.normal(size=512)),
                                    ocr_emb=_unit(RNG.normal(size=512)), near_emb=_unit(RNG.normal(size=512)),
                                    image=img))
    pool = QuestionPool("q1", "v1", "what value?", ["a", "b"], 1,
                        [TranscriptSegment("s0", 0.0, 5.0, "some speech")], cands, duration_s=60.0, question_emb=q)
    return pool, set(good)


class FakeAnswerer:
    """Correct (option 1) iff a 'good' frame is shown; counts calls."""

    def __init__(self, good_digests):
        self.good, self.n = good_digests, 0

    def answer(self, req):
        self.n += 1
        ok = any(f.digest in self.good for f in req.frames)
        return Answer(text="B" if ok else "A", option_index=1 if ok else 0), None


def _teacher(tmp_path, pool, good):
    digests = {c.digest for c in pool.candidates if c.id in good}
    fake = FakeAnswerer(digests)
    return CachedTeacher(fake, "fake", tmp_path / "cache.jsonl"), fake


def test_teacher_cache_counts_real_calls_and_reuses_identical_evidence(tmp_path):
    pool, good = _pool()
    t, fake = _teacher(tmp_path, pool, good)
    a = t.quality(pool, [pool.candidates[2]])
    b = t.quality(pool, [pool.candidates[2]])
    assert a == b == 1.0 and fake.n == 1 and t.calls == 1 and t.hits == 1
    # frame ORDER does not change the identity (the answerer sees time order)
    t.quality(pool, [pool.candidates[1], pool.candidates[2]])
    t.quality(pool, [pool.candidates[2], pool.candidates[1]])
    assert fake.n == 2
    # a fresh teacher reloads the persisted cache: no new calls
    t2 = CachedTeacher(fake, "fake", tmp_path / "cache.jsonl")
    t2.quality(pool, [pool.candidates[2]])
    assert fake.n == 2 and t2.hits == 1


def test_label_schemes_respect_the_same_budget_and_keep_signed_gains(tmp_path):
    pool, good = _pool(n=8)
    t, fake = _teacher(tmp_path, pool, good)
    ind = independent_labels(pool, t, budget=5)
    assert len(ind) == 4 and all(r.history == [] for r in ind)
    calls_after_ind = t.calls
    assert calls_after_ind <= 5
    pre = prefix_labels(pool, t, chains=2, length=2)
    assert len(pre) == 4
    assert t.calls - calls_after_ind <= 4               # the shared baseline was reused from the cache
    assert any(r.history for r in pre)                   # deeper contexts were labelled
    assert all(r.gain == r.after - r.before for r in ind + pre)


def test_head_a_labels_use_separate_text_and_frame_interventions(tmp_path):
    pool, good = _pool()
    t, fake = _teacher(tmp_path, pool, good)
    seg = TranscriptSegment("s9", 1.0, 2.0, "the answer is b")
    rows = head_a_labels(pool, t, [(seg, pool.candidates[2])])
    assert rows[0]["visual_gain"] == 1.0                  # the frame alone fixed it
    assert rows[0]["text_gain"] == 0.0                    # the fake ignores text: no shared credit


def test_selectors_return_k_distinct_candidates():
    pool, _ = _pool(n=7)
    for sel in (SimilarityTopK(), MMRSelector()):
        out = sel.select(pool, 3)
        assert len(out) == 3 and len({c.id for c in out}) == 3
    assert SimilarityTopK().select(pool, 1)[0].id == "f06"


def test_head_b_learns_and_greedy_rescoring_uses_history(tmp_path):
    pool, good = _pool(n=6, good=("f02",))
    t, _ = _teacher(tmp_path, pool, good)
    rows = independent_labels(pool, t, budget=7) + prefix_labels(pool, t, chains=2, length=3)
    by_id = {c.id: c for c in pool.candidates}
    X = np.stack([featurize(pool, by_id[r.candidate_id], [by_id[h] for h in r.history]) for r in rows])
    y = np.array([r.gain for r in rows])
    ctx = np.array([f"{r.qa_id}|{','.join(r.history)}" for r in rows])
    head = HeadB(HeadBConfig(epochs=300))
    info = head.fit(X, y, ctx)
    assert info["params"] < 20_000 and info["rank_pairs"] > 0
    assert UnaryUtility(head).select(pool, 1)[0].id == "f02"
    out = GreedyUtility(head).select(pool, 3)
    assert out[0].id == "f02" and len({c.id for c in out}) == 3


def test_stable_representatives_keep_last_frame_of_each_run():
    class F:
        def __init__(self, t, ph):
            self.decoded_pts_s, self.phash = t, ph
    frames = [F(0, "0000"), F(1, "0001"), F(2, "ffff"), F(3, "fffe")]
    reps = _stable_representatives(frames, threshold=2)
    assert [r.decoded_pts_s for r in reps] == [1, 3]


def test_summarize_groups_seeds_and_reports_paired_differences():
    from videoqa.v2.pilot import summarize

    pool, _ = _pool()
    pool.timings = {"retrieval": 0.1, "head_a": 0.1, "decode": 1.0, "clip": 0.3, "ocr": 2.0, "text_emb": 0.01}
    rows = []
    for i in range(20):
        r = {"qa_id": f"q{i}", "video_id": f"v{i % 4}", "transcript_only": 0.0, "ocr_transcript_no_image": 0.0,
             "A_mobileclip_topk": float(i % 2), "B_mobileclip_mmr": float(i % 2)}
        for s in (0, 1, 2):
            r[f"C_independent_utility_s{s}"] = float(i % 2)
            r[f"D_history_utility_s{s}"] = 1.0
            r[f"D_head_unary_s{s}"] = float(i % 3 == 0)
        rows.append(r)
    out = summarize(rows, {"q": pool}, {"A_mobileclip_topk": [0.001], "D_history_utility_s0": [0.01]}, [1.2])
    assert out["accuracy"]["D_history_utility"] == 1.0 and out["accuracy"]["A_mobileclip_topk"] == 0.5
    assert out["paired"]["D_history_utility - A_mobileclip_topk"]["diff"] == 0.5
    lat = out["latency_s_median"]
    assert lat["D_history_utility"] > lat["A_mobileclip_topk"]        # D pays for OCR + text features


def test_no_ocr_features_are_zeroed_and_score_topk_ranks_by_given_scores():
    from videoqa.v2.selectors import ScoreTopK

    pool, _ = _pool(n=5)
    for c in pool.candidates:
        c.ocr_text = "some board text 42"
    with_ocr = featurize(pool, pool.candidates[0], [])
    no_ocr = featurize(pool, pool.candidates[0], [], use_ocr=False)
    assert not np.allclose(with_ocr[512:1024], 0) and np.allclose(no_ocr[512:1024], 0)   # OCR block zeroed
    sel = ScoreTopK(lambda p: {c.id: -c.time_s for c in p.candidates}, "earliest")
    assert [c.id for c in sel.select(pool, 2)] == ["f00", "f01"]


def test_segment_matching_for_path_a_frames():
    from videoqa.v2.experiment import _nearest_segment, _segment_ending_near

    segs = [TranscriptSegment("a", 0.0, 10.0, "x"), TranscriptSegment("b", 10.0, 30.0, "y")]
    assert _segment_ending_near(segs, 29.7).id == "b"        # frame taken 0.3 s before 'b' ends
    assert _nearest_segment(segs, 9.0).id == "a"
