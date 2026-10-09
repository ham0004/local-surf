"""Head A (temporal localisation): selection rule, metric, targets, and the residual starting point."""
import numpy as np
import pytest

from videoqa.v3.temporal_head import HeadAConfig, hit_at_k, select_peaks, target_distribution


def test_select_peaks_respects_min_separation():
    times = np.arange(0, 100, 2.0)
    scores = np.zeros_like(times)
    scores[10:14] = [5, 6, 7, 6]          # one bump around t=24 s
    scores[40] = 4                        # another at t=80 s
    chosen = select_peaks(scores, times, k=2, min_sep=8.0)
    assert chosen == [24.0, 80.0]


def test_hit_and_target_distribution():
    assert hit_at_k([10.0, 50.0], [[48.5, 60.0]]) == 1.0
    assert hit_at_k([10.0, 30.0], [[48.5, 60.0]]) == 0.0
    times = np.arange(0, 20, 2.0)
    p = target_distribution(times, [[6.0, 9.0]], pad=0.0)
    assert np.isclose(p.sum(), 1.0) and p[3] > 0 and p[0] == 0
    q = target_distribution(times, [[2.5, 3.0]], pad=0.0)    # interval between bins -> nearest bin
    assert np.isclose(q.sum(), 1.0) and q.argmax() in (1, 2)


def test_untrained_head_ranks_like_mobileclip():
    torch = pytest.importorskip("torch")
    from videoqa.v3.temporal_head import build_model

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[7] + 0.1 * rng.normal(size=512).astype(np.float32)
    q /= np.linalg.norm(q)
    model = build_model(HeadAConfig()).eval()
    with torch.no_grad():
        s = model(torch.tensor(emb)[None], torch.tensor(q)[None], torch.arange(30.0)[None] * 2,
                  torch.tensor([60.0]))[0].numpy()
    assert np.array_equal(np.argsort(-s), np.argsort(-(emb @ q)))
