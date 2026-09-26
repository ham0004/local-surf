"""The utility head learns, the paired term is wired correctly, and it round-trips."""

import numpy as np

from videoqa.heads import TrainConfig, UtilityHead


def _toy(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.normal(size=(n, 3))]
    y = np.tanh(X[:, 1] - 0.5 * X[:, 2])
    return X, y


def test_head_reduces_loss_on_learnable_target():
    X, y = _toy()
    head = UtilityHead(X.shape[1], hidden=16, seed=0)
    curve = head.fit(X, y, pairs=np.zeros((0, 2), int), cfg=TrainConfig(epochs=400, lambda_pair=0))
    assert curve[-1] < 0.2 * curve[0]
    assert np.corrcoef(head.predict(X), y)[0, 1] > 0.9


def test_paired_term_adds_loss_without_changing_forward():
    # The paired term only changes the loss, never the forward pass:
    # one Adam step with lr=0 must leave predictions unchanged, and the loss
    # with lambda_pair>0 must exceed the loss with lambda_pair=0 when pairs are wrong.
    X, y = _toy(n=20)
    pairs = np.array([[0, 1], [2, 3]])
    y_bad = y.copy()
    y_bad[0] += 1.0                   # make the paired difference hard to fit
    h1, h2 = UtilityHead(X.shape[1], 8, 1), UtilityHead(X.shape[1], 8, 1)
    l_nopair = h1.fit(X, y_bad, pairs, TrainConfig(epochs=1, lr=0.0, lambda_pair=0.0))[0]
    l_pair = h2.fit(X, y_bad, pairs, TrainConfig(epochs=1, lr=0.0, lambda_pair=1.0))[0]
    assert l_pair > l_nopair
    np.testing.assert_allclose(h1.predict(X), h2.predict(X))


def test_paired_training_fits_differences():
    # Rows 2k (targeted) and 2k+1 (control) share features except one flag;
    # target difference is +0.8.  Paired training should recover it.
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 3))
    Xt = np.c_[np.ones(50), base, np.ones(50)]
    Xc = np.c_[np.ones(50), base, np.zeros(50)]
    X = np.empty((100, 5))
    X[0::2], X[1::2] = Xt, Xc
    y = np.empty(100)
    y[0::2], y[1::2] = 0.9, 0.1
    pairs = np.c_[np.arange(0, 100, 2), np.arange(1, 100, 2)]
    head = UtilityHead(5, 16, 0)
    head.fit(X, y, pairs, TrainConfig(epochs=300, lambda_pair=1.0))
    diff = head.predict(X[0::2]) - head.predict(X[1::2])
    assert abs(diff.mean() - 0.8) < 0.1


def test_save_load_roundtrip(tmp_path):
    X, y = _toy(n=30)
    head = UtilityHead(X.shape[1], 8, 0)
    head.fit(X, y, np.zeros((0, 2), int), TrainConfig(epochs=10))
    head.save(tmp_path / "h.npz", stop_threshold=0.07)
    loaded, meta = UtilityHead.load(tmp_path / "h.npz")
    np.testing.assert_allclose(loaded.predict(X), head.predict(X))
    assert meta["stop_threshold"] == 0.07
