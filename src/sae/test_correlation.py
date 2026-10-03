"""Unit tests for the streaming sufficient-statistics correlation (2606.03002 App. I)."""

import numpy as np

from src.sae.correlation import StreamingPearson


def test_streaming_matches_exact():
    rng = np.random.default_rng(0)
    n, d = 5000, 32
    x = rng.normal(0, 1, (n, d))
    y = x * np.linspace(0.1, 0.99, d) + rng.normal(0, 0.5, (n, d))
    est = StreamingPearson(d_sae=d)
    for i in range(0, n, 500):
        est.update(x[i : i + 500], y[i : i + 500])
    streamed = est.finalize()
    exact = np.array([np.corrcoef(x[:, j], y[:, j])[0, 1] for j in range(d)])
    assert np.abs(streamed - exact).max() < 1e-9


def test_chunk_boundary_invariance():
    rng = np.random.default_rng(1)
    n, d = 1000, 16
    x = rng.normal(0, 1, (n, d))
    y = x + rng.normal(0, 0.2, (n, d))
    est1 = StreamingPearson(d_sae=d)
    est1.update(x, y)
    est2 = StreamingPearson(d_sae=d)
    for i in range(0, n, 3):  # odd chunking must not change the result
        est2.update(x[i : i + 3], y[i : i + 3])
    assert np.abs(est1.finalize() - est2.finalize()).max() < 1e-9


def test_identical_inputs_give_unit_correlation():
    x = np.random.default_rng(2).normal(0, 1, (100, 8))
    est = StreamingPearson(d_sae=8)
    est.update(x, x)
    assert np.allclose(est.finalize(), 1.0)