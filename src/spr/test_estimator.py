"""Unit tests for the SPR estimator."""

import numpy as np
import pytest

from src.spr.estimator import cosine_alignment, estimate_spr_from_stats, estimate_spr_streaming


def test_high_signal_direction_is_robust():
    rng = np.random.default_rng(0)
    n, d, k = 1000, 16, 4
    dirs = rng.normal(0, 1, (k, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    fp16 = rng.normal(0, 1, (n, d))
    # Strong structure along dirs[0]; big perturbation along dirs[1].
    fp16 = fp16 + 5.0 * np.outer(np.abs(rng.normal(0, 1, n)), dirs[0])
    quant = fp16 + rng.normal(0, 0.05, (n, d)) + 0.9 * np.outer(rng.normal(0, 1, n), dirs[1])
    spr = estimate_spr_streaming(dirs, [fp16], [quant], signal_stat="mean")
    assert spr[0] > spr[1]  # high-signal direction out-SPRs the perturbed one


def test_peak_and_mean_variants_agree_in_order():
    rng = np.random.default_rng(1)
    n, d, k = 500, 8, 3
    dirs = rng.normal(0, 1, (k, d))
    fp16 = rng.normal(0, 1, (n, d))
    quant = fp16 + rng.normal(0, 0.1, (n, d))
    spr_mean = estimate_spr_streaming(dirs, [fp16], [quant], signal_stat="mean")
    spr_peak = estimate_spr_streaming(dirs, [fp16], [quant], signal_stat="peak")
    assert np.all(spr_peak >= spr_mean)  # peak signal >= mean signal


def test_streaming_matches_direct_computation():
    rng = np.random.default_rng(2)
    n, d = 600, 5
    dirs = rng.normal(0, 1, (3, d))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)  # same normalization as the estimator
    fp16 = rng.normal(0, 1, (n, d))
    quant = fp16 + rng.normal(0, 0.2, (n, d))
    # Direct computation (abs AFTER projection — matches the estimator).
    mean_delta = np.abs((quant - fp16) @ dirs.T).mean(axis=0)
    signal = np.abs(fp16 @ dirs.T).mean(axis=0)
    expected = signal / mean_delta
    streamed = estimate_spr_streaming(dirs, [fp16], [quant], signal_stat="mean")
    assert np.allclose(streamed, expected)


def test_cosine_alignment_is_symmetric_diagonal_one():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (4, 10))
    b = rng.normal(0, 1, (3, 10))
    sim = cosine_alignment(a, b)
    assert sim.shape == (4, 3)
    assert np.allclose(np.diag(cosine_alignment(a, a)), 1.0)
    assert np.allclose(cosine_alignment(a, b), cosine_alignment(b, a).T)


def test_from_stats():
    spr = estimate_spr_from_stats(np.array([2.0]), np.array([0.5]))
    assert spr[0] == 4.0


def test_primary_estimator_is_peak_over_mean_directional_delta():
    direction = np.array([[2.0, 0.0]])  # estimator must L2-normalize this
    fp16 = np.array([[1.0, 8.0], [-3.0, 4.0]])
    quant = fp16 + np.array([[0.5, 100.0], [-0.5, -100.0]])

    spr = estimate_spr_streaming(direction, [fp16], [quant])

    # peak |projection| = 3; mean directional |delta| = 0.5.
    # The large orthogonal errors must not enter the directional denominator.
    assert np.allclose(spr, [6.0])


def test_exact_reproduction_returns_finite_eps_regularized_spr():
    direction = np.array([[1.0, 0.0]])
    activations = np.array([[2.0, 1.0]])
    spr = estimate_spr_streaming(
        direction, [activations], [activations.copy()], eps=0.25
    )
    assert np.all(np.isfinite(spr))
    assert np.allclose(spr, [8.0])


@pytest.mark.parametrize(
    "directions",
    [
        np.array([[0.0, 0.0]]),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
    ],
)
def test_rejects_zero_or_nonfinite_directions(directions):
    with pytest.raises(ValueError, match="directions"):
        estimate_spr_streaming(directions, [np.ones((1, 2))], [np.ones((1, 2))])


def test_rejects_empty_misaligned_and_shape_mismatched_chunks():
    direction = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="streams are empty"):
        estimate_spr_streaming(direction, [], [])
    with pytest.raises(ValueError, match="different numbers"):
        estimate_spr_streaming(direction, [np.ones((1, 2))], [])
    with pytest.raises(ValueError, match="misaligned"):
        estimate_spr_streaming(
            direction, [np.ones((2, 2))], [np.ones((1, 2))]
        )
    with pytest.raises(ValueError, match="hidden size"):
        estimate_spr_streaming(
            direction, [np.ones((1, 3))], [np.ones((1, 3))]
        )
    with pytest.raises(ValueError, match="empty"):
        estimate_spr_streaming(
            direction, [np.ones((0, 2))], [np.ones((0, 2))]
        )


def test_from_stats_rejects_shape_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError, match="shapes differ"):
        estimate_spr_from_stats(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="must be finite"):
        estimate_spr_from_stats(np.array([np.nan]), np.ones(1))
