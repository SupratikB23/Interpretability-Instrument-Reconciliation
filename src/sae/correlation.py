"""Streaming sufficient-statistics Pearson correlation estimator.

Matches Duan, E. (2026). *Perplexity Can Miss SAE Feature Damage Under
Quantization.* arXiv:2606.03002, Appendix I: the per-feature Pearson correlation
``c_j`` between FP16 and quantized SAE feature activations is computed via
streaming sufficient statistics so that memory scales as ``O(d_sae)`` rather
than ``O(N * d_sae)``. Naive pairwise correlation over
500k tokens x 16,384 features is not viable; reuse this trick.

This class is pure (numpy-only) so it can be unit-tested without a GPU or SAE.
"""

from __future__ import annotations

import numpy as np


class StreamingPearson:
    """Incrementally accumulate per-feature correlation sufficient statistics.

    Each ``update`` call consumes a batch of aligned ``(x, y)`` feature-activation
    rows and folds them into per-feature running sums, so total memory is
    ``O(d_sae)`` regardless of the number of tokens N (Duan 2606.03002 App. I).

    Attributes:
        d: number of features (set on the first ``update``).
    """

    def __init__(self, d_sae: int | None = None) -> None:
        if d_sae is not None and (isinstance(d_sae, bool) or not isinstance(d_sae, int) or d_sae <= 0):
            raise ValueError(f"d_sae must be a positive integer or None, got {d_sae!r}")
        self.d: int | None = d_sae
        self.n: int = 0
        self.sum_x: np.ndarray | None = None
        self.sum_y: np.ndarray | None = None
        self.sum_x2: np.ndarray | None = None
        self.sum_y2: np.ndarray | None = None
        self.sum_xy: np.ndarray | None = None
        # Central co-moments are merged batchwise (Chan-Golub-LeVeque). The raw
        # sufficient statistics above are retained for auditability, while these
        # avoid catastrophic cancellation when activations have a large offset.
        self._mean_x: np.ndarray | None = None
        self._mean_y: np.ndarray | None = None
        self._m2_x: np.ndarray | None = None
        self._m2_y: np.ndarray | None = None
        self._co_moment: np.ndarray | None = None

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fold a batch of aligned (FP16, quantized) activations in.

        Args:
            x: shape ``[batch, d_sae]`` FP16 feature activations.
            y: shape ``[batch, d_sae]`` quantized feature activations.

        Docstring reference: Duan 2606.03002 Appendix I.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            raise ValueError(f"x and y must be aligned [batch, d] arrays, got {x.shape} vs {y.shape}")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"x and y must contain a non-empty batch and feature dimension, got {x.shape}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("x and y must contain only finite activations")
        if self.d is None:
            self.d = x.shape[1]
        elif x.shape[1] != self.d:
            raise ValueError(f"feature dim mismatch: {x.shape[1]} != {self.d}")
        batch = x.shape[0]
        batch_mean_x = x.mean(axis=0)
        batch_mean_y = y.mean(axis=0)
        centered_x = x - batch_mean_x
        centered_y = y - batch_mean_y
        batch_m2_x = np.einsum("ij,ij->j", centered_x, centered_x)
        batch_m2_y = np.einsum("ij,ij->j", centered_y, centered_y)
        batch_co_moment = np.einsum("ij,ij->j", centered_x, centered_y)

        if self.sum_x is None:
            self.sum_x = np.zeros(self.d, dtype=np.float64)
            self.sum_y = np.zeros(self.d, dtype=np.float64)
            self.sum_x2 = np.zeros(self.d, dtype=np.float64)
            self.sum_y2 = np.zeros(self.d, dtype=np.float64)
            self.sum_xy = np.zeros(self.d, dtype=np.float64)
            self._mean_x = batch_mean_x.copy()
            self._mean_y = batch_mean_y.copy()
            self._m2_x = batch_m2_x.copy()
            self._m2_y = batch_m2_y.copy()
            self._co_moment = batch_co_moment.copy()
        else:
            total = self.n + batch
            delta_x = batch_mean_x - self._mean_x
            delta_y = batch_mean_y - self._mean_y
            merge_weight = self.n * batch / total
            self._m2_x += batch_m2_x + delta_x * delta_x * merge_weight
            self._m2_y += batch_m2_y + delta_y * delta_y * merge_weight
            self._co_moment += batch_co_moment + delta_x * delta_y * merge_weight
            self._mean_x += delta_x * batch / total
            self._mean_y += delta_y * batch / total

        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.sum_x2 += np.einsum("ij,ij->j", x, x)
        self.sum_y2 += np.einsum("ij,ij->j", y, y)
        self.sum_xy += np.einsum("ij,ij->j", x, y)
        self.n += batch

    def finalize(self) -> np.ndarray:
        """Return the per-feature Pearson correlations ``c_j`` for all ``d_sae`` features.

        Correlation is undefined (NaN) for features with zero variance in either
        condition; those are left as NaN and excluded downstream by the
        active-feature gate.
        """
        if self.n == 0 or self.sum_x is None:
            raise RuntimeError("finalize() called before any update()")
        if self.n < 2:
            return np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = self._co_moment / np.sqrt(self._m2_x * self._m2_y)
        invalid = (self._m2_x <= 0.0) | (self._m2_y <= 0.0)
        c[invalid] = np.nan
        finite = np.isfinite(c)
        c[finite] = np.clip(c[finite], -1.0, 1.0)
        return c


def pearson_from_sufficient_statistics(
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_x2: np.ndarray,
    sum_y2: np.ndarray,
    sum_xy: np.ndarray,
    n: int,
) -> np.ndarray:
    """Direct Pearson computation from accumulated sufficient statistics.

    Exposed for testing :meth:`StreamingPearson.finalize` against an equivalent
    closed-form batch computation.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (sum_x, sum_y, sum_x2, sum_y2, sum_xy)]
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if any(value.ndim != 1 for value in arrays) or len({value.shape for value in arrays}) != 1:
        raise ValueError("all sufficient-statistic arrays must be finite, aligned 1-D arrays")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("sufficient statistics must be finite")
    sx, sy, sx2, sy2, sxy = arrays
    nf = float(n)
    cov = sxy - sx * sy / nf
    var_x = sx2 - sx * sx / nf
    var_y = sy2 - sy * sy / nf
    tolerance_x = np.finfo(np.float64).eps * np.maximum(np.abs(sx2), np.abs(sx * sx / nf)) * 16
    tolerance_y = np.finfo(np.float64).eps * np.maximum(np.abs(sy2), np.abs(sy * sy / nf)) * 16
    var_x = np.where((var_x < 0) & (var_x >= -tolerance_x), 0.0, var_x)
    var_y = np.where((var_y < 0) & (var_y >= -tolerance_y), 0.0, var_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = cov / np.sqrt(var_x * var_y)
    result[(var_x <= 0) | (var_y <= 0)] = np.nan
    finite = np.isfinite(result)
    result[finite] = np.clip(result[finite], -1.0, 1.0)
    return result
