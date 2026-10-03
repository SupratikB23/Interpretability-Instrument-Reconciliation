"""Numerically safe signal-to-perturbation-ratio (SPR) estimation.

The preregistered primary estimator is::

    peak_abs_projection / mean_abs_directional_delta

Readout directions are L2-normalized before projection.  The absolute-value
conventions make the estimate invariant to the arbitrary sign of a direction.
Activations are consumed in aligned chunks and are never materialized as one
large token-by-hidden-dimension matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest

import numpy as np

_SIGNAL_ALIASES = {
    "mean": "mean_abs_projection",
    "peak": "peak_abs_projection",
    "rms": "rms_projection",
    "mean_abs_projection": "mean_abs_projection",
    "peak_abs_projection": "peak_abs_projection",
    "rms_projection": "rms_projection",
}
_DEFAULT_SIGNAL = "peak_abs_projection"
_DEFAULT_EPS = 1e-8
_MISSING = object()


def _normalized_directions(directions: np.ndarray, eps: float) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2:
        raise ValueError(f"directions must have shape [K, D], got {directions.shape}")
    if directions.shape[0] == 0 or directions.shape[1] == 0:
        raise ValueError(f"directions must be non-empty, got {directions.shape}")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions contain non-finite values")
    norms = np.linalg.norm(directions, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= eps):
        bad = np.flatnonzero(~np.isfinite(norms) | (norms <= eps)).tolist()
        raise ValueError(f"directions must have finite L2 norm greater than eps={eps}; bad rows: {bad}")
    return directions / norms[:, None]


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and > 0, got {eps!r}")
    return eps


def estimate_spr_streaming(
    directions: np.ndarray,
    fp16_chunks: Iterable[np.ndarray],
    quant_chunks: Iterable[np.ndarray],
    signal_stat: str = _DEFAULT_SIGNAL,
    *,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """Estimate SPR for each readout direction from aligned activation chunks.

    ``signal_stat`` accepts ``peak_abs_projection`` (the primary),
    ``mean_abs_projection``, or ``rms_projection``.  The legacy aliases
    ``peak``, ``mean``, and ``rms`` are retained.  The perturbation statistic is
    always ``mean_abs_directional_delta``.  Denominators are floored at ``eps``
    so the returned values are finite even for exact FP16 reproduction.

    Raises:
        ValueError: for invalid directions, empty or differently sized streams,
            non-2D/empty chunks, non-finite activations, or shape misalignment.
    """
    eps = _validate_eps(eps)
    normalized = _normalized_directions(directions, eps)
    try:
        statistic = _SIGNAL_ALIASES[signal_stat]
    except KeyError as exc:
        choices = ", ".join(sorted(set(_SIGNAL_ALIASES.values())))
        raise ValueError(f"signal_stat must be one of {choices}, got {signal_stat!r}") from exc

    k, hidden_dim = normalized.shape
    sum_abs_delta = np.zeros(k, dtype=np.float64)
    sum_abs_signal = np.zeros(k, dtype=np.float64)
    sum_sq_signal = np.zeros(k, dtype=np.float64)
    peak_signal = np.zeros(k, dtype=np.float64)
    n_tokens = 0

    for chunk_index, (fp16, quant) in enumerate(
        zip_longest(fp16_chunks, quant_chunks, fillvalue=_MISSING)
    ):
        if fp16 is _MISSING or quant is _MISSING:
            raise ValueError(
                "fp16_chunks and quant_chunks contain different numbers of aligned chunks"
            )
        fp16 = np.asarray(fp16, dtype=np.float64)
        quant = np.asarray(quant, dtype=np.float64)
        if fp16.ndim != 2 or quant.ndim != 2:
            raise ValueError(
                f"chunk {chunk_index} must be 2D [B, D], got {fp16.shape} and {quant.shape}"
            )
        if fp16.shape != quant.shape:
            raise ValueError(
                f"chunk {chunk_index} is misaligned: FP16 {fp16.shape} != quantized {quant.shape}"
            )
        if fp16.shape[0] == 0:
            raise ValueError(f"chunk {chunk_index} is empty")
        if fp16.shape[1] != hidden_dim:
            raise ValueError(
                f"chunk {chunk_index} hidden size {fp16.shape[1]} does not match direction size {hidden_dim}"
            )
        if not np.all(np.isfinite(fp16)) or not np.all(np.isfinite(quant)):
            raise ValueError(f"chunk {chunk_index} contains non-finite activations")

        projection = fp16 @ normalized.T
        directional_delta = (quant - fp16) @ normalized.T
        sum_abs_delta += np.abs(directional_delta).sum(axis=0)
        if statistic == "mean_abs_projection":
            sum_abs_signal += np.abs(projection).sum(axis=0)
        elif statistic == "peak_abs_projection":
            np.maximum(peak_signal, np.abs(projection).max(axis=0), out=peak_signal)
        else:
            sum_sq_signal += np.square(projection).sum(axis=0)
        n_tokens += fp16.shape[0]

    if n_tokens == 0:
        raise ValueError("activation chunk streams are empty")

    perturbation = sum_abs_delta / n_tokens
    if statistic == "mean_abs_projection":
        signal = sum_abs_signal / n_tokens
    elif statistic == "peak_abs_projection":
        signal = peak_signal
    else:
        signal = np.sqrt(sum_sq_signal / n_tokens)
    return estimate_spr_from_stats(signal, perturbation, eps=eps)


def estimate_spr_from_stats(
    signal: np.ndarray,
    mean_abs_perturbation: np.ndarray,
    *,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """Compute finite SPR from aligned non-negative signal/perturbation stats."""
    eps = _validate_eps(eps)
    signal = np.asarray(signal, dtype=np.float64)
    perturbation = np.asarray(mean_abs_perturbation, dtype=np.float64)
    if signal.shape != perturbation.shape:
        raise ValueError(f"signal and perturbation shapes differ: {signal.shape} != {perturbation.shape}")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(perturbation)):
        raise ValueError("signal and perturbation statistics must be finite")
    if np.any(signal < 0) or np.any(perturbation < 0):
        raise ValueError("signal and mean absolute perturbation must be non-negative")
    result = signal / np.maximum(perturbation, eps)
    return np.nan_to_num(result, nan=0.0, posinf=np.finfo(np.float64).max)


def cosine_alignment(
    directions_a: np.ndarray,
    directions_b: np.ndarray,
    *,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """Return the cosine-similarity matrix for two valid direction sets."""
    eps = _validate_eps(eps)
    a = _normalized_directions(directions_a, eps)
    b = _normalized_directions(directions_b, eps)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"direction hidden sizes differ: {a.shape[1]} != {b.shape[1]}")
    return a @ b.T
