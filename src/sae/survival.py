"""SAE survival taxonomy and active-feature gate.

Matches Duan, E. (2026). *Perplexity Can Miss SAE Feature Damage Under
Quantization.* arXiv:2606.03002, §2.2. The survival taxonomy is reused exactly, never redefined:

- Survived:  ``c_j > 0.9``
- Degraded:  ``0.5 <= c_j <= 0.9``
- Damaged:   ``c_j < 0.5``
- Only features active in FP16 (firing rate ``f_j > 0.001``) are scored.

Pure numpy logic; unit-testable without a GPU or SAE.
"""

from __future__ import annotations

import numpy as np

SURVIVED = "survived"
DEGRADED = "degraded"
DAMAGED = "damaged"


def active_mask(firing_rates: np.ndarray, active_firing_rate_min: float) -> np.ndarray:
    """Boolean mask of features active in FP16: ``f_j > active_firing_rate_min``.

    Docstring reference: Duan 2606.03002 §2.2 ("Only features ... firing rate
    f_j > 0.001 are scored").
    """
    firing_rates = np.asarray(firing_rates)
    return firing_rates > active_firing_rate_min


def classify_survival(
    corr: np.ndarray,
    survived_corr_min: float,
    degraded_corr_min: float,
) -> np.ndarray:
    """Classify each feature into survived/degraded/damaged by its correlation.

    Args:
        corr: per-feature correlations ``c_j`` (may contain NaN for never-firing
            features; those map to the falsey "damaged" branch but are only
            meaningful after the active gate is applied).
        survived_corr_min: upper cut, 0.9. ``c_j > this`` => survived.
        degraded_corr_min: lower cut, 0.5. ``this <= c_j <= survived_corr_min``
            => degraded; ``c_j < this`` (or non-finite) => damaged.

    Returns:
        Array of ``"survived"`` / ``"degraded"`` / ``"damaged"`` per feature.

    Docstring reference: Duan 2606.03002 §2.2 taxonomy.
    """
    corr = np.asarray(corr)
    out = np.empty(corr.shape[0], dtype=object)
    nan_or_below = ~np.isfinite(corr) | (corr < degraded_corr_min)
    out[nan_or_below] = DAMAGED
    out[corr > survived_corr_min] = SURVIVED
    between = np.isfinite(corr) & (corr >= degraded_corr_min) & (corr <= survived_corr_min)
    out[between] = DEGRADED
    return out


def survival_rates(
    corr: np.ndarray,
    firing_rates: np.ndarray,
    active_firing_rate_min: float,
    survived_corr_min: float,
    degraded_corr_min: float,
) -> dict[str, float]:
    """Report survived/degraded/damaged percentages over the *active* features only.

    Matches Duan 2606.03002 Table 1 columns "Survived % / Degraded % / Damaged %":
    each is the fraction of active features in that class, expressed as a
    percentage. Non-active features are excluded from the denominator
    (2606.03002 gate).
    """
    corr = np.asarray(corr)
    firing_rates = np.asarray(firing_rates)
    active = active_mask(firing_rates, active_firing_rate_min)
    active_corr = corr[active]
    labels = classify_survival(active_corr, survived_corr_min, degraded_corr_min)
    n_active = int(active.sum())
    if n_active == 0:
        return {SURVIVED: 0.0, DEGRADED: 0.0, DAMAGED: 0.0}
    counts = {label: int((labels == label).sum()) for label in (SURVIVED, DEGRADED, DAMAGED)}
    return {label: 100.0 * count / n_active for label, count in counts.items()}