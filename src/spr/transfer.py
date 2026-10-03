"""Scientifically explicit cross-instrument SPR transfer analysis.

Cross-instrument scores are computed from predictions made by a model fitted on
source-instrument observations.  Raw within-target SPR correlations are not
transfer and are intentionally not used here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

_MODEL_FEATURES = {
    "bit_width_only": "bit_width",
    "signal_only": "signal",
    "perturbation_only": "perturbation",
    "spr": "spr",
}


@dataclass
class InstrumentData:
    """Aligned observations for one instrument.

    Targets should be normalized damage in ``[0, 1]`` for continuous transfer.
    ``groups`` identifies independent resampling units (for example prompts,
    probe cells/seeds, or SAE features), preventing token-level
    pseudo-replication.  Set ``target_kind`` to ``"binary"`` or ``"continuous"``
    to avoid inferring it from target values.
    """

    spr: np.ndarray
    target: np.ndarray
    bit_width: np.ndarray | float | None = None
    signal: np.ndarray | None = None
    perturbation: np.ndarray | None = None
    groups: np.ndarray | None = None
    target_kind: str | None = None
    # Unit-free SPR: this instrument's ``spr`` z-scored by its OWN mean/std, so
    # SPR is comparable across instruments that live on different absolute scales
    # (roadmap #2). Defaulted per-instrument in ``_coerce_data`` when not supplied.
    spr_normalized: np.ndarray | None = None


@dataclass
class PredictionMetrics:
    """Source-fitted predictive performance on a target instrument."""

    model: str
    n_source: int
    n_target: int
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    spearman: float | None = None
    log_loss: float | None = None
    brier: float | None = None
    calibration_error: float | None = None
    auc: float | None = None
    improvement_over_constant: float = 0.0
    confidence_interval: tuple[float, float] | None = None


@dataclass
class TransferResult:
    """Outcome of one ordered train-on-source/test-on-target SPR test."""

    source_instrument: str
    target_instrument: str
    metric: str
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    passes: bool
    model: str = "spr"
    n_source: int = 0
    n_target: int = 0
    metrics: PredictionMetrics | None = None
    baselines: dict[str, PredictionMetrics] = field(default_factory=dict)
    beats_constant_baseline: bool = False
    delta_over_null: float = 0.0


@dataclass
class LeaveOneInstrumentOutResult:
    """Shared-model result trained on every instrument except one."""

    held_out_instrument: str
    metrics: dict[str, PredictionMetrics]


@dataclass(frozen=True)
class TransferRunPlan:
    """Dry-run description of an SPR analysis without fitting any model."""

    mode: str
    output_directory: str
    scientifically_valid: bool
    instruments: tuple[str, ...]
    estimated_observations: dict[str, int]
    conditions: dict[str, tuple[float, ...]]
    expected_artifacts: tuple[str, ...]


def _finite_array(value, name: str, *, one_dimensional: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if one_dimensional:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _clip_damage(value: np.ndarray, name: str) -> np.ndarray:
    value = _finite_array(value, name)
    return np.clip(value, 0.0, 1.0)


def _zscore(value: np.ndarray) -> np.ndarray:
    """Standardize by the array's own mean/std (unit-free SPR, roadmap #2).

    A zero-variance array (all-equal SPR) yields all zeros rather than NaN, so
    the result is always finite. This is what makes the SPR calibration
    comparable across instruments on different absolute scales.
    """
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    std = float(value.std())
    if std <= 0:
        return np.zeros_like(value)
    return (value - float(value.mean())) / std


def normalize_sae_damage(correlation: np.ndarray) -> np.ndarray:
    """SAE damage: ``1 - clip(correlation, 0, 1)``.

    Negative correlations count as maximal damage; correlations above one are
    clipped as no damage.  Raw correlations should be retained by the caller.
    """
    return 1.0 - _clip_damage(correlation, "SAE correlation")


def normalize_probe_damage(
    auc_fp16: np.ndarray, auc_quant: np.ndarray, *, scale: float
) -> np.ndarray:
    """Probe damage: clipped positive AUC drop divided by preregistered ``scale``.

    Improvements under quantization have negative drops and are clipped to zero.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"probe damage scale must be finite and > 0, got {scale!r}")
    fp16 = _finite_array(auc_fp16, "FP16 AUC")
    quant = _finite_array(auc_quant, "quantized AUC")
    if fp16.shape != quant.shape:
        raise ValueError(f"FP16 and quantized AUC shapes differ: {fp16.shape} != {quant.shape}")
    return np.clip((fp16 - quant) / scale, 0.0, 1.0)


def normalize_circuit_damage(stability: np.ndarray) -> np.ndarray:
    """Circuit damage: ``1 - clip(stability_or_overlap, 0, 1)``."""
    return 1.0 - _clip_damage(stability, "circuit stability")


def normalize_patching_damage(
    fp16_effect: np.ndarray, quant_effect: np.ndarray, *, scale: float
) -> np.ndarray:
    """Patching damage: clipped absolute effect change divided by ``scale``.

    Absolute change is used because causal-effect direction is task-dependent;
    increases and decreases both represent departure from the FP16 effect.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"patching damage scale must be finite and > 0, got {scale!r}")
    fp16 = _finite_array(fp16_effect, "FP16 patching effect")
    quant = _finite_array(quant_effect, "quantized patching effect")
    if fp16.shape != quant.shape:
        raise ValueError(f"FP16 and quantized effect shapes differ: {fp16.shape} != {quant.shape}")
    return np.clip(np.abs(quant - fp16) / scale, 0.0, 1.0)


def _coerce_data(value: InstrumentData | tuple[np.ndarray, np.ndarray]) -> InstrumentData:
    if isinstance(value, InstrumentData):
        data = value
    else:
        try:
            spr, target = value
        except (TypeError, ValueError) as exc:
            raise TypeError("instrument data must be InstrumentData or an (spr, target) pair") from exc
        data = InstrumentData(spr=np.asarray(spr), target=np.asarray(target))

    spr = _finite_array(data.spr, "spr")
    target = _finite_array(data.target, "target")
    if spr.shape != target.shape:
        raise ValueError(f"spr and target shapes differ: {spr.shape} != {target.shape}")
    if data.target_kind not in (None, "binary", "continuous"):
        raise ValueError("target_kind must be 'binary', 'continuous', or None")
    groups = np.arange(target.size) if data.groups is None else np.asarray(data.groups).reshape(-1)
    if groups.shape != target.shape:
        raise ValueError(f"groups and target shapes differ: {groups.shape} != {target.shape}")

    def optional_feature(feature, name):
        if feature is None:
            return None
        array = np.asarray(feature, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(target.size, float(array))
        array = _finite_array(array, name)
        if array.shape != target.shape:
            raise ValueError(f"{name} and target shapes differ: {array.shape} != {target.shape}")
        return array

    # Per-instrument unit-free SPR: default to this instrument's own z-scored spr
    # when the caller did not supply one (roadmap #2). Computed from the coerced
    # (finite) spr so it is always finite and aligned with target.
    normalized = data.spr_normalized if data.spr_normalized is not None else _zscore(spr)
    return InstrumentData(
        spr=spr,
        target=target,
        bit_width=optional_feature(data.bit_width, "bit_width"),
        signal=optional_feature(data.signal, "signal"),
        perturbation=optional_feature(data.perturbation, "perturbation"),
        groups=groups,
        target_kind=data.target_kind,
        spr_normalized=optional_feature(normalized, "spr_normalized"),
    )


def _is_binary(data: InstrumentData) -> bool:
    if data.target_kind is not None:
        return data.target_kind == "binary"
    values = np.unique(data.target)
    return values.size <= 2 and np.all(np.isin(values, [0.0, 1.0]))


def _fit_predict(
    train_feature: np.ndarray | None,
    train_target: np.ndarray,
    test_feature: np.ndarray | None,
    *,
    binary: bool,
    seed: int = 0,
) -> np.ndarray:
    if train_feature is None or test_feature is None:
        return np.full(test_feature.size if test_feature is not None else 0, train_target.mean())
    x_train = _finite_array(train_feature, "source feature").reshape(-1, 1)
    x_test = _finite_array(test_feature, "target feature").reshape(-1, 1)
    if binary:
        if np.unique(train_target).size < 2:
            return np.full(x_test.shape[0], train_target.mean())
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(x_train, train_target)
        return model.predict_proba(x_test)[:, 1]
    model = LinearRegression()
    model.fit(x_train, train_target)
    return model.predict(x_test)


def _safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return 0.0
    rho, _ = spearmanr(prediction, target)
    return float(rho) if np.isfinite(rho) else 0.0


def calibration_error(target: np.ndarray, probability: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected calibration error with fixed-width probability bins."""
    target = _finite_array(target, "binary target")
    probability = np.clip(_finite_array(probability, "probability"), 0.0, 1.0)
    if target.shape != probability.shape:
        raise ValueError("target and probability must have the same shape")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bin_ids = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        if np.any(mask):
            error += mask.mean() * abs(probability[mask].mean() - target[mask].mean())
    return float(error)


def _score_predictions(
    model_name: str,
    source_target: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    binary: bool,
) -> PredictionMetrics:
    prediction = _finite_array(prediction, "prediction")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} != {target.shape}")
    metrics = PredictionMetrics(model=model_name, n_source=source_target.size, n_target=target.size)
    if binary:
        probability = np.clip(prediction, 1e-12, 1.0 - 1e-12)
        metrics.log_loss = float(log_loss(target, probability, labels=[0.0, 1.0]))
        metrics.brier = float(np.mean(np.square(probability - target)))
        metrics.calibration_error = calibration_error(target, probability)
        metrics.auc = 0.5 if np.unique(target).size < 2 else float(roc_auc_score(target, probability))
    else:
        metrics.mae = float(mean_absolute_error(target, prediction))
        metrics.rmse = float(np.sqrt(mean_squared_error(target, prediction)))
        metrics.r2 = float(r2_score(target, prediction)) if target.size > 1 else 0.0
        if not np.isfinite(metrics.r2):
            metrics.r2 = 0.0
        metrics.spearman = _safe_spearman(prediction, target)
    return metrics


def grouped_bootstrap_ci(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a metric by independent groups, retaining all rows per group."""
    target = _finite_array(target, "target")
    prediction = _finite_array(prediction, "prediction")
    groups = np.asarray(groups).reshape(-1)
    if target.shape != prediction.shape or target.shape != groups.shape:
        raise ValueError("target, prediction, and groups must have identical shapes")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("grouped bootstrap requires at least two independent groups")
    indices_by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, unique_groups.size, unique_groups.size)
        rows = np.concatenate([indices_by_group[group_index] for group_index in sampled])
        scores[index] = metric_fn(target[rows], prediction[rows])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(scores, [alpha, 1.0 - alpha])
    return float(low), float(high)


def evaluate_transfer_models(
    source: InstrumentData,
    target: InstrumentData,
    *,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, PredictionMetrics]:
    """Fit constant/factor baselines and the main SPR model on source only.

    All non-constant model parameters are learned from the source observations
    and applied unchanged to the target.  Missing baseline features are rejected
    rather than silently substituting SPR or another proxy.
    """
    source = _coerce_data(source)
    target = _coerce_data(target)
    binary = _is_binary(source) and _is_binary(target)
    results: dict[str, PredictionMetrics] = {}
    predictions: dict[str, np.ndarray] = {}
    predictions["constant"] = np.full(target.target.size, source.target.mean())
    for model_name, feature_name in _MODEL_FEATURES.items():
        source_feature = getattr(source, feature_name)
        target_feature = getattr(target, feature_name)
        if source_feature is None or target_feature is None:
            raise ValueError(
                f"{model_name} requires {feature_name} for both source and target instruments"
            )
        predictions[model_name] = _fit_predict(
            source_feature, source.target, target_feature, binary=binary, seed=seed
        )

    for model_name, prediction in predictions.items():
        results[model_name] = _score_predictions(
            model_name, source.target, target.target, prediction, binary=binary
        )
    constant = results["constant"]
    for metrics in results.values():
        if binary:
            metrics.improvement_over_constant = float(constant.log_loss - metrics.log_loss)
        else:
            metrics.improvement_over_constant = float(constant.rmse - metrics.rmse)
        # A grouped bootstrap needs >= 2 independent groups. Some instruments are
        # single-group by construction (e.g. patching: one causal component -> one
        # group), so the CI is genuinely undefined for them; leave it None rather
        # than crashing the whole leave-one-instrument-out run. This mirrors the
        # guard already used for the within-instrument bootstrap.
        if n_bootstrap and np.unique(target.groups).size >= 2:
            if binary:
                metric_fn = lambda y, p: float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0.0, 1.0]))
            else:
                metric_fn = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
            metrics.confidence_interval = grouped_bootstrap_ci(
                target.target,
                predictions[metrics.model],
                target.groups,
                metric_fn,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
    return results


def fit_logistic_auc(
    train_spr: np.ndarray,
    train_target: np.ndarray,
    test_spr: np.ndarray,
    test_target: np.ndarray,
    seed: int,
) -> float:
    """AUC from a logistic model fitted only on source-instrument data."""
    train_spr = _finite_array(train_spr, "train_spr")
    train_target = _finite_array(train_target, "train_target")
    test_spr = _finite_array(test_spr, "test_spr")
    test_target = _finite_array(test_target, "test_target")
    prediction = _fit_predict(
        train_spr, train_target, test_spr, binary=True, seed=seed
    )
    if np.unique(test_target).size < 2:
        return 0.5
    return float(roc_auc_score(test_target, prediction))


def spearman_transfer(train_spr, train_target, test_spr, test_target) -> float:
    """Spearman of target outcomes against source-fitted regression predictions.

    Secondary diagnostic only: rank correlation is insensitive
    to calibration, so it must not be the pass/fail statistic.
    """
    train_spr = _finite_array(train_spr, "train_spr")
    train_target = _finite_array(train_target, "train_target")
    test_spr = _finite_array(test_spr, "test_spr")
    test_target = _finite_array(test_target, "test_target")
    prediction = _fit_predict(train_spr, train_target, test_spr, binary=False)
    return _safe_spearman(prediction, test_target)


def neg_log_loss_transfer(
    train_spr: np.ndarray,
    train_target: np.ndarray,
    test_spr: np.ndarray,
    test_target: np.ndarray,
    seed: int,
) -> float:
    """Preregistered primary statistic for binary targets: negative log loss.

    Probabilities are fitted on the source instrument only and scored on the
    target. Log loss is calibration-sensitive: a source whose
    ranking transfers but whose prevalence/calibration does not will fail this
    statistic even when target AUC is high. Higher (less negative) is better.
    """
    train_spr = _finite_array(train_spr, "train_spr")
    train_target = _finite_array(train_target, "train_target")
    test_spr = _finite_array(test_spr, "test_spr")
    test_target = _finite_array(test_target, "test_target")
    prediction = _fit_predict(train_spr, train_target, test_spr, binary=True, seed=seed)
    probability = np.clip(prediction, 1e-12, 1.0 - 1e-12)
    return -float(log_loss(test_target, probability, labels=[0.0, 1.0]))


def neg_rmse_transfer(
    train_spr: np.ndarray,
    train_target: np.ndarray,
    test_spr: np.ndarray,
    test_target: np.ndarray,
) -> float:
    """Preregistered primary statistic for continuous targets: negative RMSE.

    The regression is fitted on the source instrument only and scored on the
    target. RMSE is an error metric in the target's units, unlike Spearman,
    which ignores scale and calibration. Higher (less negative) is better.
    """
    train_spr = _finite_array(train_spr, "train_spr")
    train_target = _finite_array(train_target, "train_target")
    test_spr = _finite_array(test_spr, "test_spr")
    test_target = _finite_array(test_target, "test_target")
    prediction = _fit_predict(train_spr, train_target, test_spr, binary=False)
    return -float(np.sqrt(mean_squared_error(test_target, prediction)))


def null_baseline(
    train_spr: np.ndarray,
    train_target: np.ndarray,
    test_spr: np.ndarray,
    test_target: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    *,
    n_shuffles: int,
    seed: int,
) -> tuple[float, float]:
    """Permutation null matching the source-fit/target-score statistic."""
    if n_shuffles <= 0:
        raise ValueError("n_shuffles must be positive")
    rng = np.random.default_rng(seed)
    test_target = _finite_array(test_target, "test_target")
    scores = np.empty(n_shuffles)
    for index in range(n_shuffles):
        scores[index] = metric_fn(
            train_spr, train_target, test_spr, rng.permutation(test_target)
        )
    return float(scores.mean()), float(scores.std())


def passes_margin(
    observed: float,
    null_mean: float,
    null_std: float,
    margin: float,
    z_threshold: float = 2.0,
) -> bool:
    """Require both a fixed improvement margin and a shuffle-null z threshold."""
    if null_std <= 0:
        return observed > null_mean + margin
    z = (observed - null_mean) / null_std
    return observed > null_mean + margin and z >= z_threshold


def run_transfer_tests(
    instruments: Mapping[str, InstrumentData | tuple[np.ndarray, np.ndarray]],
    *,
    margin: float,
    n_null_shuffles: int,
    cv_folds: int,
    seed: int,
    z_threshold: float = 2.0,
    n_bootstrap: int = 0,
    spr_feature: str = "spr",
) -> list[TransferResult]:
    """Run ordered pairwise source-fitted SPR transfer tests.

    ``spr_feature`` selects the primary SPR array: ``"spr"`` (default, the raw
    instrument SPR — unchanged behavior) or ``"spr_normalized"`` (the per-instrument
    z-scored, unit-free SPR, roadmap #2). Only the primary observed/null/passes use
    it; the baseline models are unchanged, so the two tables are directly comparable.

    The preregistered primary statistic is calibration/error based: negative log loss for binary target pairs, negative RMSE for
    continuous target pairs. AUC and Spearman remain available as secondary
    diagnostics in ``metrics`` but never decide ``passes``. ``passes`` requires
    BOTH the margin/z criterion against the permutation null AND that the
    source-fit model beats the source-prevalence/source-mean constant baseline
    on the target (a ranking-perfect but miscalibrated model fails this).

    ``cv_folds`` remains accepted for configuration/API compatibility.  Rich
    baseline metrics are populated when callers supply :class:`InstrumentData`
    with bit-width, signal, and perturbation features; legacy ``(spr, target)``
    pairs still run the corrected main SPR model.
    """
    if cv_folds <= 0:
        raise ValueError("cv_folds must be positive")
    data = {name: _coerce_data(value) for name, value in instruments.items()}
    results: list[TransferResult] = []
    for source_name, source in data.items():
        for target_name, target in data.items():
            if source_name == target_name:
                continue
            binary = _is_binary(source) and _is_binary(target)
            if binary:
                def metric_fn(a, b, c, d, _seed=seed):
                    return neg_log_loss_transfer(a, b, c, d, _seed)
                metric_name = "neg_log_loss"
            else:
                metric_fn = neg_rmse_transfer
                metric_name = "neg_rmse"
            src_primary = getattr(source, spr_feature)
            tgt_primary = getattr(target, spr_feature)
            if src_primary is None or tgt_primary is None:
                raise ValueError(f"transfer requires {spr_feature} for both instruments")
            observed = metric_fn(src_primary, source.target, tgt_primary, target.target)
            null_mean, null_std = null_baseline(
                src_primary,
                source.target,
                tgt_primary,
                target.target,
                metric_fn,
                n_shuffles=n_null_shuffles,
                seed=seed,
            )
            z_score = float("inf") if null_std <= 0 else (observed - null_mean) / null_std

            prediction = _fit_predict(
                src_primary, source.target, tgt_primary, binary=binary, seed=seed
            )
            main_metrics = _score_predictions(
                spr_feature, source.target, target.target, prediction, binary=binary
            )
            # Absolute calibration/error requirement: beating a
            # permutation null is not enough, because a model with perfect target
            # ranking but source-mismatched calibration beats shuffled targets
            # while remaining useless. The source-fit model must also beat the
            # source-prevalence/source-mean constant baseline on the target.
            if binary:
                constant_probability = np.clip(float(source.target.mean()), 1e-12, 1.0 - 1e-12)
                constant_loss = float(
                    log_loss(
                        target.target,
                        np.full(target.target.size, constant_probability),
                        labels=[0.0, 1.0],
                    )
                )
                beats_constant = bool(main_metrics.log_loss < constant_loss)
            else:
                constant_rmse = float(
                    np.sqrt(mean_squared_error(target.target, np.full(target.target.size, source.target.mean())))
                )
                beats_constant = bool(main_metrics.rmse < constant_rmse)
            baselines: dict[str, PredictionMetrics] = {}
            complete = all(
                getattr(source, feature) is not None and getattr(target, feature) is not None
                for feature in ("bit_width", "signal", "perturbation")
            )
            if complete:
                all_metrics = evaluate_transfer_models(
                    source, target, n_bootstrap=n_bootstrap, seed=seed
                )
                if spr_feature == "spr":
                    main_metrics = all_metrics.pop("spr")
                    baselines = all_metrics
                else:
                    # Keep the normalized primary metrics; leave the raw "spr"
                    # model in baselines as a direct reference alongside it.
                    baselines = all_metrics
            results.append(
                TransferResult(
                    source_instrument=source_name,
                    target_instrument=target_name,
                    metric=metric_name,
                    observed=observed,
                    null_mean=null_mean,
                    null_std=null_std,
                    z_score=float(z_score),
                    passes=passes_margin(observed, null_mean, null_std, margin, z_threshold)
                    and beats_constant,
                    n_source=source.target.size,
                    n_target=target.target.size,
                    metrics=main_metrics,
                    baselines=baselines,
                    beats_constant_baseline=beats_constant,
                    delta_over_null=float(observed - null_mean),
                )
            )
    return results


def leave_one_instrument_out(
    instruments: Mapping[str, InstrumentData],
    *,
    n_bootstrap: int = 0,
    seed: int = 0,
    spr_feature: str = "spr",
) -> list[LeaveOneInstrumentOutResult]:
    """Fit shared models on N-1 instruments and test the held-out instrument.

    ``spr_feature`` selects which SPR array drives the shared "spr" model:
    ``"spr"`` (default, raw) or ``"spr_normalized"`` (per-instrument z-scored,
    roadmap #2). Because each instrument is z-scored by its OWN stats before the
    N-1 concatenation, the shared model and the held-out target end up on a common
    unit-free scale; the other baseline models are untouched.
    """
    if len(instruments) < 2:
        raise ValueError("leave-one-instrument-out requires at least two instruments")
    data = {name: _coerce_data(value) for name, value in instruments.items()}
    output: list[LeaveOneInstrumentOutResult] = []
    for held_out_name, held_out in data.items():
        training = [value for name, value in data.items() if name != held_out_name]
        kinds = {_is_binary(value) for value in training + [held_out]}
        if len(kinds) != 1:
            raise ValueError("leave-one-instrument-out requires a common binary or continuous target type")

        def concatenate(feature: str, training=training):
            values = [getattr(value, feature) for value in training]
            if any(value is None for value in values):
                raise ValueError(f"leave-one-instrument-out requires {feature} for every instrument")
            return np.concatenate(values)

        shared = InstrumentData(
            spr=concatenate(spr_feature),
            target=concatenate("target"),
            bit_width=concatenate("bit_width"),
            signal=concatenate("signal"),
            perturbation=concatenate("perturbation"),
            groups=np.arange(sum(value.target.size for value in training)),
            target_kind="binary" if _is_binary(held_out) else "continuous",
        )
        test_instrument = (
            held_out if spr_feature == "spr"
            else replace(held_out, spr=getattr(held_out, spr_feature))
        )
        output.append(
            LeaveOneInstrumentOutResult(
                held_out_instrument=held_out_name,
                metrics=evaluate_transfer_models(
                    shared, test_instrument, n_bootstrap=n_bootstrap, seed=seed
                ),
            )
        )
    return output


def build_transfer_run_plan(
    instruments: Mapping[str, InstrumentData | tuple[np.ndarray, np.ndarray]],
    *,
    smoke_test: bool = False,
) -> TransferRunPlan:
    """Build a no-fit dry-run plan for pairwise and shared SPR artifacts.

    Smoke plans are always rooted under ``results/smoke/spr`` and explicitly
    marked scientifically invalid, preventing accidental use as paper results.
    """
    data = {name: _coerce_data(value) for name, value in instruments.items()}
    conditions: dict[str, tuple[float, ...]] = {}
    for name, value in data.items():
        conditions[name] = (
            tuple(float(item) for item in np.unique(value.bit_width))
            if value.bit_width is not None
            else ()
        )
    return TransferRunPlan(
        mode="smoke" if smoke_test else "harmonized",
        output_directory="results/smoke/spr" if smoke_test else "results/harmonized/spr",
        scientifically_valid=not smoke_test,
        instruments=tuple(data),
        estimated_observations={name: value.target.size for name, value in data.items()},
        conditions=conditions,
        expected_artifacts=(
            "within_instrument.csv",
            "pairwise_transfer.csv",
            "leave_one_instrument_out.csv",
        ),
    )


def make_smoke_test_data(
    instruments: Mapping[str, InstrumentData],
    *,
    max_groups: int = 4,
    max_bit_widths: int = 2,
) -> dict[str, InstrumentData]:
    """Select a deterministic, group-preserving non-scientific smoke subset."""
    if max_groups <= 0 or max_bit_widths <= 0:
        raise ValueError("smoke-test limits must be positive")
    smoke: dict[str, InstrumentData] = {}
    for name, raw in instruments.items():
        data = _coerce_data(raw)
        selected_groups = np.unique(data.groups)[:max_groups]
        mask = np.isin(data.groups, selected_groups)
        if data.bit_width is not None:
            selected_bits = np.unique(data.bit_width)[:max_bit_widths]
            mask &= np.isin(data.bit_width, selected_bits)
        if not np.any(mask):
            raise ValueError(f"smoke-test selection produced no observations for {name!r}")

        def subset(feature: str, data=data, mask=mask):
            value = getattr(data, feature)
            return None if value is None else value[mask]

        smoke[name] = InstrumentData(
            spr=data.spr[mask],
            target=data.target[mask],
            bit_width=subset("bit_width"),
            signal=subset("signal"),
            perturbation=subset("perturbation"),
            groups=data.groups[mask],
            target_kind=data.target_kind,
        )
    return smoke


def write_transfer_json(results: list[TransferResult], path: Path) -> None:
    """Persist pairwise transfer results and predictive metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump({"results": [asdict(result) for result in results]}, file_handle, indent=2)
