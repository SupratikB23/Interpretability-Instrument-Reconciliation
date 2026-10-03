"""Unit tests for the cross-instrument SPR transfer test."""

import numpy as np
import pytest

from src.spr.transfer import (
    InstrumentData,
    build_transfer_run_plan,
    evaluate_transfer_models,
    grouped_bootstrap_ci,
    leave_one_instrument_out,
    make_smoke_test_data,
    normalize_circuit_damage,
    normalize_patching_damage,
    normalize_probe_damage,
    normalize_sae_damage,
    null_baseline,
    passes_margin,
    run_transfer_tests,
    spearman_transfer,
)


def _shared_spr_instruments(seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.gamma(2.0, 1.0, 400)
    spr_sae = shared + 0.1 * rng.normal(0, 1, 400)
    spr_probe = shared + 0.1 * rng.normal(0, 1, 400)
    p_sae = 1 / (1 + np.exp(-(spr_sae - 2.0)))
    p_probe = 1 / (1 + np.exp(-(spr_probe - 2.0)))
    return {
        "sae": (spr_sae, (rng.uniform(0, 1, 400) < p_sae).astype(int)),
        "probe": (spr_probe, (rng.uniform(0, 1, 400) < p_probe).astype(int)),
    }


def test_coerce_defaults_unit_free_spr_per_instrument():
    from src.spr.transfer import _coerce_data
    data = _coerce_data(InstrumentData(spr=np.array([1.0, 2.0, 3.0, 4.0]), target=np.array([0.0, 0.3, 0.6, 0.9])))
    assert data.spr_normalized is not None
    assert abs(float(data.spr_normalized.mean())) < 1e-9
    assert abs(float(data.spr_normalized.std()) - 1.0) < 1e-9
    # Constant SPR -> all zeros, never NaN.
    flat = _coerce_data(InstrumentData(spr=np.full(4, 5.0), target=np.array([0.0, 1.0, 0.0, 1.0])))
    assert np.all(flat.spr_normalized == 0.0)


def test_normalized_spr_fixes_scale_mismatch_transfer():
    # Two instruments with the SAME monotone spr->damage law but spr on very
    # different absolute scales. Raw SPR transfer mis-calibrates (slope fit on the
    # source scale misapplied to the target); the per-instrument z-scored SPR is
    # scale-free and transfers. This is exactly roadmap #2.
    n = 60
    lin = np.linspace(0.0, 1.0, n)
    instruments = {
        "a": InstrumentData(spr=1.0 + lin, target=lin.copy(), target_kind="continuous"),
        "b": InstrumentData(spr=1000.0 + 1000.0 * lin, target=lin.copy(), target_kind="continuous"),
    }
    raw = {(r.source_instrument, r.target_instrument): r
           for r in run_transfer_tests(instruments, margin=0.05, n_null_shuffles=50, cv_folds=5, seed=0)}
    norm = {(r.source_instrument, r.target_instrument): r
            for r in run_transfer_tests(instruments, margin=0.05, n_null_shuffles=50, cv_folds=5, seed=0,
                                        spr_feature="spr_normalized")}
    # Raw SPR fails to beat the constant baseline on the scale-mismatched pair.
    assert not raw[("a", "b")].beats_constant_baseline
    # Unit-free SPR recovers calibrated transfer: beats constant and far lower RMSE.
    assert norm[("a", "b")].beats_constant_baseline
    assert norm[("a", "b")].metrics.rmse < raw[("a", "b")].metrics.rmse
    assert norm[("a", "b")].metrics.model == "spr_normalized"


def test_normalized_loio_runs_and_labels_spr_rows():
    n = 40
    lin = np.linspace(0.0, 1.0, n)
    instruments = {
        "a": InstrumentData(spr=1.0 + lin, target=lin.copy(), bit_width=np.full(n, 6),
                            signal=lin.copy(), perturbation=1.0 - lin, target_kind="continuous"),
        "b": InstrumentData(spr=1000.0 + 1000.0 * lin, target=lin.copy(), bit_width=np.full(n, 6),
                            signal=lin.copy(), perturbation=1.0 - lin, target_kind="continuous"),
        "c": InstrumentData(spr=50.0 + 5.0 * lin, target=lin.copy(), bit_width=np.full(n, 6),
                            signal=lin.copy(), perturbation=1.0 - lin, target_kind="continuous"),
    }
    out = leave_one_instrument_out(instruments, spr_feature="spr_normalized")
    assert {r.held_out_instrument for r in out} == {"a", "b", "c"}
    assert all("spr" in r.metrics for r in out)


def test_transfer_passes_when_spr_shared():
    instruments = _shared_spr_instruments()
    results = run_transfer_tests(instruments, margin=0.05, n_null_shuffles=100, cv_folds=5, seed=0)
    assert len(results) == 2  # sae->probe and probe->sae
    assert all(r.passes for r in results)


def test_transfer_fails_when_spr_unrelated():
    rng = np.random.default_rng(5)
    spr_a = rng.normal(0, 1, 300)
    spr_b = rng.normal(0, 1, 300)
    p = 1 / (1 + np.exp(-0.1 * spr_a))
    instruments = {
        "a": (spr_a, (rng.uniform(0, 1, 300) < p).astype(int)),
        "b": (spr_b, (rng.uniform(0, 1, 300) < p).astype(int)),
    }
    results = run_transfer_tests(instruments, margin=0.05, n_null_shuffles=100, cv_folds=5, seed=0)
    assert not any(r.passes for r in results)


def test_margin_requires_both_criteria():
    assert passes_margin(0.60, 0.50, 0.01, margin=0.05)  # margin met + z=10
    assert not passes_margin(0.51, 0.50, 0.001, margin=0.05)  # margin not met
    assert not passes_margin(0.60, 0.50, 0.10, margin=0.05)  # z=1.0 < 2 -> fails


def test_spearman_transfer_on_monotone_data():
    rng = np.random.default_rng(6)
    spr = np.linspace(0.1, 5.0, 200) + rng.normal(0, 0.1, 200)
    outcome = spr + rng.normal(0, 0.2, 200)  # perfectly monotone
    rho = spearman_transfer(spr, outcome, spr, outcome)
    assert rho > 0.9

def test_spearman_transfer_scores_the_target_not_the_source():
    """Regression: transfer must be measured on the held-out target instrument.

    Correlating the SOURCE's own spr-vs-outcome would make every transfer result
    circular — a source that fits itself perfectly would "transfer" perfectly to
    an unrelated target.
    """
    rng = np.random.default_rng(11)
    n = 300
    src_spr = np.linspace(0.1, 5.0, n)
    src_outcome = src_spr.copy()                 # source fits itself perfectly
    tgt_spr = rng.normal(0, 1, n)
    tgt_outcome = rng.normal(0, 1, n)            # target is pure noise

    rho = spearman_transfer(src_spr, src_outcome, tgt_spr, tgt_outcome)
    assert abs(rho) < 0.3, "transfer score leaked the source's in-sample fit"


def test_mixed_binary_and_continuous_instruments_do_not_crash():
    """Regression: metric choice follows the TARGET instrument's outcome type.

    Selecting it from the source fed a continuous target into roc_auc_score.
    """
    rng = np.random.default_rng(12)
    n = 300
    shared = rng.gamma(2.0, 1.0, n)
    binary = (shared > np.median(shared)).astype(int)
    continuous = shared + rng.normal(0, 0.3, n)

    results = run_transfer_tests(
        {"sae": (shared, binary), "probe": (shared, continuous)},
        margin=0.05, n_null_shuffles=50, cv_folds=5, seed=0,
    )
    assert len(results) == 2
    assert all(np.isfinite(r.observed) for r in results)
    # Any pair touching the continuous instrument must use the error metric.
    assert {r.metric for r in results} == {"neg_rmse"}


def test_primary_metric_is_calibration_sensitive_not_auc():
    """Regression: the primary statistic must not be AUC.

    The target preserves the source ranking (target AUC ~ 1) but the source-fit
    probabilities are badly calibrated for the target prevalence. An AUC-based
    gate would declare transfer; the preregistered negative-log-loss statistic
    must refuse it.
    """
    rng = np.random.default_rng(4)
    n = 400
    src_spr = rng.normal(0, 1, n)
    src_target = (src_spr > -1.5).astype(int)   # ~93% prevalence
    tgt_spr = rng.normal(0, 1, n)
    tgt_target = (tgt_spr > 1.5).astype(int)    # ~7% prevalence, same direction

    results = run_transfer_tests(
        {"a": (src_spr, src_target), "b": (tgt_spr, tgt_target)},
        margin=0.05, n_null_shuffles=100, cv_folds=5, seed=0,
    )
    assert all(r.metric == "neg_log_loss" for r in results)
    for result in results:
        assert result.metrics.auc > 0.9, "ranking transfers perfectly..."
        assert not result.beats_constant_baseline
        assert not result.passes, "...but calibration does not, so the gate must fail"


def test_null_baseline_mirrors_the_observed_statistic():
    """Regression: the null permutes the TARGET outcomes, not the source's.

    An unrelated target must land the observed value inside its own null.
    """
    rng = np.random.default_rng(13)
    n = 400
    src_spr = rng.gamma(2.0, 1.0, n)
    src_outcome = (src_spr > np.median(src_spr)).astype(int)
    tgt_spr = rng.gamma(2.0, 1.0, n)
    tgt_outcome = rng.integers(0, 2, n)          # unrelated to tgt_spr

    def metric_fn(a, b, c, d):
        from src.spr.transfer import fit_logistic_auc
        return fit_logistic_auc(a, b, c, d, 0)

    observed = metric_fn(src_spr, src_outcome, tgt_spr, tgt_outcome)
    null_mean, null_std = null_baseline(
        src_spr, src_outcome, tgt_spr, tgt_outcome, metric_fn, n_shuffles=100, seed=0
    )
    assert abs(observed - null_mean) < 4 * max(null_std, 1e-6)


def test_continuous_transfer_uses_the_source_fitted_regression():
    test_spr = np.arange(1.0, 6.0)
    test_target = test_spr.copy()
    positive = spearman_transfer(
        np.arange(1.0, 6.0), np.arange(1.0, 6.0), test_spr, test_target
    )
    negative = spearman_transfer(
        np.arange(1.0, 6.0), np.arange(5.0, 0.0, -1.0), test_spr, test_target
    )
    assert positive == pytest.approx(1.0)
    assert negative == pytest.approx(-1.0)


def test_normalized_damage_helpers_have_explicit_clipping_and_signs():
    assert np.allclose(normalize_sae_damage(np.array([-0.2, 0.4, 1.2])), [1.0, 0.6, 0.0])
    assert np.allclose(
        normalize_probe_damage(
            np.array([0.9, 0.7, 0.8]), np.array([0.7, 0.8, 0.0]), scale=0.2
        ),
        [1.0, 0.0, 1.0],
    )
    assert np.allclose(normalize_circuit_damage(np.array([-0.1, 0.75, 1.1])), [1.0, 0.25, 0.0])
    assert np.allclose(
        normalize_patching_damage(
            np.array([1.0, -1.0]), np.array([0.5, 0.0]), scale=2.0
        ),
        [0.25, 0.5],
    )
    with pytest.raises(ValueError, match="scale"):
        normalize_probe_damage(np.ones(1), np.ones(1), scale=0.0)


def _continuous_instrument(offset=0.0, n=80):
    spr = np.linspace(0.1, 2.0, n) + offset
    signal = spr * 2.0 + 0.1
    perturbation = signal / spr
    bit_width = np.tile(np.array([4.0, 6.0, 8.0, 5.0]), n // 4)
    damage = np.clip(0.15 + 0.3 * spr, 0.0, 1.0)
    groups = np.repeat(np.arange(n // 4), 4)
    return InstrumentData(
        spr=spr,
        target=damage,
        bit_width=bit_width,
        signal=signal,
        perturbation=perturbation,
        groups=groups,
        target_kind="continuous",
    )


def test_all_preregistered_baselines_and_main_model_are_reported():
    metrics = evaluate_transfer_models(
        _continuous_instrument(), _continuous_instrument(0.05), n_bootstrap=30, seed=7
    )
    assert set(metrics) == {
        "constant",
        "bit_width_only",
        "signal_only",
        "perturbation_only",
        "spr",
    }
    assert metrics["spr"].rmse < metrics["constant"].rmse
    assert metrics["spr"].r2 > 0.9
    assert metrics["spr"].spearman > 0.9
    assert metrics["spr"].improvement_over_constant > 0
    assert metrics["spr"].confidence_interval is not None


def test_binary_transfer_reports_calibration_metrics():
    source = _continuous_instrument()
    target = _continuous_instrument(0.05)
    source.target = (source.spr > np.median(source.spr)).astype(float)
    target.target = (target.spr > np.median(target.spr)).astype(float)
    source.target_kind = target.target_kind = "binary"
    metrics = evaluate_transfer_models(source, target)
    assert metrics["spr"].log_loss is not None
    assert metrics["spr"].brier is not None
    assert metrics["spr"].calibration_error is not None
    assert metrics["spr"].auc > 0.9


def test_grouped_bootstrap_is_seeded_and_requires_independent_groups():
    target = np.array([0.0, 0.0, 1.0, 1.0])
    prediction = np.array([0.1, 0.1, 0.8, 0.8])
    groups = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"])
    metric = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
    first = grouped_bootstrap_ci(
        target, prediction, groups, metric, n_bootstrap=50, seed=3
    )
    second = grouped_bootstrap_ci(
        target, prediction, groups, metric, n_bootstrap=50, seed=3
    )
    assert first == second
    with pytest.raises(ValueError, match="two independent groups"):
        grouped_bootstrap_ci(
            target, prediction, np.zeros(4), metric, n_bootstrap=10, seed=0
        )


def test_leave_one_instrument_out_fits_shared_models():
    instruments = {
        name: _continuous_instrument(offset)
        for name, offset in zip(
            ["sae", "probe", "circuit", "patching"], [0.0, 0.02, 0.04, 0.06]
        )
    }
    results = leave_one_instrument_out(instruments, n_bootstrap=20, seed=4)
    assert {result.held_out_instrument for result in results} == set(instruments)
    assert all(set(result.metrics) == {
        "constant", "bit_width_only", "signal_only", "perturbation_only", "spr"
    } for result in results)
    assert all(result.metrics["spr"].rmse < result.metrics["constant"].rmse for result in results)


def test_leave_one_instrument_out_tolerates_single_group_instrument():
    # patching is single-group by construction (one causal component); LOIO must
    # not crash on its ungroupable bootstrap. The CI is simply None when it is the
    # held-out target, while the point metrics are still produced.
    instruments = {
        name: _continuous_instrument(offset)
        for name, offset in zip(["sae", "probe", "circuit"], [0.0, 0.02, 0.04])
    }
    single = _continuous_instrument(0.06, n=8)
    single.groups = np.zeros(single.target.size)  # a single independent group
    instruments["patching"] = single

    results = leave_one_instrument_out(instruments, n_bootstrap=20, seed=4)
    held = {result.held_out_instrument: result for result in results}
    assert set(held) == set(instruments)
    assert held["patching"].metrics["spr"].confidence_interval is None
    assert held["patching"].metrics["spr"].rmse is not None
    assert held["sae"].metrics["spr"].confidence_interval is not None


def test_dry_run_and_smoke_contracts_are_explicit_and_non_scientific():
    instruments = {
        "sae": _continuous_instrument(),
        "probe": _continuous_instrument(0.05),
    }
    full_plan = build_transfer_run_plan(instruments)
    assert full_plan.mode == "harmonized"
    assert full_plan.scientifically_valid
    assert full_plan.output_directory == "results/harmonized/spr"
    assert "pairwise_transfer.csv" in full_plan.expected_artifacts

    smoke = make_smoke_test_data(instruments, max_groups=2, max_bit_widths=1)
    smoke_plan = build_transfer_run_plan(smoke, smoke_test=True)
    assert smoke_plan.mode == "smoke"
    assert not smoke_plan.scientifically_valid
    assert smoke_plan.output_directory.startswith("results/smoke/")
    assert all(count <= 8 for count in smoke_plan.estimated_observations.values())
    assert all(len(bits) <= 1 for bits in smoke_plan.conditions.values())
