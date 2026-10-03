"""Tests for condition-safe NIE/PIE decomposition."""

import csv

import numpy as np
import pytest
import torch

from src.patching.nie_pie import (
    PatchingCondition,
    compute_nie,
    compute_pie,
    decompose_component,
    is_int_confounded,
    run_patching_arm,
)
from src.patching.run import _metric


class _FakeHookPoint:
    name = "blocks.12.hook_resid_post"


class _FakeModel:
    """Minimal stand-in exercising TransformerLens's hook-call convention.

    The point of the test is the calling convention: TransformerLens invokes a
    forward hook as ``fn(activation, hook=hook_point)`` — ``hook`` is a keyword
    argument — so ``_metric``'s replacement fn must accept it. A hook fn that
    names its second parameter positionally (``_hook``) raises "unexpected
    keyword argument 'hook'"; this fake reproduces that call so the regression
    stays caught without a real GPU model.
    """

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def to_single_token(self, text: str) -> int:
        return {" Paris": 1, " Tokyo": 2}[text]

    def run_with_hooks(self, tokens, *, fwd_hooks):
        logits = torch.zeros(1, tokens.shape[1], 3)
        for name, fn in fwd_hooks:
            assert name == _FakeHookPoint.name
            value = torch.zeros(1, tokens.shape[1], 4)
            # Exactly how TransformerLens calls it: hook passed by keyword.
            replaced = fn(value, hook=_FakeHookPoint())
            logits[0, -1, 1] = float(replaced.reshape(-1)[0])
        return logits

    def __call__(self, tokens):
        return torch.zeros(1, tokens.shape[1], 3)


def test_metric_hook_accepts_transformerlens_hook_keyword():
    model = _FakeModel()
    tokens = torch.zeros(1, 3, dtype=torch.long)
    activation = np.full((1, 3, 4), 5.0, dtype=np.float32)
    # Must not raise "unexpected keyword argument 'hook'".
    value = _metric(model, tokens, " Paris", " Tokyo", _FakeHookPoint.name, activation)
    assert value == pytest.approx(5.0)


def _condition(label: str, scale: float, offset: float = 0.0) -> PatchingCondition:
    clean = {"c1": np.array([1.0 + offset]), "c2": np.array([0.6 + offset])}
    corrupted = {"c1": np.array([0.0 + offset]), "c2": np.array([0.1 + offset])}
    state = {"scale": scale, "label": label}

    def model_fn(component, act, baseline):
        bypass = clean[component][0] if baseline == "clean" else corrupted[component][0]
        return 2.0 + scale * (0.7 * act[0] + 0.5 * act[0] * bypass)

    return PatchingCondition(
        label=label,
        state_id=f"weights-{label}",
        state=state,
        model_fn=model_fn,
        clean_acts=clean,
        corrupted_acts=corrupted,
    )


def test_nie_equals_pie_plus_int():
    condition = _condition("fp16", 1.0)
    clean = condition.clean_acts["c1"]
    corrupted = condition.corrupted_acts["c1"]
    clean_metric = condition.model_fn("c1", clean, "clean")
    corrupted_metric = condition.model_fn("c1", corrupted, "corrupted")
    nie = compute_nie(condition.model_fn, "c1", clean, corrupted, clean_metric)
    pie = compute_pie(condition.model_fn, "c1", clean, corrupted, corrupted_metric)
    assert nie == pytest.approx(1.2)
    assert pie == pytest.approx(0.7)
    assert nie - pie == pytest.approx(0.5)


def test_decomposition_persists_baseline_and_patched_metrics():
    result = decompose_component(_condition("fp16", 1.0), "c1")
    assert result.clean_metric == pytest.approx(3.2)
    assert result.corrupted_metric == pytest.approx(2.0)
    assert result.noised_metric == pytest.approx(2.0)
    assert result.denoised_metric == pytest.approx(2.7)
    assert result.nie == pytest.approx(result.pie + result.int_term)
    assert not result.int_confounded


def test_sign_disagreement_is_flagged_confounded():
    clean = {"c1": np.array([1.0])}
    corrupted = {"c1": np.array([0.0])}

    def model_fn(component, act, baseline):
        bypass = clean[component][0] if baseline == "clean" else corrupted[component][0]
        return 2.0 - 0.7 * act[0] + 2.0 * act[0] * bypass

    condition = PatchingCondition("fp16", "fp", object(), model_fn, clean, corrupted)
    result = decompose_component(condition, "c1")
    assert is_int_confounded(result.nie, result.pie)


def test_conditions_use_distinct_callable_state_activations_and_results(tmp_path):
    fp16 = _condition("fp16", 1.0)
    int4 = _condition("int4", 0.45, offset=0.03)
    assert fp16.model_fn is not int4.model_fn
    assert fp16.state is not int4.state

    results = run_patching_arm(
        ["c1", "c2"], {"fp16": fp16, "int4": int4}, output_dir=tmp_path
    )
    by_key = {(result.label, result.component): result for result in results}
    fp = by_key[("fp16", "c1")]
    quant = by_key[("int4", "c1")]
    assert quant.nie != pytest.approx(fp.nie)
    assert quant.pie != pytest.approx(fp.pie)
    assert quant.delta_nie == pytest.approx(quant.nie - fp.nie)
    assert quant.normalized_nie_damage == pytest.approx((fp.nie - quant.nie) / abs(fp.nie))

    with (tmp_path / "harmonized" / "patching" / "nie_pie.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {"clean_metric", "corrupted_metric", "noised_metric", "denoised_metric"} <= rows[0].keys()
    assert {"nie", "pie", "int_term", "normalized_baseline_damage"} <= rows[0].keys()


def test_reused_callable_or_state_is_rejected(tmp_path):
    fp16 = _condition("fp16", 1.0)
    fake = PatchingCondition(
        "int4",
        "weights-int4",
        fp16.state,
        fp16.model_fn,
        {"c1": np.array([0.9])},
        {"c1": np.array([0.2])},
    )
    with pytest.raises(ValueError, match="reuses another condition's model state"):
        run_patching_arm(["c1"], {"fp16": fp16, "int4": fake}, output_dir=tmp_path)


def test_cosmetic_label_with_identical_activations_is_rejected(tmp_path):
    fp16 = _condition("fp16", 1.0)
    other = _condition("int4", 0.5)
    with pytest.raises(ValueError, match="identical clean/corrupted activations"):
        run_patching_arm(["c1", "c2"], {"fp16": fp16, "int4": other}, output_dir=tmp_path)
