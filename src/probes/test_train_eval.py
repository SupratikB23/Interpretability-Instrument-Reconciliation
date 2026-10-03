"""Contract tests for config-driven data and frozen probe evaluation."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.probes import train_eval
from src.probes.train_eval import (
    ProbeCell,
    big_drop_rate,
    evaluate_frozen_probe,
    expected_calibration_error,
    fit_frozen_probe,
    load_cells_csv,
    load_probe_dataset,
    operational_failure_rate,
    probe_eval_cell,
    write_cells_csv,
)

ELIG = 0.75
BIG_DROP = 0.05


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n, d = 1500, 32
    x = rng.normal(0, 1, (n, d))
    y = (x[:, 0] + 0.5 * rng.normal(0, 1, n) > 0).astype(int)
    return x, y


def test_delta_auc_and_calibration_metrics():
    x, y = _data()
    rng = np.random.default_rng(3)
    cell = probe_eval_cell(
        x,
        y,
        x + 0.01 * rng.normal(0, 1, x.shape),
        seed=1,
        base_auc_eligibility=ELIG,
    )
    assert cell.eligible
    assert abs(cell.delta_auc) < BIG_DROP
    assert 0 <= cell.brier <= 1
    assert cell.log_loss >= 0
    assert 0 <= cell.ece <= 1


def test_expected_calibration_error_is_zero_for_perfect_predictions():
    assert expected_calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0]), n_bins=2
    ) == 0.0


def test_frozen_probe_is_trained_once_and_reused(monkeypatch):
    x, y = _data()
    calls = 0
    original = train_eval.train_probe

    def counted_train(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(train_eval, "train_probe", counted_train)
    frozen = fit_frozen_probe(x, y, seed=2, base_auc_eligibility=ELIG)
    fp16 = evaluate_frozen_probe(frozen, x, bits=16)
    int8 = evaluate_frozen_probe(frozen, x + 0.01, bits=8)
    int4 = evaluate_frozen_probe(frozen, x + 0.1, bits=4)

    assert calls == 1
    assert fp16.base_auc == int8.base_auc == int4.base_auc
    assert fp16.seed == int8.seed == int4.seed == 2


def test_big_drop_definition_and_ineligible_exclusion():
    cells = [
        ProbeCell("h", 12, 0, 6, base_auc=0.9, auc=0.85, eligible=True),
        ProbeCell("h", 12, 1, 6, base_auc=0.9, auc=0.4, eligible=False),
    ]
    assert big_drop_rate(cells, BIG_DROP) == 1.0


def test_operational_failure_is_absolute_post_update_auc():
    cells = [
        ProbeCell("h", 12, 0, 6, base_auc=0.95, auc=0.71, eligible=True),
        ProbeCell("h", 12, 1, 6, base_auc=0.76, auc=0.69, eligible=True),
    ]
    # The large first delta is not a failure; the smaller second delta is,
    # because operational failure is based on post-update AUC < 0.70.
    assert operational_failure_rate(cells, post_update_auc_min=0.70) == 0.5


def _dataset_config():
    return {
        "harmfulness": {
            "dataset_id": "org/monitor-data",
            "config": "harmful",
            "revision": "abc123",
            "split": "validation",
            "text_field": "prompt",
            "label_field": "kind",
            "label_map": {"safe": 0, "unsafe": 1},
        }
    }


def test_dataset_loader_uses_complete_config_and_disables_remote_code(monkeypatch):
    observed = {}

    def fake_load_dataset(dataset_id, config, **kwargs):
        observed.update(dataset_id=dataset_id, config=config, **kwargs)
        return {"prompt": ["safe text", "unsafe text"], "kind": ["safe", "unsafe"]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    texts, labels = load_probe_dataset("harmfulness", _dataset_config())

    assert texts == ["safe text", "unsafe text"]
    assert labels.tolist() == [0, 1]
    assert observed == {
        "dataset_id": "org/monitor-data",
        "config": "harmful",
        "split": "validation",
        "revision": "abc123",
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    "dataset, message",
    [
        ({"prompt": ["only"], "kind": ["safe"]}, "both binary classes"),
        ({"prompt": ["one", "two"], "kind": ["safe"]}, "aligned nonempty"),
        # Empty/non-string rows are filtered before mapping (real corpora carry a
        # few blank rows); dropping the only "safe" row here collapses to one class.
        ({"prompt": ["", "two"], "kind": ["safe", "unsafe"]}, "both binary classes"),
    ],
)
def test_dataset_loader_rejects_invalid_alignment_classes_and_text(monkeypatch, dataset, message):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset),
    )
    with pytest.raises(ValueError, match=message):
        load_probe_dataset("harmfulness", _dataset_config())


def test_dataset_loader_requires_generic_contract():
    with pytest.raises(ValueError, match="must map dataset_id"):
        load_probe_dataset("harmfulness", {"harmfulness": "wildguardmix"})


def test_csv_round_trip_includes_mode_and_calibration(tmp_path):
    cell = ProbeCell(
        "h", 12, 0, 6, 0.9, 0.85, True,
        base_brier=0.1, brier=0.2,
        base_log_loss=0.3, log_loss=0.4,
        base_ece=0.05, ece=0.08,
        mode="reproduction",
    )
    path = tmp_path / "cells.csv"
    write_cells_csv([cell], path)
    loaded = load_cells_csv(path)[0]
    assert loaded.mode == "reproduction"
    assert loaded.brier == 0.2
    assert loaded.log_loss == 0.4
    assert loaded.ece == 0.08


def test_placeholder_probe_dataset_ids_are_rejected():
    """A cosmetic placeholder identifier must cause config validation failure."""
    from src.config import ConfigValidationError, _reject_placeholder_dataset_specs

    for identifier in ("should/should", "synthetic/high-stakes-monitor"):
        with pytest.raises(ConfigValidationError, match="placeholder"):
            _reject_placeholder_dataset_specs(
                {"datasets": {"harmfulness": {"dataset_id": identifier}}}
            )
        with pytest.raises(ConfigValidationError, match="placeholder"):
            _reject_placeholder_dataset_specs(
                {"model_config": {"probe": {"datasets": {"harmfulness": {"identifier": identifier}}}}}
            )
    # Real identifiers pass validation.
    _reject_placeholder_dataset_specs(
        {"datasets": {"harmfulness": {"dataset_id": "allenai/wildguardmix"}}}
    )


def test_probe_train_test_leakage_is_detected(monkeypatch):
    """Overlapping train/test indices must fail loudly."""
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    def overlapping_split(indices, test_size=0.2, random_state=None, stratify=None):
        train = np.arange(12)
        test = np.arange(10, 20)  # indices 10, 11 overlap with train
        return train, test

    monkeypatch.setattr(train_eval, "train_test_split", overlapping_split)
    with pytest.raises(RuntimeError, match="leakage"):
        fit_frozen_probe(x, y, seed=0, base_auc_eligibility=0.5)
