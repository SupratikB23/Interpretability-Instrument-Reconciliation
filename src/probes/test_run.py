"""Contracts for probe experiment modes and metadata."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.config import Config
from src.probes.run import ProbeMode, _probe_config, _validate_mode, _write_metadata


def _cfg(probe: dict[str, Any]) -> Config:
    return cast(Config, cast(object, SimpleNamespace(
        model={
            "model": {"name": "model/id", "device": "cpu"},
            "perplexity": {"seq_len": 128},
            "probe": probe,
        },
        quant={
            "compression": {
                "method": "RTN",
                "bit_widths": [8, 4],
                "quant_modules": ["q_proj"],
                "skip_modules": ["embed"],
            }
        },
    )))


def _probe() -> dict[str, Any]:
    return {
        "layers": [12],
        "n_seeds": 2,
        "datasets": {
            "harmfulness": {
                "dataset_id": "org/data",
                "split": "test",
                "text_field": "text",
                "label_field": "label",
                "label_map": {"safe": 0, "unsafe": 1},
            }
        },
        "monitors": ["harmfulness"],
        "base_auc_eligibility": 0.75,
        "big_drop_auc_threshold": 0.05,
        "test_size": 0.2,
    }


def test_mode_api_rejects_ambiguous_values():
    assert _validate_mode("reproduction") == "reproduction"
    assert _validate_mode("harmonized") == "harmonized"
    with pytest.raises(ValueError, match="probe mode"):
        _validate_mode("r1")


def test_mode_nested_config_overrides_common_values():
    probe = _probe()
    probe["reproduction"] = {"layers": [4]}
    probe["harmonized"] = {"layers": [12]}
    assert _probe_config(_cfg(probe), "reproduction")["layers"] == [4]
    assert _probe_config(_cfg(probe), "harmonized")["layers"] == [12]


@pytest.mark.parametrize(
    ("mode", "semantics"),
    [
        ("reproduction", "exact_reproduction"),
        ("harmonized", "directional_harmonized_r1"),
    ],
)
def test_metadata_distinguishes_result_families(
    tmp_path: Path, mode: ProbeMode, semantics: str
) -> None:
    probe = _probe()
    path = tmp_path / mode / "probes" / "metadata.json"
    _write_metadata(path, _cfg(probe), probe, mode, [])
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["experiment_mode"] == mode
    assert metadata["check_semantics"] == semantics
    assert metadata["probe_training_precision"] == "FP16"
    assert metadata["probe_retrained_per_condition"] is False
    assert metadata["metrics"] == ["auc", "brier", "log_loss", "ece"]
    assert metadata["operational_failure_criterion"]["threshold"] == 0.70
