"""Unit tests for the SAE INT6 reproduction gate."""

import json

import pytest

from src.sae.gates import (
    assert_sae_int6_gate,
    sae_gate_report,
    sae_int6_violations,
    write_sae_gate_report,
)

TARGETS = {"survived": 51.30, "degraded": 48.70, "damaged": 0.00}
TOL = 3.0


def test_gate_passes_within_tolerance():
    # Matches Duan's reported Gemma-2-2B INT6 row within ±3pp.
    assert_sae_int6_gate({"survived": 52.5, "degraded": 47.5, "damaged": 0.0}, TARGETS, TOL)


def test_gate_fails_outside_tolerance():
    with pytest.raises(AssertionError):
        assert_sae_int6_gate({"survived": 40.0, "degraded": 60.0, "damaged": 0.0}, TARGETS, TOL)


def test_gate_fails_on_missing_class():
    viol = sae_int6_violations({"survived": 51.3, "degraded": 48.7}, TARGETS, TOL)
    assert any("damaged" in v for v in viol)


def test_damaged_class_respects_3pp():
    # Duan reports 0.00% damaged with ±3pp tolerance: a band at or under 3pp
    # passes, above it fails.
    assert_sae_int6_gate({"survived": 51.3, "degraded": 45.7, "damaged": 3.0}, TARGETS, TOL)
    with pytest.raises(AssertionError):
        assert_sae_int6_gate({"survived": 51.3, "degraded": 44.7, "damaged": 4.0}, TARGETS, TOL)


def test_gate_report_contains_expected_fields():
    report = sae_gate_report(
        {"survived": 51.9, "degraded": 48.1, "damaged": 0.0}, TARGETS, TOL, bits=6
    )
    assert report["pass"] is True
    for key in ("expected", "observed", "absolute_error", "tolerance_pp", "pass", "notes"):
        assert key in report
    assert report["expected"]["survived"] == 51.30
    assert report["absolute_error"]["survived"] == pytest.approx(0.6)


def test_gate_report_writes_json(tmp_path):
    report = sae_gate_report(
        {"survived": 60.0, "degraded": 40.0, "damaged": 0.0}, TARGETS, TOL
    )
    assert report["pass"] is False
    path = tmp_path / "results" / "reproduction" / "sae" / "reproduction_gate.json"
    write_sae_gate_report(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["pass"] is False
    assert loaded["gate"] == "sae_int6"