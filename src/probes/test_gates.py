"""Contract tests for strict probe reproduction and directional gates."""

import pytest

from src.probes.gates import (
    assert_probe_big_drop_gate,
    probe_big_drop_violations,
    probe_gate_report,
    wilson_interval,
)
from src.probes.train_eval import ProbeCell, quantization_cells

REF_MIN, REF_MAX = 0.0083, 0.0333


def _cells(big_drops: int, total: int = 100) -> list[ProbeCell]:
    cells = []
    for i in range(total):
        auc = 0.84 if i < big_drops else 0.89
        cells.append(ProbeCell("h", 12, i, 6, base_auc=0.9, auc=auc, eligible=True))
    return cells


def test_exact_reproduction_gate_passes_in_published_interval():
    cells = _cells(big_drops=2)
    assert probe_big_drop_violations(cells, REF_MIN, REF_MAX) == []
    assert_probe_big_drop_gate(cells, REF_MIN, REF_MAX)
    report = probe_gate_report(cells, REF_MIN, REF_MAX)
    assert report["check_type"] == "reproduction"
    assert report["observed_big_drop_rate"] == 0.02
    assert report["pass"] is True


def test_zero_big_drops_does_not_prove_exact_reproduction():
    cells = _cells(big_drops=0)
    with pytest.raises(AssertionError, match="outside published reference interval"):
        assert_probe_big_drop_gate(cells, REF_MIN, REF_MAX)


def test_zero_big_drops_can_pass_labelled_directional_check():
    report = probe_gate_report(
        _cells(big_drops=0),
        REF_MIN,
        REF_MAX,
        require_reference_interval=False,
    )
    assert report["check_type"] == "directional"
    assert report["pass"] is True


def test_old_ten_x_argument_no_longer_loosens_gate():
    cells = _cells(big_drops=20)
    assert probe_big_drop_violations(cells, REF_MIN, REF_MAX, 10.0)


def test_gate_requires_eligible_count_and_coverage():
    cells = _cells(big_drops=1, total=10)
    for cell in cells[2:]:
        cell.eligible = False
    violations = probe_big_drop_violations(
        cells,
        REF_MIN,
        REF_MAX,
        min_eligible_count=5,
        min_eligible_coverage=0.8,
    )
    assert any("eligible cell count" in violation for violation in violations)
    assert any("eligible coverage" in violation for violation in violations)


def test_gate_rejects_absolute_operational_failure_and_allows_configured_max():
    cells = _cells(big_drops=2)
    cells[-1].base_auc = 0.72
    cells[-1].auc = 0.69  # delta is only -0.03, but absolute AUC fails.
    strict = probe_gate_report(cells, REF_MIN, REF_MAX)
    assert strict["operational_failure_count"] == 1
    assert strict["pass"] is False

    configured = probe_gate_report(
        cells, REF_MIN, REF_MAX, max_operational_failures=1
    )
    assert configured["pass"] is True


def test_report_contains_wilson_interval_and_benchmark():
    report = probe_gate_report(_cells(big_drops=2), REF_MIN, REF_MAX)
    low, high = report["big_drop_wilson_95_ci"]
    assert 0 <= low < 0.02 < high <= 1
    assert report["published_benchmark_range"] == (REF_MIN, REF_MAX)
    assert wilson_interval(2, 100) == (low, high)


def test_gate_fails_on_empty_or_fp16_only_cells():
    with pytest.raises(AssertionError, match="no probe cells"):
        assert_probe_big_drop_gate([], REF_MIN, REF_MAX)
    fp16 = [ProbeCell("h", 12, 0, 16, 0.9, 0.9, True)]
    with pytest.raises(AssertionError, match="no quantized probe cells"):
        assert_probe_big_drop_gate(fp16, REF_MIN, REF_MAX)


def test_fp16_reference_cells_are_excluded_from_gate_rate():
    quant = _cells(big_drops=50, total=50)
    fp16 = [ProbeCell("h", 12, i, 16, 0.9, 0.9, True) for i in range(50)]
    assert probe_big_drop_violations(quant + fp16, REF_MIN, REF_MAX)


def test_quantization_cells_drops_fp16():
    cells = [
        ProbeCell("h", 12, 0, 16, 0.9, 0.9, True),
        ProbeCell("h", 12, 0, 6, 0.9, 0.8, True),
    ]
    assert [cell.bits for cell in quantization_cells(cells)] == [6]
