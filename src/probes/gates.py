"""Statistical gate for probe-arm reproduction and directional checks."""

from __future__ import annotations

import math
from typing import Any

from .train_eval import ProbeCell, quantization_cells


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError(f"invalid binomial counts: successes={successes}, total={total}")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def probe_gate_report(
    cells: list[ProbeCell],
    reference_min: float,
    reference_max: float,
    *,
    big_drop_auc_threshold: float = 0.05,
    operational_failure_auc: float = 0.70,
    min_eligible_count: int = 1,
    min_eligible_coverage: float = 0.5,
    max_operational_failures: int = 0,
    require_reference_interval: bool = True,
) -> dict[str, Any]:
    """Build the complete gate report; ``require_reference_interval`` means exact reproduction.

    Directional harmonized checks set ``require_reference_interval=False``. They
    enforce the published upper bound but are never labelled reproductions.
    """
    if not (0.0 <= reference_min <= reference_max <= 1.0):
        raise ValueError("reference big-drop interval must satisfy 0 <= min <= max <= 1")
    if min_eligible_count < 1:
        raise ValueError("min_eligible_count must be at least one")
    if not (0.0 <= min_eligible_coverage <= 1.0):
        raise ValueError("min_eligible_coverage must be within [0, 1]")
    if max_operational_failures < 0:
        raise ValueError("max_operational_failures cannot be negative")

    quantized = quantization_cells(cells)
    eligible = [cell for cell in quantized if cell.eligible]
    total = len(quantized)
    eligible_count = len(eligible)
    coverage = eligible_count / total if total else 0.0
    big_drops = sum(cell.delta_auc <= -big_drop_auc_threshold for cell in eligible)
    failures = sum(cell.auc < operational_failure_auc for cell in eligible)
    rate = big_drops / eligible_count if eligible_count else 0.0
    failure_rate = failures / eligible_count if eligible_count else 0.0
    ci = wilson_interval(big_drops, eligible_count) if eligible_count else (float("nan"), float("nan"))

    violations: list[str] = []
    if not cells:
        violations.append("no probe cells to evaluate (run the probe arm first)")
    elif not quantized:
        violations.append("no quantized probe cells to evaluate (only FP16 reference rows present)")
    if eligible_count < min_eligible_count:
        violations.append(
            f"eligible cell count {eligible_count} below required minimum {min_eligible_count}"
        )
    if coverage < min_eligible_coverage:
        violations.append(
            f"eligible coverage {coverage:.4f} below required minimum {min_eligible_coverage:.4f}"
        )
    if eligible_count:
        if require_reference_interval and not (reference_min <= rate <= reference_max):
            violations.append(
                f"observed big-drop rate {rate:.4f} is outside published reference interval "
                f"[{reference_min:.4f}, {reference_max:.4f}]"
            )
        elif not require_reference_interval and rate > reference_max:
            violations.append(
                f"directional big-drop rate {rate:.4f} exceeds published reference upper bound "
                f"{reference_max:.4f}"
            )
        if failures > max_operational_failures:
            violations.append(
                f"operational failures {failures} exceed configured maximum "
                f"{max_operational_failures} (absolute post-update AUC < {operational_failure_auc:.2f})"
            )

    return {
        "check_type": "reproduction" if require_reference_interval else "directional",
        "observed_big_drop_rate": rate,
        "big_drop_wilson_95_ci": ci,
        "observed_operational_failure_rate": failure_rate,
        "operational_failure_count": failures,
        "eligible_count": eligible_count,
        "quantized_count": total,
        "eligible_coverage": coverage,
        "published_benchmark_range": (reference_min, reference_max),
        "pass": not violations,
        "violations": violations,
    }


def probe_big_drop_violations(
    cells: list[ProbeCell],
    reference_min: float,
    reference_max: float,
    order_of_magnitude_tolerance: float | None = None,
    big_drop_auc_threshold: float = 0.05,
    *,
    operational_failure_auc: float = 0.70,
    min_eligible_count: int = 1,
    min_eligible_coverage: float = 0.5,
    max_operational_failures: int = 0,
    require_reference_interval: bool = True,
) -> list[str]:
    """Return gate violations.

    ``order_of_magnitude_tolerance`` remains accepted for call compatibility but
    is intentionally ignored: Priority 3 forbids the old 10x acceptance band.
    """
    del order_of_magnitude_tolerance
    report = probe_gate_report(
        cells,
        reference_min,
        reference_max,
        big_drop_auc_threshold=big_drop_auc_threshold,
        operational_failure_auc=operational_failure_auc,
        min_eligible_count=min_eligible_count,
        min_eligible_coverage=min_eligible_coverage,
        max_operational_failures=max_operational_failures,
        require_reference_interval=require_reference_interval,
    )
    return list(report["violations"])


def assert_probe_big_drop_gate(
    cells: list[ProbeCell],
    reference_min: float,
    reference_max: float,
    order_of_magnitude_tolerance: float | None = None,
    big_drop_auc_threshold: float = 0.05,
    *,
    operational_failure_auc: float = 0.70,
    min_eligible_count: int = 1,
    min_eligible_coverage: float = 0.5,
    max_operational_failures: int = 0,
    require_reference_interval: bool = True,
) -> None:
    """Assert the strict probe gate and list every failed criterion."""
    violations = probe_big_drop_violations(
        cells,
        reference_min,
        reference_max,
        order_of_magnitude_tolerance,
        big_drop_auc_threshold,
        operational_failure_auc=operational_failure_auc,
        min_eligible_count=min_eligible_count,
        min_eligible_coverage=min_eligible_coverage,
        max_operational_failures=max_operational_failures,
        require_reference_interval=require_reference_interval,
    )
    if violations:
        label = "reproduction" if require_reference_interval else "directional"
        raise AssertionError(
            f"Probe-arm {label} gate FAILED:\n  " + "\n  ".join(violations)
        )
