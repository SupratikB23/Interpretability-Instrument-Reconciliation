"""SAE-arm reproduction gate.

Gate 1: SAE-arm Gemma-2-2B INT6 survival must be within ±3 percentage points of
Duan 2606.03002's reported 51.30% survived / 48.70% degraded / 0.00% damaged.
This is code-enforced: the pipeline refuses to proceed downstream until
it passes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .survival import DAMAGED, DEGRADED, SURVIVED


def sae_int6_violations(
    observed: Mapping[str, float],
    targets: Mapping[str, float],
    tolerance_pp: float,
) -> list[str]:
    """Return a list of human-readable gate violations (empty if the gate passes).

    Args:
        observed: ``{"survived": %, "degraded": %, "damaged": %}`` from the
            INT6 run.
        targets: Duan 2606.03002 Table 1 Gemma-2-2B INT6 row
            ``{survived: 51.30, degraded: 48.70, damaged: 0.00}``.
        tolerance_pp: allowed deviation in percentage points.

    Docstring reference: Duan 2606.03002 Table 1.
    """
    violations: list[str] = []
    for label in (SURVIVED, DEGRADED, DAMAGED):
        obs = observed.get(label)
        target = targets[label]
        if obs is None:
            violations.append(f"missing class '{label}' in observed survival rates")
            continue
        if abs(obs - target) > tolerance_pp:
            violations.append(
                f"{label}: observed {obs:.2f}% vs target {target:.2f}% "
                f"(|diff| {abs(obs - target):.2f}pp > {tolerance_pp}pp)"
            )
    return violations


def assert_sae_int6_gate(
    observed: Mapping[str, float],
    targets: Mapping[str, float],
    tolerance_pp: float,
) -> None:
    """Assert the SAE INT6 reproduction gate; raise AssertionError on violation."""
    violations = sae_int6_violations(observed, targets, tolerance_pp)
    if violations:
        raise AssertionError(
            "SAE-arm INT6 reproduction gate FAILED:\n  "
            + "\n  ".join(violations)
        )


def sae_gate_report(
    observed: Mapping[str, float],
    targets: Mapping[str, float],
    tolerance_pp: float,
    *,
    bits: int = 6,
) -> dict:
    """Build a machine-readable gate report.

    Returns a dict with ``expected``, ``observed`` (per survival class),
    ``absolute_error``, ``tolerance_pp``, ``pass``, and ``notes``.
    """
    errors = {
        label: round(abs(float(observed[label]) - float(targets[label])), 4)
        for label in (SURVIVED, DEGRADED, DAMAGED)
    }
    violations = sae_int6_violations(observed, targets, tolerance_pp)
    return {
        "gate": "sae_int6",
        "bits": int(bits),
        "expected": {label: float(targets[label]) for label in (SURVIVED, DEGRADED, DAMAGED)},
        "observed": {label: round(float(observed[label]), 4) for label in (SURVIVED, DEGRADED, DAMAGED)},
        "absolute_error": errors,
        "tolerance_pp": float(tolerance_pp),
        "pass": not violations,
        "notes": violations,
    }


def write_sae_gate_report(report: dict, path: str | Path) -> None:
    """Persist the gate report as results/reproduction/sae/reproduction_gate.json."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True)
    _ = destination.write_text(content + "\n", encoding="utf-8")