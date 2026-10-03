"""Single-command runner for BOTH reproduction gates

Usage:
    python run_gates.py                       # run gates 1 and 2 against results artifacts
    python run_gates.py --bits 6              # scope (informational; gates read results)

Gates are real assertions, not eyeballing a plot. Each gate reads
the pipeline's saved results artifact from ``configs/quant_rtn_sweep.yaml``
(results/*.csv). If an artifact is missing the gate FAILS with a message telling
the human to run the corresponding arm first — a gate that isn't code-enforced
silently stops being checked, so missing data is not silently skipped.

Exit code 0 = both gates pass; non-zero otherwise.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from src.config import Config, results_path


def _load_survival_rates(path: Path) -> dict[int, dict[str, float]]:
    """Read results/sae/survival_rates.csv -> {bits: {class: pct}}."""
    if not path.exists():
        raise FileNotFoundError(
            f"SAE survival artifact not found at {path}. Run the SAE QDM sweep "
            f"(src/sae/qdm.py) first — gate 1 cannot be checked without it."
        )
    rows: dict[int, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[int(row["bits"])] = {
                "survived": float(row["survived_pct"]),
                "degraded": float(row["degraded_pct"]),
                "damaged": float(row["damaged_pct"]),
            }
    return rows


def _load_probe_cells(path: Path) -> list:
    """Read results/reproduction/probes/auc_calibration.csv -> list of ProbeCell.

    Delegates the parsing to ``train_eval.load_cells_csv`` (the writer's own
    counterpart) rather than duplicating the column mapping here — two copies of
    the schema drift apart the moment a column is added.
    """
    from src.probes.train_eval import load_cells_csv

    if not path.exists():
        raise FileNotFoundError(
            f"Probe cells artifact not found at {path}. Run the probe RTN sweep "
            f"first — gate 2 cannot be checked without it."
        )
    return load_cells_csv(path)


def run_gate_1(cfg: Config) -> None:
    """SAE-arm Gemma-2-2B INT6 survival within ±3pp of 2606.03002."""
    from src.sae.gates import assert_sae_int6_gate, sae_gate_report, write_sae_gate_report

    rates = _load_survival_rates(results_path(cfg, "sae_survival"))
    targets = cfg.quant["gate_sae_int6"]
    bits = targets["bits"]
    if bits not in rates:
        raise AssertionError(
            f"gate 1 requires an INT{bits} SAE row; not present in survival_rates.csv"
        )
    observed = {
        "survived": rates[bits]["survived"],
        "degraded": rates[bits]["degraded"],
        "damaged": rates[bits]["damaged"],
    }
    expected = {
        "survived": targets["survived_target"],
        "degraded": targets["degraded_target"],
        "damaged": targets["damaged_target"],
    }
    report = sae_gate_report(observed, expected, tolerance_pp=targets["tolerance_pp"], bits=bits)
    write_sae_gate_report(report, results_path(cfg, "sae_gate_report"))
    assert_sae_int6_gate(observed, expected, tolerance_pp=targets["tolerance_pp"])
    print(
        f"[GATE 1 PASS] SAE INT{bits} survived={rates[bits]['survived']:.2f}% degraded="
        f"{rates[bits]['degraded']:.2f}% damaged={rates[bits]['damaged']:.2f}% "
        f"(targets {targets['survived_target']}/{targets['degraded_target']}/{targets['damaged_target']}%, "
        f"tol {targets['tolerance_pp']}pp); report -> {results_path(cfg, 'sae_gate_report')}"
    )


def run_gate_2(cfg: Config) -> None:
    """Probe-arm quantization big-drop rate consistent with 2606.15980.

    Strict semantics: the observed big-drop rate must fall
    INSIDE the reference interval; eligibility coverage, minimum eligible
    count, and operational-failure limits are enforced. The legacy 10x
    order-of-magnitude band is not supported.
    """
    from src.probes.gates import assert_probe_big_drop_gate

    cells = _load_probe_cells(results_path(cfg, "probe_auc"))
    # Gates are reproduction gates: they evaluate the reproduction family only.
    # A harmonized-only artifact must not be able to satisfy a reproduction gate.
    reproduction = [c for c in cells if c.mode == "reproduction"]
    if not reproduction:
        raise AssertionError(
            "gate 2 found no reproduction-mode probe cells in the artifact; "
            "run the probe sweep in reproduction mode first"
        )
    g = cfg.quant["gate_probe"]
    assert_probe_big_drop_gate(
        reproduction,
        reference_min=g["reference_big_drop_rate_min"],
        reference_max=g["reference_big_drop_rate_max"],
        # Must come from config, not the function default — otherwise editing
        # probe.big_drop_auc_threshold silently stops affecting the gate.
        big_drop_auc_threshold=cfg.model["probe"]["big_drop_auc_threshold"],
        operational_failure_auc=cfg.model["probe"]["operational_failure_auc"],
        min_eligible_count=g["min_eligible_count"],
        min_eligible_coverage=g["min_eligible_coverage"],
        max_operational_failures=g["max_operational_failures"],
        require_reference_interval=g["require_reference_interval"],
    )
    print(
        f"[GATE 2 PASS] probe big-drop rate inside reference band "
        f"[{g['reference_big_drop_rate_min']}, {g['reference_big_drop_rate_max']}] "
        f"(strict gate: no order-of-magnitude band)"
    )


def main() -> int:
    cfg = Config.from_yaml()
    failures: list[str] = []
    for name, fn in (("gate 1 (SAE INT6)", run_gate_1), ("gate 2 (probe big-drop)", run_gate_2)):
        try:
            fn(cfg)
        except Exception as exc:  # noqa: BLE001 - surface every gate failure
            failures.append(f"{name}: {exc}")
    if failures:
        print("\n[GATES FAILED]")
        for f in failures:
            print("  -", f)
        return 1
    print("\nBoth reproduction gates PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())