"""Plot 3: probe delta AUC by bit width, eligible cells only (reads results CSV, no analysis)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("results_new/harmonized/probes/auc_calibration.csv")
OUTPUT = Path("paper/fig_probe_delta_auc.png")


def read_box_data(path: Path) -> dict[int, list[float]]:
    by_bits: dict[int, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("eligible", "").lower() != "true":
                continue
            bits = int(row["bits"])
            if bits == 16:
                continue
            by_bits.setdefault(bits, []).append(float(row["delta_auc"]))
    return {bits: by_bits[bits] for bits in sorted(by_bits)}


def main() -> None:
    by_bits = read_box_data(INPUT)
    bits = list(by_bits.keys())
    values = [by_bits[b] for b in bits]

    plt.figure(figsize=(3.2, 2.6))
    ax = plt.gca()
    ax.boxplot(values, positions=bits, widths=0.5, patch_artist=False)
    ax.invert_xaxis()
    ax.set_xticks(bits)
    ax.set_xticklabels([str(b) for b in bits])
    ax.set_xlabel("bits")
    ax.set_ylabel("ΔAUC")
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
