"""Plot 1: SAE survival vs bit width, both substrates (reads results CSVs, no analysis)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUTS = {
    "no BOS (flat_stream)": Path("results_new/novelty_flat_stream/harmonized/sae/survival_rates.csv"),
    "BOS present": Path("results_new/harmonized/sae/survival_rates.csv"),
}
OUTPUT = Path("paper/fig_sae_survival.png")


def read_curve(path: Path) -> tuple[list[int], list[float]]:
    bits, survived = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["bits"]) == 16:
                continue
            bits.append(int(row["bits"]))
            survived.append(float(row["survived_pct"]))
    return bits, survived


def main() -> None:
    plt.figure(figsize=(3.2, 2.4))
    for label, path in INPUTS.items():
        bits, survived = read_curve(path)
        plt.plot(bits, survived, marker="o", label=label)
    plt.gca().invert_xaxis()
    plt.xlabel("bits")
    plt.ylabel("survived %")
    plt.legend(fontsize=7)
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()