"""Plot 2: delta PPL vs bit width."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("results_new/novelty_flat_stream/harmonized/sae/perplexity.csv")
OUTPUT = Path("paper/fig_ppl_delta.png")


def read_curve(path: Path) -> tuple[list[int], list[float]]:
    bits, delta = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["bits"]) == 16:
                continue
            bits.append(int(row["bits"]))
            delta.append(float(row["delta_ppl_percent"]))
    return bits, delta


def main() -> None:
    bits, delta = read_curve(INPUT)
    plt.figure(figsize=(3.2, 2.4))
    plt.plot(bits, delta, marker="o", color="C0")
    plt.gca().invert_xaxis()
    plt.xlabel("bits")
    plt.ylabel("ΔPPL %")
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()