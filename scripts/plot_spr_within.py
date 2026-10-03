"""Plot 5: SPR vs target within each instrument, four small panels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTRUMENTS = ["sae", "probe", "circuits", "patching"]
INPUTS = {inst: Path(f"results_new/derived/spr_points_{inst}.csv") for inst in INSTRUMENTS}
WITHIN = Path("results_new/novelty_flat_stream/harmonized/spr/within_instrument.json")
OUTPUT = Path("paper/fig_spr_within.png")


def read_points(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            spr = float(row["spr"])
            target = float(row["target"])
            rows.append((max(spr, 1e-6), target))
    return rows


def main() -> None:
    payload = json.loads(WITHIN.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, instrument in zip(axes, INSTRUMENTS):
        rows = read_points(INPUTS[instrument])
        spr = [row[0] for row in rows]
        target = [row[1] for row in rows]
        ax.scatter(spr, target, s=6, alpha=0.45)
        ax.set_xscale("log")
        rho = float(payload[instrument]["spearman_spr_vs_damage"])
        ax.set_title(f"{instrument}\nρ={rho:.3f}")
        ax.set_xlabel("SPR")
        ax.set_ylabel("target")
        ax.grid(True, alpha=0.2)

    for ax in axes[2:]:
        ax.set_xlabel("SPR")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("target")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
