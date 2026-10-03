"""Plot 6: raw and normalized SPR transfer heatmaps with pass-cell markers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INSTRUMENTS = ["sae", "probe", "circuits", "patching"]
RAW_INPUT = Path("results_new/novelty_flat_stream/harmonized/spr/transfer_results.json")
NORMALIZED_INPUT = Path("results_new/novelty_flat_stream/harmonized/spr/transfer_results_normalized.json")
OUTPUT = Path("paper/fig_spr_transfer.png")


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.full((len(INSTRUMENTS), len(INSTRUMENTS)), np.nan)
    pass_mask = np.zeros((len(INSTRUMENTS), len(INSTRUMENTS)), dtype=bool)

    for item in payload["results"]:
        src = item["source_instrument"]
        tgt = item["target_instrument"]
        i = INSTRUMENTS.index(src)
        j = INSTRUMENTS.index(tgt)
        matrix[i, j] = float(item["delta_over_null"])
        pass_mask[i, j] = bool(item.get("passes", False))

    return matrix, pass_mask


def main() -> None:
    raw_matrix, raw_pass = load_matrix(RAW_INPUT)
    norm_matrix, norm_pass = load_matrix(NORMALIZED_INPUT)

    all_values = np.concatenate([
        raw_matrix[~np.isnan(raw_matrix)],
        norm_matrix[~np.isnan(norm_matrix)],
    ])
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, matrix, pass_mask, label in [
        (axes[0], raw_matrix, raw_pass, "raw"),
        (axes[1], norm_matrix, norm_pass, "normalized"),
    ]:
        image = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks(range(len(INSTRUMENTS)))
        ax.set_yticks(range(len(INSTRUMENTS)))
        ax.set_xticklabels(INSTRUMENTS)
        ax.set_yticklabels(INSTRUMENTS)
        ax.set_xlabel("target")
        ax.set_ylabel("source")

        for i in range(len(INSTRUMENTS)):
            for j in range(len(INSTRUMENTS)):
                if pass_mask[i, j]:
                    ax.plot(j, i, "o", markerfacecolor="none", markeredgecolor="black", markersize=6, markeredgewidth=1.2)
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black" if abs(matrix[i, j]) < 0.5 else "white", fontsize=7)

    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
