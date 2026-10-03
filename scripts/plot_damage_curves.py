"""Plot 4: single-panel damage curves for SAE non-survival, probe big-drop rate, and PPL delta."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("results_new/novelty_flat_stream/harmonized/core_divergence_gate.json")
OUTPUT = Path("paper/fig_damage_curves.png")


def read_curves(path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_bit_width = payload["by_bit_width"]
    bits = [8, 7, 6, 5, 4]
    sae_nonsurvival = [by_bit_width[f"int{b}"]["sae_nonsurvival_pct"] for b in bits]
    probe_big_drop = [by_bit_width[f"int{b}"]["probe_big_drop_rate"] * 100.0 for b in bits]
    ppl_delta = [by_bit_width[f"int{b}"]["ppl_delta_percent"] for b in bits]
    return bits, sae_nonsurvival, probe_big_drop, ppl_delta


def main() -> None:
    bits, sae_nonsurvival, probe_big_drop, ppl_delta = read_curves(INPUT)

    plt.figure(figsize=(4.0, 2.8))
    plt.plot(bits, sae_nonsurvival, marker="o", label="SAE non-survival %")
    plt.plot(bits, probe_big_drop, marker="o", label="probe big-drop %")
    plt.plot(bits, ppl_delta, marker="o", label="ΔPPL %")
    plt.gca().invert_xaxis()
    plt.xlabel("bits")
    plt.ylabel("% / delta")
    plt.legend(fontsize=7)
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
