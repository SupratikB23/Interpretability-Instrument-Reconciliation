"""Export the NPZ-only data plotting needs into committable CSV/JSON.

NPZ files are gitignored, so plotting scripts never receive them. Every
task that read an NPZ (T0, variance decomposition, plots 5 and 7) is re-pointed
at these derived files. Written by LEAD, read-only for juniors.

Outputs under results_new/derived/:
  sae_int6_correlations.csv  16,384 rows: feature_id, corr + active flag per substrate
  spr_points_<inst>.csv      spr / target / bit_width per instrument (circuits subsampled)
  spr_scales.json            exact full-population scale stats (NOT subsampled)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("results_new")
DERIVED = ROOT / "derived"
FLAT = ROOT / "novelty_flat_stream/harmonized"
BOS = ROOT / "harmonized"
CIRCUIT_SAMPLE = 20_000  # circuits has 323,295 rows; keep the CSV small
SEED = 0


def export_correlations() -> None:
    """Per-feature INT6 correlation on both substrates, plus the active-feature gate."""
    out = DERIVED / "sae_int6_correlations.csv"
    corr_bos = np.load(BOS / "sae/correlations_int6.npz")["correlations"]
    corr_nobos = np.load(FLAT / "sae/correlations_int6.npz")["correlations"]
    assert corr_bos.shape == corr_nobos.shape, "dictionaries differ"

    def active(path: Path) -> set[int]:
        with np.load(path) as d:
            return set(d["feature_id"][d["bit_width"] == 6].tolist())

    act_bos = active(BOS / "spr/instruments/sae.npz")
    act_nobos = active(FLAT / "spr/instruments/sae.npz")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_id", "corr_bos_present", "corr_bos_absent",
                    "active_bos_present", "active_bos_absent"])
        for i in range(corr_bos.size):
            a, b = corr_bos[i], corr_nobos[i]
            w.writerow([i,
                        "" if not np.isfinite(a) else f"{a:.6f}",
                        "" if not np.isfinite(b) else f"{b:.6f}",
                        int(i in act_bos), int(i in act_nobos)])
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {corr_bos.size} rows, "
          f"active both = {len(act_bos & act_nobos)})")


def export_spr_points() -> None:
    """SPR / damage points per instrument, plus exact scale stats over the FULL data."""
    scales = {}
    rng = np.random.default_rng(SEED)
    for npz in sorted((FLAT / "spr/instruments").glob("*.npz")):
        with np.load(npz) as d:
            spr = np.asarray(d["spr"], dtype=np.float64)
            tgt = np.asarray(d["target"], dtype=np.float64)
            bits = np.asarray(d["bit_width"], dtype=np.int64)
        scales[npz.stem] = {"n": int(spr.size), "spr_mean": float(spr.mean()),
                            "spr_std": float(spr.std()), "target_mean": float(tgt.mean()),
                            "target_std": float(tgt.std())}
        idx = np.arange(spr.size)
        sampled = False
        if spr.size > CIRCUIT_SAMPLE:
            idx = np.sort(rng.choice(spr.size, CIRCUIT_SAMPLE, replace=False))
            sampled = True
        out = DERIVED / f"spr_points_{npz.stem}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["bit_width", "spr", "target"])
            for i in idx:
                w.writerow([int(bits[i]), f"{spr[i]:.6g}", f"{tgt[i]:.6g}"])
        scales[npz.stem]["csv_rows"] = int(idx.size)
        scales[npz.stem]["csv_subsampled"] = sampled
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {idx.size} rows"
              f"{', subsampled' if sampled else ''})")
    meta = {"note": "scale stats are computed on the FULL instrument data, not the CSV sample",
            "seed": SEED, "source": "results_new/novelty_flat_stream/harmonized/spr/instruments",
            "per_instrument_scales": scales}
    out = DERIVED / "spr_scales.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    export_correlations()
    export_spr_points()


if __name__ == "__main__":
    main()
