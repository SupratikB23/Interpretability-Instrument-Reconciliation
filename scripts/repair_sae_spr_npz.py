"""One-time repair for an SAE SPR instrument artifact with non-finite targets.

Background: the SAE SPR writer used ``np.clip(1 - corr, 0, 1)`` for the target,
and ``np.clip`` does not sanitize NaN. A degenerate feature (zero variance under
quantization) has an undefined Pearson correlation (0/0 -> NaN), so its target was
NaN, and ``spr-within`` correctly refused the artifact ("target contains
non-finite values"). ``src/sae/qdm.py`` now pins such correlations to 0.0 at write
time, so fresh runs are clean. This script repairs an ALREADY-WRITTEN sae.npz so a
completed (multi-hour) sweep does not have to be re-run.

A degenerate feature is maximal damage: correlation -> 0.0, target -> 1.0. Every
other array and key is preserved exactly. Only the SAE artifact needs this; the
loader confirmed spr/signal/perturbation are already finite.

Usage (from the repo root):
    python scripts/repair_sae_spr_npz.py results/novelty_flat_stream/harmonized/spr/instruments/sae.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def repair(path: Path) -> int:
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    with np.load(path, allow_pickle=True) as handle:
        data = {key: handle[key] for key in handle.files}

    corr = np.asarray(data.get("correlation"), dtype=np.float64)
    # Degenerate features: any non-finite correlation (NaN from 0/0, or +/-inf)
    # -> 0.0 (maximal damage). Matches the writer fix in src/sae/qdm.py exactly.
    clean_corr = np.where(np.isfinite(corr), corr, 0.0)
    n_fixed = int(np.count_nonzero(~np.isfinite(corr)))

    data["correlation"] = clean_corr
    if "raw_target" in data:
        data["raw_target"] = clean_corr
    data["target"] = np.clip(1.0 - np.clip(clean_corr, 0.0, 1.0), 0.0, 1.0)

    # Defensive: any residual non-finite in the numeric analysis arrays.
    for key in ("spr", "signal", "perturbation"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float64)
            data[key] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.all(np.isfinite(data["target"])):
        raise SystemExit("repair failed: target still non-finite")

    np.savez_compressed(path, **data)
    print(f"[REPAIR] {path}: sanitized {n_fixed} non-finite correlation row(s); "
          f"target now finite ({data['target'].size} rows).")
    return n_fixed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/repair_sae_spr_npz.py <path-to-sae.npz>")
    repair(Path(sys.argv[1]))
