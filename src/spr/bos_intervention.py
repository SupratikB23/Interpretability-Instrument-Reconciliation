"""BOS-as-intervention test of the SPR mechanism (leg 1 -> causal test of leg 2).

The BOS attention sink is a single controllable knob applied to every feature at
once: with it, activations are stabilized, so the compression perturbation along a
readout direction shrinks (SPR denominator down -> SPR up), and features survive
better; without it, SPR falls and features are damaged.

That makes the BOS an *intervention* on SPR. If SPR is the mechanism (not just a
correlate), then the per-feature CHANGE in SPR caused by toggling the BOS should
predict the per-feature CHANGE in survival. This module measures exactly that from
the two already-written SAE SPR artifacts -- no model re-run, no GPU.

Inputs are the ``sae.npz`` instrument artifacts from the two substrates:
  * BOS-present : results/harmonized/spr/instruments/sae.npz         (~98.7% INT6)
  * BOS-absent  : results/novelty_flat_stream/.../sae.npz            (~55% INT6)

Both store, per (bit_width, feature): ``spr`` (= signal/perturbation), ``signal``,
``perturbation``, and ``correlation``. Features are matched by ``feature_id`` and
intersected (the active sets differ: BOS-present has more active features).

Run:
    python -m src.spr.bos_intervention \
        results/harmonized/spr/instruments/sae.npz \
        results/novelty_flat_stream/harmonized/spr/instruments/sae.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_SURVIVED = 0.9  # Duan 2606.03002 survival threshold c_j > 0.9


def _feature_maps(path: Path, bits: int) -> dict[int, tuple[float, float]]:
    """Return {feature_id: (spr, correlation)} for one bit width in one artifact."""
    with np.load(path, allow_pickle=True) as handle:
        bit_width = np.asarray(handle["bit_width"]).astype(int)
        feature_id = np.asarray(handle["feature_id"]).astype(int)
        spr = np.asarray(handle["spr"]).astype(np.float64)
        corr = np.asarray(handle["correlation"]).astype(np.float64)
    mask = bit_width == bits
    return {int(f): (float(s), float(c)) for f, s, c in zip(feature_id[mask], spr[mask], corr[mask])}


def analyze_bos_intervention(
    bos_path: str | Path, nobos_path: str | Path, *, bits_list: tuple[int, ...] = (8, 7, 6, 5, 4)
) -> dict:
    """Per-bit BOS intervention analysis matched by feature.

    Headline statistic per bit: Spearman(ΔSPR, Δcorrelation) over features active in
    BOTH substrates, where Δ = (BOS-present) − (BOS-absent). A strong positive value
    means the features whose SPR the BOS raised most are exactly those whose survival
    improved most -- the intervention moving damage the way SPR predicts.
    """
    bos_path, nobos_path = Path(bos_path), Path(nobos_path)
    per_bit: dict[str, dict] = {}
    for bits in bits_list:
        bos = _feature_maps(bos_path, bits)
        nobos = _feature_maps(nobos_path, bits)
        shared = sorted(set(bos) & set(nobos))
        if len(shared) < 3:
            per_bit[f"int{bits}"] = {"n_matched": len(shared), "note": "too few shared features"}
            continue
        d_spr = np.array([bos[f][0] - nobos[f][0] for f in shared])
        d_corr = np.array([bos[f][1] - nobos[f][1] for f in shared])
        spr_bos = np.array([bos[f][0] for f in shared])
        spr_nobos = np.array([nobos[f][0] for f in shared])
        corr_bos = np.array([bos[f][1] for f in shared])
        corr_nobos = np.array([nobos[f][1] for f in shared])
        rho = spearmanr(d_spr, d_corr).statistic if np.unique(d_spr).size > 1 else float("nan")
        per_bit[f"int{bits}"] = {
            "n_matched": len(shared),
            "spearman_dspr_dcorr": float(rho),
            "frac_spr_raised_by_bos": float(np.mean(d_spr > 0)),
            "frac_corr_raised_by_bos": float(np.mean(d_corr > 0)),
            "median_spr_bos": float(np.median(spr_bos)),
            "median_spr_nobos": float(np.median(spr_nobos)),
            "mean_corr_bos": float(np.mean(corr_bos)),
            "mean_corr_nobos": float(np.mean(corr_nobos)),
            "survived_pct_bos": float(100.0 * np.mean(corr_bos > _SURVIVED)),
            "survived_pct_nobos": float(100.0 * np.mean(corr_nobos > _SURVIVED)),
        }
    return {
        "bos_present_artifact": str(bos_path),
        "bos_absent_artifact": str(nobos_path),
        "survived_threshold": _SURVIVED,
        "per_bit": per_bit,
        "headline_bit": "int6",
    }


def _print_summary(report: dict) -> None:
    print(f"[BOS-INTERVENTION] BOS-present: {report['bos_present_artifact']}")
    print(f"[BOS-INTERVENTION] BOS-absent : {report['bos_absent_artifact']}")
    print("  bit  n_match  rho(ΔSPR,Δcorr)  SPR bos/nobos(med)  survived% bos/nobos")
    for bit, r in report["per_bit"].items():
        if "spearman_dspr_dcorr" not in r:
            print(f"  {bit}: {r.get('note', 'n/a')} (n={r['n_matched']})")
            continue
        print(
            f"  {bit}: {r['n_matched']:>6}  {r['spearman_dspr_dcorr']:+.3f}          "
            f"{r['median_spr_bos']:.2f}/{r['median_spr_nobos']:.2f}        "
            f"{r['survived_pct_bos']:.1f}/{r['survived_pct_nobos']:.1f}"
        )
    h = report["per_bit"].get(report["headline_bit"], {})
    if "spearman_dspr_dcorr" in h:
        print(
            f"[BOS-INTERVENTION] INT6: toggling the BOS moved per-feature SPR and survival "
            f"together at rho={h['spearman_dspr_dcorr']:+.3f} "
            f"({h['frac_spr_raised_by_bos'] * 100:.0f}% of features gained SPR, "
            f"{h['frac_corr_raised_by_bos'] * 100:.0f}% gained survival)."
        )


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        raise SystemExit(
            "usage: python -m src.spr.bos_intervention <bos_present_sae.npz> <bos_absent_sae.npz> [out.json]"
        )
    report = analyze_bos_intervention(argv[0], argv[1])
    _print_summary(report)
    out = Path(argv[2]) if len(argv) == 3 else Path(argv[1]).with_name("bos_spr_intervention.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[BOS-INTERVENTION] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
