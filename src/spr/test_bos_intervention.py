"""Unit tests for the BOS-as-intervention SPR analysis."""

from __future__ import annotations

import numpy as np

from src.spr.bos_intervention import analyze_bos_intervention


def _write_sae_like(path, feature_ids, spr, corr, bits=6):
    n = len(feature_ids)
    np.savez_compressed(
        path,
        bit_width=np.full(n, bits, dtype=np.int64),
        feature_id=np.asarray(feature_ids, dtype=np.int64),
        spr=np.asarray(spr, dtype=np.float64),
        correlation=np.asarray(corr, dtype=np.float64),
    )


def test_bos_intervention_detects_cofluctuation(tmp_path):
    # 10 shared features: the BOS raises SPR and survival TOGETHER, monotonically,
    # so ΔSPR and Δcorr should rank-correlate ~+1 (the intervention moving damage
    # exactly the way SPR predicts). Feature 99 is active only BOS-absent -> dropped.
    feats = list(range(10))
    nobos_spr = np.linspace(1.0, 2.0, 10)
    nobos_corr = np.linspace(0.30, 0.55, 10)
    gain = np.linspace(0.1, 3.0, 10)          # per-feature SPR gain from adding BOS
    bos_spr = nobos_spr + gain
    bos_corr = np.clip(nobos_corr + 0.15 * gain, 0, 1)  # survival rises with the gain

    _write_sae_like(tmp_path / "bos.npz", feats, bos_spr, bos_corr)
    _write_sae_like(tmp_path / "nobos.npz", feats + [99], list(nobos_spr) + [9.0],
                    list(nobos_corr) + [0.9])

    report = analyze_bos_intervention(tmp_path / "bos.npz", tmp_path / "nobos.npz", bits_list=(6,))
    r = report["per_bit"]["int6"]
    assert r["n_matched"] == 10                      # feature 99 excluded
    assert r["spearman_dspr_dcorr"] > 0.9            # ΔSPR predicts Δsurvival
    assert r["frac_spr_raised_by_bos"] == 1.0        # BOS raised SPR for all
    assert r["survived_pct_bos"] >= r["survived_pct_nobos"]


def test_bos_intervention_handles_thin_overlap(tmp_path):
    _write_sae_like(tmp_path / "bos.npz", [1], [2.0], [0.9])
    _write_sae_like(tmp_path / "nobos.npz", [2], [1.0], [0.5])
    report = analyze_bos_intervention(tmp_path / "bos.npz", tmp_path / "nobos.npz", bits_list=(6,))
    assert report["per_bit"]["int6"]["n_matched"] == 0
