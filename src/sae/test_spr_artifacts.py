"""Feature-level SAE SPR artifact contract tests."""

from __future__ import annotations

import numpy as np

from src.sae.qdm import _write_sae_spr_artifact


def test_sae_spr_artifact_contains_all_configured_quantized_bits(tmp_path):
    path = _write_sae_spr_artifact(
        output_dir=tmp_path,
        correlations={16: np.ones(3), 8: np.array([.9, .8, .7]), 4: np.array([.6, .5, .4])},
        firing_rates=np.array([.01, .02, 0.]),
        directional={8: (np.array([2., 3., 4.]), np.array([.1, .2, .3])), 4: (np.array([2., 3., 4.]), np.array([.3, .4, .5]))},
        active_threshold=.001,
    )
    artifact = np.load(path)
    assert set(artifact["bit_width"].tolist()) == {4, 8}
    assert artifact["feature_id"].tolist() == [0, 1, 0, 1]
    assert np.all(artifact["target"] >= 0)


def test_sae_spr_artifact_sanitizes_nonfinite_correlation(tmp_path):
    # A degenerate feature (zero variance under quantization) yields a NaN Pearson
    # correlation. np.clip does not sanitize NaN, so the target must be pinned to
    # finite maximal damage (1.0) or the SPR loader rejects the whole artifact.
    path = _write_sae_spr_artifact(
        output_dir=tmp_path,
        correlations={8: np.array([0.95, np.nan]), 4: np.array([np.inf, 0.30])},
        firing_rates=np.array([0.01, 0.02]),
        directional={8: (np.array([2.0, 3.0]), np.array([0.1, 0.2])),
                     4: (np.array([2.0, 3.0]), np.array([0.3, 0.4]))},
        active_threshold=0.001,
    )
    artifact = np.load(path)
    assert np.all(np.isfinite(artifact["target"]))
    assert np.all(np.isfinite(artifact["correlation"]))
    # NaN/inf correlation -> corr 0.0 -> target 1.0, classed "damaged".
    rows = {(int(b), int(f)): t for b, f, t in
            zip(artifact["bit_width"], artifact["feature_id"], artifact["target"])}
    assert rows[(8, 1)] == 1.0        # NaN corr -> full damage
    assert rows[(4, 0)] == 1.0        # inf corr -> full damage
    classes = artifact["survival_class"].tolist()
    assert classes[artifact["feature_id"].tolist().index(1)] == "damaged"
