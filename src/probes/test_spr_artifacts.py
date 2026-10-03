"""Regression tests for real frozen-probe direction/SPR persistence."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.probes.run import _append_probe_spr_rows, _write_probe_direction, _write_probe_spr_artifact
from src.probes.train_eval import FrozenProbe, ProbeCell, ProbeMetrics


def _frozen() -> FrozenProbe:
    estimator = LogisticRegression().fit(np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), np.array([0, 1, 1, 0]))
    return FrozenProbe(estimator, np.array([0, 1], dtype=np.int64), np.array([0, 1]), ProbeMetrics(.9, .1, .2, .1), 7, .75, 10)


def test_probe_direction_and_spr_artifact_are_persisted(tmp_path):
    frozen = _frozen()
    direction = _write_probe_direction(tmp_path / "directions", monitor="harm", layer=12, frozen=frozen)
    assert np.load(direction)["w_probe"].shape == (2,)
    rows: list[dict[str, object]] = []
    cell = ProbeCell("harm", 12, 7, 4, .9, .8, True)
    _append_probe_spr_rows(rows, frozen=frozen, x_fp16=np.array([[1., 0.], [0., 1.]]), x_quant=np.array([[.9, .1], [.2, .8]]), cell=cell, damage_scale=.05)
    output = _write_probe_spr_artifact(rows, tmp_path)
    artifact = np.load(output)
    assert {"spr", "target", "bit_width", "groups", "signal", "perturbation"} <= set(artifact.files)
    assert artifact["bit_width"].tolist() == [4]
