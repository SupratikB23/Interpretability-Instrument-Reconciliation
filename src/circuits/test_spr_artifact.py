"""CIRCUS output must emit a real, labelled transfer artifact."""

from __future__ import annotations

import numpy as np

from src.circuits.extract import CircusResult, _write_circuit_spr_artifact
from src.circuits.stability import CircusConfig


def _result(label, inclusion):
    return CircusResult(label, f"state-{label}", [CircusConfig("ci-01", .55, .95, 0)] * 9, set(), inclusion, f"hash-{label}")


def test_stable_core_edge_spr_is_bounded_by_resolution(tmp_path):
    # A core edge unchanged by quantization has Δinclusion=0. With a 1e-8 floor this
    # produced SPR≈1e8 and destroyed the transfer RMSE; the floor is now the
    # ensemble resolution 1/B (B=9 configs here), so SPR is capped at signal*B = 9.
    fp = _result("fp16", {("a", "b"): 1.0})
    q4 = _result("int4", {("a", "b"): 1.0})  # perfectly stable -> Δ = 0
    path = _write_circuit_spr_artifact({"fp16": fp, "int4": q4}, tmp_path, contingent_min=0.5)
    data = np.load(path)
    assert data["spr"].tolist() == [9.0]  # 1.0 / (1/9), NOT 1e8
    assert data["perturbation"].tolist() == [0.0]  # raw perturbation is kept honest
    assert float(data["perturbation_floor"]) == 1.0 / 9.0


def test_circuit_spr_artifact_is_condition_specific(tmp_path):
    # ("a","b") is core in FP16 (s=1.0) -> a valid SPR sample. ("c","d") is FP16
    # noise (s<0.5) and ("b","c") is not present in FP16 at all; SPR conditions on
    # FP16 readout structure, so neither may be scored.
    fp = _result("fp16", {("a", "b"): 1.0, ("c", "d"): 0.25})
    q4 = _result("int4", {("a", "b"): .5, ("b", "c"): .4})
    path = _write_circuit_spr_artifact({"fp16": fp, "int4": q4}, tmp_path, contingent_min=0.5)
    data = np.load(path)
    assert data["groups"].tolist() == ["a->b"]
    assert data["bit_width"].tolist() == [4]
    assert data["direction_source"].item() == "CIRCUS_stability_approximation"
