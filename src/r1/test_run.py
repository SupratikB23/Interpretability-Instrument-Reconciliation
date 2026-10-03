"""Tests for the staged runner CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DEFAULT_EXPERIMENT_FILES, Config
from src.r1 import run as runner


def _ctx(stage: str, **kwargs) -> runner.RunContext:
    defaults = {
        "config_path": None,
        "dry_run": False,
        "smoke_test": False,
        "override_failed_gate": False,
        "seed": None,
    }
    defaults.update(kwargs)
    return runner.RunContext(stage=stage, **defaults)


def test_dry_run_audit_prints_plan_without_loading_models(capsys):
    rc = runner.main(
        ["audit", "--dry-run", "--config", str(DEFAULT_EXPERIMENT_FILES["smoke"])]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[DRY RUN]" in out
    assert "bit widths" in out
    assert "conditions:" in out
    assert "expected RTN module counts" in out
    assert "expected artifacts:" in out


def test_smoke_test_routes_every_artifact_under_results_smoke():
    cfg = runner.load_config(_ctx("audit", smoke_test=True))
    assert Path(cfg.quant["results"]["root"]).as_posix() == runner.SMOKE_ROOT
    for key, value in cfg.quant["results"].items():
        if key == "root":
            continue
        assert Path(value).as_posix().startswith(runner.SMOKE_ROOT + "/"), (
            f"{key} not rerouted: {value}"
        )


def test_smoke_profile_provenance_is_harmonized_and_valid():
    from src.sae.protocol import HarmonizedProvenance, validate_protocol
    from src.sae.qdm import provenance_from_config

    cfg = Config.from_experiment(DEFAULT_EXPERIMENT_FILES["smoke"])
    built = provenance_from_config(cfg, "harmonized")
    assert isinstance(built, HarmonizedProvenance)
    assert validate_protocol("harmonized", built).value == "harmonized"


def test_reproduction_provenance_is_resolved_and_validates():
    import dataclasses

    from src.sae.protocol import ReferenceProvenance, validate_protocol
    from src.sae.qdm import provenance_from_config

    cfg = Config.from_experiment(DEFAULT_EXPERIMENT_FILES["reproduction"])
    provenance = provenance_from_config(cfg, "reproduction")
    assert isinstance(provenance, ReferenceProvenance)
    assert validate_protocol("reproduction", provenance).value == "reproduction"

    # The hard-stop itself must still fire for any genuinely unresolved field.
    broken = dataclasses.replace(provenance, corpus_revision="")
    with pytest.raises(ValueError, match="unresolved"):
        validate_protocol("reproduction", broken)


def test_harmonized_config_provenance_resolves():
    from src.sae.protocol import validate_protocol
    from src.sae.qdm import provenance_from_config

    cfg = Config.from_experiment(DEFAULT_EXPERIMENT_FILES["harmonized"])
    provenance = provenance_from_config(cfg, "harmonized")
    assert validate_protocol("harmonized", provenance).value == "harmonized"


def test_circuits_and_patching_refuse_to_fabricate_tracer_output(monkeypatch):
    import sys

    import torch

    # Simulate the backend being uninstalled so the guard path is exercised on
    # machines where circuit-tracer really is installed (the GPU box).
    monkeypatch.setitem(sys.modules, "circuit_tracer", None)
    with pytest.raises(RuntimeError, match="(?i)real tracer adapter"):
        runner.stage_circuits(_ctx("circuits"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="(?i)CUDA is unavailable"):
        runner.stage_patching(_ctx("patching"))


def _pythia_quant_modules():
    return {
        "attn_q": ["q_proj", "W_Q"],
        "attn_k": ["k_proj", "W_K"],
        "attn_v": ["v_proj", "W_V"],
        "attn_o": ["o_proj", "W_O"],
        "mlp_in": ["up_proj", "W_in"],
        "mlp_out": ["down_proj", "W_out"],
        "mlp_gate": ["gate_proj", "W_gate"],
    }


def _pythia_quantized_names():
    names = []
    for i in range(6):
        names += [
            f"blocks.{i}.attn.W_Q", f"blocks.{i}.attn.W_K",
            f"blocks.{i}.attn.W_V", f"blocks.{i}.attn.W_O",
            f"blocks.{i}.mlp_in.W_in", f"blocks.{i}.mlp_out.W_out",
        ]
    return names


def test_rtn_group_gate_is_disjoint_under_tl_name_collision():
    """attn W_O is a substring of mlp W_out; the gate must not double-count.

    Regression for the Pythia smoke audit failing `attn_o quantized 12 parameters,
    expected 6` on the real 3070 run: a naive substring match let every W_out
    tensor also land in the attn_o bucket.
    """
    from types import SimpleNamespace

    audit = SimpleNamespace(quantized_names=_pythia_quantized_names())
    expected = {
        "attn_q": 6, "attn_k": 6, "attn_v": 6, "attn_o": 6,
        "mlp_in": 6, "mlp_out": 6, "mlp_gate": 0,
    }
    runner._check_group_counts(audit, _pythia_quant_modules(), expected)

    wrong = SimpleNamespace(quantized_names=_pythia_quantized_names() + ["blocks.6.attn.W_O"])
    with pytest.raises(RuntimeError, match="attn_o"):
        runner._check_group_counts(wrong, _pythia_quant_modules(), expected)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        import csv as _csv

        writer = _csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _fake_harmonized_artifacts(root: Path, *, sae_nonsurvival_pct: float, probe_delta_auc: float) -> None:
    """Minimal real-schema harmonized artifacts for the divergence gate."""
    damaged = 0.0
    survived = 100.0 - sae_nonsurvival_pct - damaged
    _write_csv(
        root / "harmonized" / "sae" / "survival_rates.csv",
        ["bits", "survived_pct", "degraded_pct", "damaged_pct", "mean_corr", "n_active"],
        [[16, 100.0, 0.0, 0.0, 1.0, 100]]
        + [[b, survived, sae_nonsurvival_pct, damaged, 0.9, 100] for b in (8, 6, 4)],
    )
    _write_csv(
        root / "harmonized" / "sae" / "perplexity.csv",
        ["condition", "bits", "ppl", "delta_ppl_percent", "protocol", "n_tokens", "seq_len", "stride", "seed"],
        [["FP16", 16, 10.0, 0.0, "sliding_window", 100, 32, 16, 0]]
        + [[f"INT{b}", b, 11.0, 10.0, "sliding_window", 100, 32, 16, 0] for b in (8, 6, 4)],
    )
    _write_csv(
        root / "harmonized" / "probes" / "auc_calibration.csv",
        ["mode", "monitor", "layer", "seed", "bits", "base_auc", "auc", "eligible",
         "base_brier", "brier", "base_log_loss", "log_loss", "base_ece", "ece"],
        [["harmonized", "harmfulness", 12, 0, b, 0.9, 0.9 + probe_delta_auc, True,
          0.2, 0.2, 0.5, 0.5, 0.05, 0.05] for b in (8, 6, 4)],
    )


def _cfg_rooted_at(root: Path) -> Config:
    cfg = Config.from_experiment(DEFAULT_EXPERIMENT_FILES["smoke"])
    cfg.quant["results"]["root"] = str(root)
    return cfg


def test_divergence_gate_blocks_when_divergence_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_config", lambda ctx: _cfg_rooted_at(tmp_path))
    _fake_harmonized_artifacts(tmp_path, sae_nonsurvival_pct=2.0, probe_delta_auc=-0.01)
    with pytest.raises(RuntimeError, match="divergence NOT observed"):
        runner.stage_divergence_gate(_ctx("divergence-gate"))
    report = tmp_path / "harmonized" / "core_divergence_gate.json"
    assert report.exists(), "gate must persist its diagnostic report even when blocking"


def test_divergence_gate_passes_when_divergence_present(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(runner, "load_config", lambda ctx: _cfg_rooted_at(tmp_path))
    _fake_harmonized_artifacts(tmp_path, sae_nonsurvival_pct=48.7, probe_delta_auc=-0.01)
    runner.stage_divergence_gate(_ctx("divergence-gate"))
    report = json.loads((tmp_path / "harmonized" / "core_divergence_gate.json").read_text())
    assert report["divergence_observed"] is True
    assert set(report["by_bit_width"]) == {"int8", "int6", "int4"}
    row = report["by_bit_width"]["int6"]
    assert row["probe_big_drop_rate"] == 0.0  # -0.01 never crosses the -0.05 threshold
    assert row["sae_nonsurvival_pct"] == 48.7


def test_divergence_gate_detects_dissociation_at_low_bit_width(tmp_path, monkeypatch):
    """The real harmonized case: SAE robust at INT6 but damaged at INT4, probes
    robust throughout. The gate must observe divergence (at INT4) rather than
    anchoring only to INT6 and missing it."""
    import json

    monkeypatch.setattr(runner, "load_config", lambda ctx: _cfg_rooted_at(tmp_path))
    _write_csv(
        tmp_path / "harmonized" / "sae" / "survival_rates.csv",
        ["bits", "survived_pct", "degraded_pct", "damaged_pct", "mean_corr", "n_active"],
        [
            [16, 100.0, 0.0, 0.0, 1.0, 100],
            [8, 99.0, 1.0, 0.0, 0.99, 100],   # 1% non-survival
            [6, 98.0, 2.0, 0.0, 0.9, 100],    # 2% non-survival at the paper's gate point
            [5, 77.0, 23.0, 0.0, 0.8, 100],   # 23% — below the 25% floor
            [4, 43.0, 45.0, 12.0, 0.5, 100],  # 57% non-survival — the real damage
        ],
    )
    _write_csv(
        tmp_path / "harmonized" / "sae" / "perplexity.csv",
        ["condition", "bits", "ppl", "delta_ppl_percent", "protocol", "n_tokens", "seq_len", "stride", "seed"],
        [["FP16", 16, 10.0, 0.0, "sliding_window", 100, 32, 16, 0]]
        + [[f"INT{b}", b, 11.0, 10.0, "sliding_window", 100, 32, 16, 0] for b in (8, 6, 5, 4)],
    )
    _write_csv(
        tmp_path / "harmonized" / "probes" / "auc_calibration.csv",
        ["mode", "monitor", "layer", "seed", "bits", "base_auc", "auc", "eligible",
         "base_brier", "brier", "base_log_loss", "log_loss", "base_ece", "ece"],
        [["harmonized", "harmfulness", 12, 0, b, 0.9, 0.89, True,
          0.2, 0.2, 0.5, 0.5, 0.05, 0.05] for b in (8, 6, 5, 4)],  # delta -0.01, robust
    )
    runner.stage_divergence_gate(_ctx("divergence-gate"))
    report = json.loads((tmp_path / "harmonized" / "core_divergence_gate.json").read_text())
    assert report["divergence_observed"] is True
    assert report["divergence_bit_widths"] == ["int4"]           # detected at the low bit width
    assert report["sae_nonsurvival_pct_at_gate_bits"] == 2.0     # INT6 alone would have missed it


def _write_instrument_npz(root: Path, name: str) -> None:
    import numpy as np

    directory = root / "harmonized" / "spr" / "instruments"
    directory.mkdir(parents=True, exist_ok=True)
    n = 24
    rng = np.random.default_rng(0)
    bit_width = np.repeat([8, 6, 4], n // 3).astype(np.int64)
    np.savez_compressed(
        directory / f"{name}.npz",
        spr=rng.uniform(0.5, 5.0, n), target=rng.uniform(0.0, 1.0, n),
        bit_width=bit_width, signal=rng.uniform(0.1, 1.0, n),
        perturbation=rng.uniform(0.05, 0.5, n),
        groups=np.asarray([f"g{i}" for i in range(n)]), target_kind=np.asarray("continuous"),
    )


def test_spr_loio_writes_all_required_columns(tmp_path, monkeypatch):
    import csv

    for name in ("sae", "probe", "circuits", "patching"):
        _write_instrument_npz(tmp_path, name)
    monkeypatch.setattr(runner, "load_config", lambda ctx: _cfg_rooted_at(tmp_path))
    cfg = _cfg_rooted_at(tmp_path)
    cfg.experiment["statistics"]["bootstrap"]["n_resamples"] = 20
    monkeypatch.setattr(runner, "load_config", lambda ctx: cfg)
    runner.stage_spr_loio(_ctx("spr-loio"))
    path = tmp_path / "harmonized" / "spr" / "leave_one_instrument_out.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    held_out = {row["held_out_instrument"] for row in rows}
    assert held_out == {"sae", "probe", "circuits", "patching"}
    for row in rows:
        assert row["baseline_score"] != "", "baseline_score must be populated"
        assert row["delta"] != ""
        assert int(row["n_train"]) > 0 and int(row["n_test"]) > 0
        assert row["bootstrap_ci_low"] != "" and row["bootstrap_ci_high"] != ""


def test_instruments_flag_restricts_required_artifacts(tmp_path):
    _write_instrument_npz(tmp_path, "sae")
    _write_instrument_npz(tmp_path, "probe")
    cfg = _cfg_rooted_at(tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        runner._load_spr_instruments(cfg)
    instruments = runner._load_spr_instruments(cfg, required=("sae", "probe"))
    assert sorted(instruments) == ["probe", "sae"]


def test_all_rejects_instruments_flag():
    with pytest.raises(ValueError, match="--instruments"):
        runner.stage_all(_ctx("all", instruments=("sae", "probe")))


def test_smoke_all_sequence_covers_every_instrument_and_gate(monkeypatch, tmp_path):
    """Smoke-all must run env, audit, all four instruments, and all SPR stages."""
    calls: list[str] = []
    sequence = ["audit", "harmonized-sae", "harmonized-probes", "divergence-gate",
                "circuits", "patching", "spr-build", "spr-within", "spr-transfer", "spr-loio"]
    monkeypatch.setattr(
        runner, "STAGE_DISPATCH", {name: lambda ctx, _n=name: calls.append(_n) for name in sequence}
    )
    monkeypatch.setattr(runner, "stage_verify_env", lambda ctx, soft=False: calls.append("verify-env"))
    monkeypatch.setattr(runner, "load_config", lambda ctx: _cfg_rooted_at(tmp_path))
    monkeypatch.setattr(runner, "_register_stage_completion", lambda cfg, stage: None)
    runner.stage_all(_ctx("all", smoke_test=True))
    assert calls == ["verify-env", *sequence]


def test_full_run_readiness_is_no_go_without_evidence(tmp_path, monkeypatch):
    import src.config

    monkeypatch.setattr(runner, "load_config", lambda ctx: _cfg_rooted_at(tmp_path))
    monkeypatch.setattr(runner, "_run_test_suite", lambda: (True, "ok"))
    monkeypatch.setattr(src.config, "REPO_ROOT", tmp_path)
    runner.stage_full_run_readiness(_ctx("full-run-readiness"))
    import json

    payload = json.loads((tmp_path / "full_run_readiness.json").read_text())
    assert payload["verdict"] == "NO-GO"
    assert payload["checks"]["sae_reproduction_gate_pass"] == "BLOCKED"
    assert payload["checks"]["all_4_spr_artifact_builders_executable"] == "BLOCKED"


def test_spr_stages_fail_loudly_without_instrument_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SMOKE_ROOT", str(tmp_path / "smoke"))
    cfg = runner.load_config(_ctx("spr-transfer", smoke_test=True))
    with pytest.raises(RuntimeError, match="no SPR instrument artifacts"):
        runner._load_spr_instruments(cfg)


def test_all_stops_at_failed_gate_without_override(monkeypatch):
    calls: list[str] = []

    def make(name):
        def stage(ctx):
            calls.append(name)
            if name == "gate-sae":
                raise AssertionError("gate failed")

        return stage

    sequence = ["audit", "reproduce-sae", "gate-sae", "reproduce-probes", "gate-probes"]
    monkeypatch.setattr(
        runner,
        "STAGE_DISPATCH",
        {name: make(name) for name in sequence},
    )
    cfg = Config.from_experiment(DEFAULT_EXPERIMENT_FILES["reproduction"])
    monkeypatch.setattr(runner, "load_config", lambda ctx: cfg)

    with pytest.raises(AssertionError, match="gate failed"):
        runner.stage_all(_ctx("all"))
    assert calls == ["audit", "reproduce-sae", "gate-sae"]


def test_all_continues_past_failed_gate_with_override(monkeypatch, capsys, tmp_path):
    calls: list[str] = []

    def make(name):
        def stage(ctx):
            calls.append(name)
            if name == "gate-sae":
                raise AssertionError("gate failed")

        return stage

    sequence = ["audit", "reproduce-sae", "gate-sae", "reproduce-probes", "gate-probes"]
    monkeypatch.setattr(
        runner,
        "STAGE_DISPATCH",
        {name: make(name) for name in sequence},
    )
    cfg = Config.from_experiment(DEFAULT_EXPERIMENT_FILES["reproduction"])
    monkeypatch.setattr(runner, "load_config", lambda ctx: cfg)

    import src.config

    monkeypatch.setattr(src.config, "REPO_ROOT", tmp_path)

    runner.stage_all(_ctx("all", override_failed_gate=True))
    assert calls == sequence
    assert "overridden gate failures" in capsys.readouterr().out
    override = tmp_path / "results" / "gate_override.json"
    assert override.exists(), "override flag must be written into result metadata"


def test_main_returns_nonzero_on_stage_failure():
    rc = runner.main(["circuits"])
    assert rc == 1


def test_unknown_stage_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["not-a-stage"])
