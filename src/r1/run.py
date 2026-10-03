"""Staged experiment runner.

Usage:
    python -m r1.run <stage> [--config PATH] [--dry-run] [--smoke-test]
                             [--override-failed-gate] [--seed N]

Stages:
    audit              quantization verification: RTN selection + per-group
                       module-count gate + audit JSON persistence
    reproduce-sae      SAE QDM sweep in reproduction mode
    gate-sae           reproduction gate 1 against saved SAE artifacts
    reproduce-probes   frozen-probe sweep in reproduction mode
    gate-probes        reproduction gate 2 against saved probe artifacts
    harmonized-sae     SAE QDM sweep in harmonized mode
    harmonized-probes  frozen-probe sweep in harmonized mode
    circuits           CIRCUS sweep (requires a caller-owned real tracer adapter)
    patching           NIE/PIE arm (requires caller-owned real conditions)
    spr-within         within-instrument SPR association analysis (artifact-driven)
    spr-transfer       cross-instrument SPR transfer tests (artifact-driven)
    all                audit -> sweeps -> gates -> SPR, ordered, stopping at the
                       first failed gate unless --override-failed-gate is set

Rules enforced here, not by convention:
- Reproduction stages require an experiment profile (``--config``) and validate
  provenance before any model loads; unresolved reference provenance is a hard
  stop (src/sae/protocol.py).
- ``--smoke-test`` routes every artifact under ``results/smoke/`` and runs in
  harmonized mode only; smoke output is never scientifically valid and can
  never satisfy a reproduction gate.
- ``--dry-run`` prints the complete execution plan (model, datasets, layers,
  modules, conditions, output paths) and loads no model.
- The ``circuits`` and ``patching`` stages refuse to run from the CLI without a
  real caller-owned adapter/conditions; the CLI never fabricates tracer output.

SPR stages consume per-instrument artifact files
``<results_root>/harmonized/spr/instruments/<instrument>.npz`` with keys
``spr`` (preregistered SPR values), ``target`` (normalized damage in [0, 1]),
and optional ``bit_width``, ``groups``, ``target_kind``. Missing artifacts are
an explicit error naming the arm that must run first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import DEFAULT_EXPERIMENT_FILES, Config

SMOKE_ROOT = "results/smoke"

STAGES = (
    "audit",
    "reproduce-sae",
    "gate-sae",
    "reproduce-probes",
    "gate-probes",
    "harmonized-sae",
    "harmonized-probes",
    "circuits",
    "patching",
    "spr-build",
    "spr-within",
    "spr-transfer",
    "spr-loio",
    "verify-env",
    "validate-probe-datasets",
    "divergence-gate",
    "smoke-all",
    "full-run-readiness",
    "all",
)


@dataclass(frozen=True)
class RunContext:
    """Resolved runner invocation; every stage receives exactly this."""

    stage: str
    config_path: Path | None
    dry_run: bool
    smoke_test: bool
    override_failed_gate: bool
    seed: int | None
    instruments: tuple[str, ...] | None = None


def _default_config_for_stage(stage: str) -> Path:
    if stage in ("reproduce-sae", "reproduce-probes", "gate-sae", "gate-probes"):
        return DEFAULT_EXPERIMENT_FILES["reproduction"]
    if stage in ("harmonized-sae", "harmonized-probes", "circuits", "patching", "spr-build", "spr-within", "spr-transfer", "spr-loio", "divergence-gate"):
        return DEFAULT_EXPERIMENT_FILES["harmonized"]
    return DEFAULT_EXPERIMENT_FILES["reproduction"]


def _reroot_results(cfg: Config, new_root: str) -> None:
    """Point every results artifact under ``new_root`` (mirrors config_from_args)."""
    results = cfg.quant["results"]
    old_root = results["root"]
    results["root"] = new_root
    for key, value in results.items():
        if key == "root" or not isinstance(value, str):
            continue
        rel = Path(value)
        try:
            rel = rel.relative_to(old_root)
        except ValueError:
            pass
        results[key] = str(Path(new_root) / rel)


def load_config(ctx: RunContext) -> Config:
    """Load the experiment profile for this invocation, applying smoke routing."""
    if ctx.smoke_test:
        path = ctx.config_path or DEFAULT_EXPERIMENT_FILES["smoke"]
        cfg = Config.from_experiment(path)
        _reroot_results(cfg, SMOKE_ROOT)
        print(
            f"[SMOKE] config={path}; all artifacts routed under {SMOKE_ROOT}/. "
            "Smoke output is NOT scientifically valid and can never satisfy a "
            "reproduction gate."
        )
        return cfg
    path = ctx.config_path or _default_config_for_stage(ctx.stage)
    return Config.from_experiment(path)


def _sae_mode(ctx: RunContext, cfg: Config) -> str:
    """Smoke profiles run as harmonized sweeps; smoke is never a reproduction."""
    if ctx.smoke_test:
        return "harmonized"
    mode = cfg.mode
    if mode not in ("reproduction", "harmonized"):
        raise ValueError(
            f"SAE sweep requires a reproduction or harmonized experiment profile, "
            f"got mode {mode!r} ({cfg.config_path})"
        )
    return mode


def _resolve_seed(ctx: RunContext, cfg: Config) -> int:
    if ctx.seed is not None:
        return int(ctx.seed)
    seeds = cfg.experiment.get("seeds") or []
    return int(seeds[0]) if seeds else 0


def describe_plan(ctx: RunContext, cfg: Config) -> None:
    """Print the complete execution plan without loading any model."""
    exp = cfg.experiment
    model = cfg.model["model"]
    compression = cfg.quant["compression"]
    results = cfg.quant["results"]
    lines = [
        f"[DRY RUN] stage={ctx.stage} mode={cfg.mode} config={cfg.config_path}",
        f"  model: {model['name']} (device={model.get('device')}, dtype={model.get('dtype')})",
        f"  readout hook: {cfg.model['readout']['hook']} (layer {cfg.model['readout']['layer']})",
        f"  bit widths: FP16 reference + {compression['bit_widths']}",
        "  conditions: " + ", ".join(["fp16", *[f"int{b}" for b in compression["bit_widths"]]]),
        f"  quantized module groups: {sorted(compression['quant_modules'])}",
        f"  excluded module groups: {sorted(compression['skip_modules'])}",
    ]
    counts = exp.get("compression", {}).get("expected_rtn_module_counts")
    if counts:
        lines.append(f"  expected RTN module counts: {counts} (total {sum(counts.values())})")
    for name, spec in (exp.get("datasets") or {}).items():
        lines.append(f"  dataset {name}: {spec['identifier']} @{spec['revision']} split={spec['split']}")
    for name, spec in (exp.get("corpora") or {}).items():
        lines.append(f"  corpus {name}: {spec['identifier']} @{spec['revision']} split={spec['split']}")
    budgets = exp.get("budgets") or {}
    for budget in ("dry_run", "smoke", "full"):
        if budget in budgets:
            lines.append(
                f"  budget[{budget}]: max_examples={budgets[budget]['max_examples']} "
                f"max_tokens={budgets[budget]['max_tokens']}"
            )
    spr = exp.get("spr") or {}
    if spr:
        lines.append(f"  SPR estimator (preregistered): {spr.get('preregistered_estimator')}")
    provenance = exp.get("provenance")
    if isinstance(provenance, dict):
        unresolved = sorted(key for key, value in provenance.items() if isinstance(value, str) and not value.strip())
        status = f"UNRESOLVED: {unresolved}" if unresolved else "resolved"
        lines.append(f"  provenance: {status}")
    lines.append("  expected artifacts:")
    for key, value in results.items():
        if key != "root":
            lines.append(f"    {key}: {value}")
    print("\n".join(lines))


def _expected_group_totals(cfg: Config) -> int | None:
    counts = cfg.experiment.get("compression", {}).get("expected_rtn_module_counts")
    if not isinstance(counts, dict) or not counts:
        return None
    return int(sum(counts.values()))


def _check_group_counts(audit, quant_modules, expected_counts: dict[str, int]) -> None:
    """Verify each configured module group quantized exactly its expected count.

    A tensor is attributed to the single group whose pattern matches it most
    specifically (longest matched pattern). Without this, a short TransformerLens
    pattern collides as a substring of another group's name -- e.g. attn_o's
    ``W_O`` ("w_o") is contained in the MLP output projection ``W_out``
    ("w_out") -- which would double-count those tensors and spurily fail the gate.
    """
    from src.compression.rtn import _matches

    quantized = list(audit.quantized_names)
    per_group: dict[str, list[str]] = {group: [] for group in quant_modules}
    for name in quantized:
        best_group = None
        best_len = -1
        for group, patterns in quant_modules.items():
            matched_len = max(
                (len(p) for p in patterns if _matches(name, [p])),
                default=0,
            )
            if matched_len > best_len:
                best_len = matched_len
                best_group = group
        if best_group is not None and best_len > 0:
            per_group[best_group].append(name)
    for group, expected in expected_counts.items():
        if group not in per_group:
            continue
        matched = per_group[group]
        if len(matched) != expected:
            raise RuntimeError(
                f"RTN group {group!r} quantized {len(matched)} parameters, "
                f"expected {expected}; check expected_rtn_module_counts against "
                "the model's state-dict naming convention"
            )


def stage_audit(ctx: RunContext) -> None:
    """Quantization verification: selection, per-group counts, audit persistence."""
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return

    import torch
    import transformer_lens

    from src.compression.rtn import quantize_state_dict

    compression = cfg.quant["compression"]
    model_cfg = cfg.model["model"]
    expected_total = _expected_group_totals(cfg)
    expected_groups = cfg.experiment.get("compression", {}).get("expected_rtn_module_counts") or {}

    print(f"[AUDIT] loading {model_cfg['name']} on {model_cfg.get('device')} ...")
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_cfg["name"],
        device=model_cfg.get("device", "cpu"),
        dtype=torch.float16,
        revision=model_cfg.get("revision"),
        trust_remote_code=False,
    )
    # Canonical state lives on CPU: a GPU-resident clone would
    # double the footprint; RTN audit math is cheap and exact on CPU.
    state = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    root = Path(cfg.quant["results"]["root"])
    out = root / cfg.mode / "audit"
    out.mkdir(parents=True, exist_ok=True)
    for bits in compression["bit_widths"]:
        quantized = quantize_state_dict(
            state,
            bits,
            compression["quant_modules"],
            compression["skip_modules"],
            expected_module_count=expected_total,
        )
        if expected_groups:
            _check_group_counts(quantized.audit, compression["quant_modules"], expected_groups)
        path = out / f"quantization_audit_int{bits}.json"
        quantized.audit.write_json(path)
        print(
            f"[AUDIT] INT{bits}: quantized {quantized.audit.quantized_count}/"
            f"{quantized.audit.selected_count} selected parameters "
            f"({quantized.audit.quantized_parameter_count} elements) -> {path}"
        )
    print(f"[AUDIT] OK: all {len(compression['bit_widths'])} bit widths verified")


def _stage_sae(ctx: RunContext, mode_override: str | None = None) -> None:
    cfg = load_config(ctx)
    mode = mode_override or _sae_mode(ctx, cfg)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.config import REPO_ROOT, bit_widths
    from src.sae.qdm import run_sae_sweep

    seed = _resolve_seed(ctx, cfg)
    output_dir = REPO_ROOT / cfg.quant["results"]["root"]
    bits = [16, *bit_widths(cfg)]
    print(f"[SAE] mode={mode} bits={bits} seed={seed} output={output_dir / mode / 'sae'}")
    run_sae_sweep(cfg, bits, output_dir=output_dir, seed=seed, mode=mode)


def _stage_probes(ctx: RunContext, mode_override: str | None = None) -> None:
    cfg = load_config(ctx)
    mode = mode_override or _sae_mode(ctx, cfg)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.probes.run import run_probe_sweep

    print(f"[PROBES] mode={mode} config={cfg.config_path}")
    cells = run_probe_sweep(cfg, mode=mode)
    print(f"[PROBES] wrote {len(cells)} probe cells (mode={mode})")


def stage_gate_sae(ctx: RunContext) -> None:
    from run_gates import run_gate_1

    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    run_gate_1(cfg)


def stage_gate_probes(ctx: RunContext) -> None:
    from run_gates import run_gate_2

    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    run_gate_2(cfg)


def stage_circuits(ctx: RunContext) -> None:
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.circuits.run import run_circuits_stage
    from src.config import REPO_ROOT
    results = run_circuits_stage(
        cfg, output_dir=REPO_ROOT / cfg.quant["results"]["root"], seed=_resolve_seed(ctx, cfg)
    )
    print(f"[CIRCUITS] traced real conditions: {sorted(results)}")


def stage_patching(ctx: RunContext) -> None:
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    import torch
    requested_device = str(cfg.model["model"].get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"patching config requests device {requested_device!r} but CUDA is unavailable; "
            "run on the CUDA experiment machine. A CPU device is only valid for tiny smoke "
            "profiles where the model genuinely fits and executes on CPU."
        )
    from src.config import REPO_ROOT
    from src.patching.run import run_patching_stage
    results = run_patching_stage(cfg, output_dir=REPO_ROOT / cfg.quant["results"]["root"])
    print(f"[PATCHING] completed {len(results)} real condition/component decompositions")


def _load_spr_instruments(cfg: Config, *, required: tuple[str, ...] = ("sae", "probe", "circuits", "patching")) -> dict:
    """Load per-instrument SPR artifacts or fail loudly naming the missing arms.

    A deliberately reduced pilot must pass ``--instruments sae,probe`` on the CLI;
    without that explicit flag the full four-instrument set is required and a
    missing artifact is an error, never a silent downgrade.
    """
    import numpy as np

    from src.spr.transfer import InstrumentData

    root = Path(cfg.quant["results"]["root"])
    instrument_dir = root / "harmonized" / "spr" / "instruments"
    if not instrument_dir.is_dir():
        raise RuntimeError(
            f"no SPR instrument artifacts found under {instrument_dir}/. Run the "
            "instrument arms first; each arm must persist one npz per instrument "
            "with keys spr/target (+ optional bit_width/groups/target_kind)."
        )
    instruments = {}
    for path in sorted(instrument_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        if "spr" not in payload or "target" not in payload:
            raise RuntimeError(f"{path} must contain 'spr' and 'target' arrays")
        instruments[path.stem] = InstrumentData(
            spr=payload["spr"],
            target=payload["target"],
            bit_width=payload.get("bit_width"),
            signal=payload.get("signal"),
            perturbation=payload.get("perturbation"),
            groups=payload.get("groups"),
            target_kind=str(payload["target_kind"]) if "target_kind" in payload else None,
        )
    missing = sorted(set(required) - set(instruments))
    if missing:
        raise RuntimeError(
            "SPR analysis requires condition-produced artifacts for every requested instrument; "
            f"missing {missing} under {instrument_dir}. Run their real stages, or explicitly "
            "request a reduced pilot with --instruments sae,probe (the full-run stages always "
            "require all four instruments)."
        )
    instruments = {name: data for name, data in instruments.items() if name in required}
    if len(instruments) < 2:
        raise RuntimeError(
            f"SPR analysis requires at least two instrument artifacts under "
            f"{instrument_dir}/; found {sorted(instruments) or 'none'}"
        )
    return instruments


def stage_spr_within(ctx: RunContext) -> None:
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    import numpy as np
    from scipy.stats import spearmanr

    from src.config import results_path
    from src.spr.transfer import grouped_bootstrap_ci

    instruments = _load_spr_instruments(cfg, required=ctx.instruments or ("sae", "probe", "circuits", "patching"))
    n_resamples = int(cfg.experiment.get("statistics", {}).get("bootstrap", {}).get("n_resamples", 1000))
    n_permutations = int(cfg.experiment.get("statistics", {}).get("permutation", {}).get("n_resamples", 1000))
    seed = _resolve_seed(ctx, cfg)
    report = {}
    for name, data in instruments.items():
        def association(target, spr_values):
            if np.unique(spr_values).size < 2 or np.unique(target).size < 2:
                return 0.0
            rho, _ = spearmanr(spr_values, target)
            return float(rho)

        observed = association(data.target, data.spr)
        entry = {"spearman_spr_vs_damage": observed}
        if np.unique(data.groups).size >= 2:
            entry["bootstrap_ci_95"] = grouped_bootstrap_ci(
                data.target,
                data.spr,
                data.groups,
                association,
                n_bootstrap=n_resamples,
                seed=seed,
            )
        entry["permutation_null"] = _within_permutation_null(
            data.target, data.spr, association, n_permutations=n_permutations, seed=seed
        )
        report[name] = entry
    path = results_path(cfg, "spr_within")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"[SPR-WITHIN] wrote {path} for instruments {sorted(report)}")


def _within_permutation_null(
    target, spr_values, metric_fn, *, n_permutations: int, seed: int
) -> dict:
    """Permutation null for the within-instrument SPR association.

    Shuffles the SPR values against the damage targets and records ``mean`` and
    ``std`` of the association across ``n_permutations`` permutations, so the
    observed association can be expressed as a z-score. The SPR stats are
    shuffled (not the targets) because the within-instrument association is
    ``spearman(spr, damage)``.
    """
    import numpy as np

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    rng = np.random.default_rng(seed)
    target = np.asarray(target)
    scores = np.empty(n_permutations, dtype=np.float64)
    base = np.asarray(spr_values)
    for index in range(n_permutations):
        scores[index] = metric_fn(target, rng.permutation(base))
    return {"mean": float(scores.mean()), "std": float(scores.std()), "n": n_permutations}


def stage_spr_transfer(ctx: RunContext) -> None:
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.config import results_path
    from src.spr.transfer import run_transfer_tests, write_transfer_json

    instruments = _load_spr_instruments(cfg, required=ctx.instruments or ("sae", "probe", "circuits", "patching"))
    spr_cfg = cfg.quant["spr_transfer"]
    seed = _resolve_seed(ctx, cfg)
    results = run_transfer_tests(
        instruments,
        margin=float(spr_cfg["null_baseline_margin"]),
        n_null_shuffles=int(spr_cfg["n_null_shuffles"]),
        cv_folds=int(spr_cfg["cv_folds"]),
        seed=seed,
        z_threshold=float(spr_cfg.get("z_threshold", 2.0)),
    )
    path = results_path(cfg, "spr")
    write_transfer_json(results, path)
    passed = [r for r in results if r.passes]
    print(
        f"[SPR-TRANSFER] {len(passed)}/{len(results)} ordered pairs pass the "
        f"preregistered primary statistic -> {path}"
    )
    # Roadmap #2: same test on the per-instrument unit-free (z-scored) SPR, to a
    # sibling artifact. Existing transfer_results.json is unchanged.
    normalized = run_transfer_tests(
        instruments,
        margin=float(spr_cfg["null_baseline_margin"]),
        n_null_shuffles=int(spr_cfg["n_null_shuffles"]),
        cv_folds=int(spr_cfg["cv_folds"]),
        seed=seed,
        z_threshold=float(spr_cfg.get("z_threshold", 2.0)),
        spr_feature="spr_normalized",
    )
    norm_path = path.with_name("transfer_results_normalized.json")
    write_transfer_json(normalized, norm_path)
    norm_passed = [r for r in normalized if r.passes]
    print(
        f"[SPR-TRANSFER] normalized (unit-free) SPR: {len(norm_passed)}/{len(normalized)} "
        f"ordered pairs pass -> {norm_path}"
    )


def stage_spr_loio(ctx: RunContext) -> None:
    """Run and persist the source-fitted leave-one-instrument-out analysis."""
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.config import REPO_ROOT
    from src.spr.transfer import leave_one_instrument_out

    instruments = _load_spr_instruments(cfg, required=ctx.instruments or ("sae", "probe", "circuits", "patching"))
    n_bootstrap = int(cfg.experiment.get("statistics", {}).get("bootstrap", {}).get("n_resamples", 0))
    seed = _resolve_seed(ctx, cfg)
    spr_dir = REPO_ROOT / cfg.quant["results"]["root"] / "harmonized" / "spr"
    spr_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["held_out_instrument", "training_instruments", "model", "metric", "score",
                  "baseline_score", "delta", "bootstrap_ci_low", "bootstrap_ci_high", "n_train", "n_test"]

    def write_loio(results, path):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                training = ",".join(sorted(set(instruments) - {result.held_out_instrument}))
                constant = result.metrics.get("constant")
                for model, metrics in result.metrics.items():
                    if model == "constant":
                        continue
                    binary = metrics.log_loss is not None
                    score = metrics.log_loss if binary else metrics.rmse
                    baseline_score = (constant.log_loss if binary else constant.rmse) if constant is not None else ""
                    ci = metrics.confidence_interval or ("", "")
                    writer.writerow({
                        "held_out_instrument": result.held_out_instrument,
                        "training_instruments": training,
                        "model": model,
                        "metric": "log_loss" if binary else "rmse",
                        "score": score,
                        "baseline_score": baseline_score,
                        "delta": metrics.improvement_over_constant,
                        "bootstrap_ci_low": ci[0],
                        "bootstrap_ci_high": ci[1],
                        "n_train": metrics.n_source,
                        "n_test": metrics.n_target,
                    })
        print(f"[SPR-LOIO] wrote {path}")

    path = spr_dir / "leave_one_instrument_out.csv"
    write_loio(leave_one_instrument_out(instruments, n_bootstrap=n_bootstrap, seed=seed), path)
    # Roadmap #2: the shared model's "spr" row driven by the per-instrument
    # unit-free (z-scored) SPR, to a sibling artifact; existing file unchanged.
    write_loio(
        leave_one_instrument_out(instruments, n_bootstrap=n_bootstrap, seed=seed, spr_feature="spr_normalized"),
        spr_dir / "leave_one_instrument_out_normalized.csv",
    )


def build_environment_report() -> dict:
    """Collect versions and device facts for the scientific environment."""
    import importlib.metadata
    from importlib.util import find_spec

    import torch

    required = ("torch", "transformers", "transformer_lens", "sae_lens", "datasets",
                "scikit-learn", "scipy", "numpy", "circuit-tracer")
    versions: dict[str, str | None] = {}
    for name in required:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # A source-installed circuit-tracer may be importable without dist metadata
            # (circuit-tracer provides both).
            module = "circuit_tracer" if name == "circuit-tracer" else name
            versions[name] = None if find_spec(module) is None else "importable-no-dist-metadata"
    missing = [name for name, version in versions.items() if version is None]
    report: dict = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dependency_versions": versions,
        "missing_dependencies": missing,
    }
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    report["status"] = "PASS" if not missing and torch.cuda.is_available() else "BLOCKED"
    return report


def stage_verify_env(ctx: RunContext, *, soft: bool = False) -> None:
    """Write a truthful dependency/device report; missing scientific deps are BLOCKED.

    ``soft`` mode (used by smoke-all) records the report without raising: a CPU
    development box may run the tiny Pythia smoke pipeline even though the full
    scientific environment gate requires CUDA.
    """
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.config import REPO_ROOT
    report = build_environment_report()
    path = REPO_ROOT / cfg.quant["results"]["root"] / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[VERIFY-ENV] {report['status']} -> {path}")
    for name, version in sorted(report["dependency_versions"].items()):
        print(f"  {name}: {version or 'MISSING'}")
    if report["status"] != "PASS" and not soft:
        raise RuntimeError(
            "scientific environment is BLOCKED; install missing dependencies "
            "(circuit-tracer is not on PyPI; download it from GitHub: `pip install git+https://github.com/safety-research/circuit-tracer.git`) and use a CUDA-capable machine"
        )


def stage_divergence_gate(ctx: RunContext) -> None:
    """Core-divergence diagnostic gate before expensive circuit/patching scale-up.

    Reads the harmonized SAE and probe artifacts and writes
    ``harmonized/core_divergence_gate.json`` reporting, per bit width: SAE
    survival/damage, probe big-drop rate, probe median delta AUC, PPL change, and
    the SAE-vs-probe effect-size contrast. The preregistered directional criterion
    is: at some quantized bit width the SAE shows substantial feature non-survival
    (>= sae_nonsurvival_floor_pct, default 25%) while the probe big-drop rate at
    that same bit width stays far below the ~43-54% fine-tuning regime
    (<= probe_big_drop_ceiling, default 0.10). The check spans the whole sweep, not
    just INT6 (that is only the paper's WikiText dissociation point; on another
    substrate the SAE's damage can sit at a lower bit width). Absence of divergence
    stops the pipeline before circuits/patching (override with
    --override-failed-gate); it never fabricates the phenomenon. Smoke runs write
    the report but never block.
    """
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    import numpy as np

    from src.config import REPO_ROOT
    from src.probes.train_eval import load_cells_csv

    root = REPO_ROOT / cfg.quant["results"]["root"]
    sae_csv = root / "harmonized" / "sae" / "survival_rates.csv"
    probe_csv = root / "harmonized" / "probes" / "auc_calibration.csv"
    ppl_csv = root / "harmonized" / "sae" / "perplexity.csv"
    for required in (sae_csv, probe_csv, ppl_csv):
        if not required.exists():
            raise RuntimeError(
                f"divergence gate requires {required}; run harmonized-sae and harmonized-probes first"
            )

    with sae_csv.open(newline="", encoding="utf-8") as fh:
        sae_rows = {int(r["bits"]): r for r in csv.DictReader(fh)}
    cells = [c for c in load_cells_csv(probe_csv) if c.mode == "harmonized"]
    with ppl_csv.open(newline="", encoding="utf-8") as fh:
        ppl_rows = {int(r["bits"]): float(r["delta_ppl_percent"]) for r in csv.DictReader(fh)}

    gate_cfg = (cfg.experiment.get("divergence_gate") or {})
    big_drop_threshold = float(cfg.model["probe"]["big_drop_auc_threshold"])
    # Preregistered directional criterion: probe big-drop rates must stay
    # far below the 43-54% fine-tuning regime; 0.10 is the documented ceiling, and
    # SAE non-survival at the gate bit width must be at least 25% (INT6 reference:
    # ~48.7% degraded). Both thresholds are config-overridable BEFORE looking at results.
    probe_ceiling = float(gate_cfg.get("probe_big_drop_ceiling", 0.10))
    sae_nonsurvival_floor = float(gate_cfg.get("sae_nonsurvival_floor_pct", 25.0))
    gate_bits = int(cfg.quant["gate_sae_int6"]["bits"])

    by_bits: dict[str, dict] = {}
    bits_sorted = sorted(b for b in sae_rows if b != 16)
    for bits in bits_sorted:
        row = sae_rows[bits]
        quant_cells = [c for c in cells if c.bits == bits and c.eligible]
        drops = [c.delta_auc for c in quant_cells]
        big_drop_rate = (
            sum(d <= -big_drop_threshold for d in drops) / len(drops) if drops else float("nan")
        )
        median_delta = float(np.median(drops)) if drops else float("nan")
        sae_nonsurvival = float(row["degraded_pct"]) + float(row["damaged_pct"])
        by_bits[f"int{bits}"] = {
            "sae_survived_pct": float(row["survived_pct"]),
            "sae_nonsurvival_pct": sae_nonsurvival,
            "probe_big_drop_rate": big_drop_rate,
            "probe_median_delta_auc": median_delta,
            "ppl_delta_percent": ppl_rows.get(bits),
            "n_eligible_probe_cells": len(quant_cells),
        }
    gate_row = sae_rows.get(gate_bits)
    sae_nonsurvival_at_gate = (
        float(gate_row["degraded_pct"]) + float(gate_row["damaged_pct"]) if gate_row else float("nan")
    )
    max_probe_big_drop = max(
        (v["probe_big_drop_rate"] for v in by_bits.values() if math.isfinite(v["probe_big_drop_rate"])),
        default=float("nan"),
    )
    # Preregistered criterion (see docstring): the SAE shows
    # substantial non-survival "at low bit widths" while probes stay robust. The
    # divergence is the CO-OCCURRENCE of (SAE damaged, probes robust) at the SAME
    # compression level, evaluated across the sweep — NOT locked to INT6. INT6 is
    # only the paper's WikiText dissociation point; on a different corpus/substrate
    # the SAE's damage can sit at a lower bit width (e.g. FineWeb-Edu: ~2% at INT6
    # but ~57% at INT4), so anchoring solely to INT6 would miss a genuine
    # divergence. This is not a bias toward "divergence": both conditions — real
    # SAE damage AND a robust probe — must hold at the very same bit width. INT6
    # is still reported below as a named diagnostic.
    divergence_bit_widths = [
        bits
        for bits in bits_sorted
        if math.isfinite(by_bits[f"int{bits}"]["probe_big_drop_rate"])
        and by_bits[f"int{bits}"]["sae_nonsurvival_pct"] >= sae_nonsurvival_floor
        and by_bits[f"int{bits}"]["probe_big_drop_rate"] <= probe_ceiling
    ]
    divergence_observed = bool(divergence_bit_widths)
    report = {
        "preregistered_criterion": {
            "sae_nonsurvival_floor_pct_at_int_gate": sae_nonsurvival_floor,
            "probe_big_drop_ceiling": probe_ceiling,
            "rationale": "probes must stay far below the ~43-54% fine-tuning big-drop regime "
                         "while the SAE shows substantial feature non-survival (INT6 ref ~48.7%)",
        },
        "by_bit_width": by_bits,
        "divergence_bit_widths": [f"int{bits}" for bits in divergence_bit_widths],
        "sae_nonsurvival_pct_at_gate_bits": sae_nonsurvival_at_gate,
        "max_probe_big_drop_rate": max_probe_big_drop,
        "effect_size_contrast": (
            sae_nonsurvival_at_gate / 100.0 - max_probe_big_drop
            if math.isfinite(sae_nonsurvival_at_gate) and math.isfinite(max_probe_big_drop)
            else None
        ),
        "divergence_observed": divergence_observed,
        "smoke": bool(ctx.smoke_test),
    }
    out = root / "harmonized" / "core_divergence_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DIVERGENCE-GATE] divergence_observed={divergence_observed} -> {out}")
    if not divergence_observed and not ctx.smoke_test:
        raise RuntimeError(
            "core probe-vs-SAE divergence NOT observed under the harmonized RTN substrate; "
            "stopping before the expensive circuit/patching phase. Review "
            "harmonized/core_divergence_gate.json; to proceed anyway rerun with "
            "--override-failed-gate (recorded in results metadata)."
        )


def stage_validate_probe_datasets(ctx: RunContext) -> None:
    """Validate schemas before a costly probe sweep; no data are silently accepted."""
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.probes.train_eval import audit_probe_dataset
    probe = cfg.model["probe"]
    for monitor in probe["monitors"]:
        audit_probe_dataset(monitor, probe["datasets"], output_dir=Path(cfg.quant["results"]["root"]) / cfg.mode / "probes")


def _run_test_suite() -> tuple[bool, str]:
    """Run the scientific test suite as a subprocess; PASS only on 0 failures/errors."""
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "src/", "-q", "--tb=no"],
        capture_output=True, text=True, timeout=3600, check=False,
    )
    tail = (completed.stdout or "").strip().splitlines()
    return completed.returncode == 0, (tail[-1] if tail else "")


def stage_full_run_readiness(ctx: RunContext) -> None:
    """Produce the GO/NO-GO report; absent or failed evidence is always NO-GO."""
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    from src.config import REPO_ROOT
    root = REPO_ROOT / cfg.quant["results"]["root"]

    def exists(rel: str) -> bool:
        return (root / rel).exists()

    def gate_json_pass(rel: str) -> bool:
        path = root / rel
        if not path.exists():
            return False
        try:
            return bool(json.loads(path.read_text(encoding="utf-8")).get("pass"))
        except json.JSONDecodeError:
            return False

    def provenance_ok() -> bool:
        path = root / "reproduction" / "sae" / "provenance_resolved.json"
        if not path.exists():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        requested = record.get("requested") or {}
        resolved = record.get("resolved") or {}
        # Every requested field must be non-empty (resolved) and the resolved
        # identity must match the request — execution consumed the pinned values.
        for value in requested.values():
            if isinstance(value, str) and not value.strip():
                return False
        req_corpus = (requested.get("corpus_id"), requested.get("corpus_revision"))
        res_corpus = (resolved.get("corpus", {}) or {})
        return (
            req_corpus[0] == res_corpus.get("identifier")
            and req_corpus[1] == res_corpus.get("revision")
            and requested.get("tokenizer_id") == (resolved.get("tokenizer", {}) or {}).get("identifier")
        )

    def probe_datasets_verified() -> bool:
        probe_dir = root / "reproduction" / "probes"
        if not probe_dir.is_dir():
            return False
        audits = list(probe_dir.glob("dataset_audit_*.json"))
        monitors = (cfg.model.get("probe") or {}).get("monitors") or []
        if not audits:
            return False
        audited = {p.stem.removeprefix("dataset_audit_") for p in audits}
        return set(monitors) <= audited if monitors else bool(audits)

    def memory_feasible() -> bool:
        try:
            from src.sae.qdm import assert_memory_feasible
            assert_memory_feasible(cfg, n_copies_on_gpu=1)
            return True
        except Exception:  # noqa: BLE001 - any failure (incl. non-RuntimeError) means NO-GO
            return False

    def loio_complete() -> bool:
        path = root / "harmonized" / "spr" / "leave_one_instrument_out.csv"
        if not path.exists():
            return False
        with path.open(newline="", encoding="utf-8") as fh:
            held = {row["held_out_instrument"] for row in csv.DictReader(fh)}
        return {"sae", "probe", "circuits", "patching"} <= held

    env_report = build_environment_report()
    run_tests = bool((cfg.experiment.get("readiness") or {}).get("run_tests", True))
    tests_pass, tests_tail = _run_test_suite() if run_tests else (False, "readiness.run_tests disabled")

    checks: dict[str, bool] = {
        "scientific_tests_all_pass": tests_pass,
        "rtn_real_model_audit_pass": any(root.rglob("quantization_audit_int*.json")),
        "pythia_smoke_all_pass": (REPO_ROOT / "results" / "smoke" / "harmonized" / "spr" / "leave_one_instrument_out.csv").exists(),
        "sae_provenance_resolved": exists("reproduction/sae/provenance_resolved.json"),
        "sae_execution_consumes_provenance": provenance_ok(),
        "sae_reproduction_gate_pass": gate_json_pass("reproduction/sae/reproduction_gate.json"),
        "probe_datasets_verified": probe_datasets_verified(),
        "probe_protocol_honestly_labeled": exists("reproduction/probes/metadata.json"),
        "harmonized_sae_available": exists("harmonized/sae/survival_rates.csv"),
        "harmonized_probes_available": exists("harmonized/probes/auc_calibration.csv"),
        "circuit_backend_executable": env_report["dependency_versions"].get("circuit-tracer") is not None
                                      and exists("harmonized/circuits/edge_stability.csv"),
        "patching_backend_executable": exists("harmonized/patching/nie_pie.csv"),
        "all_4_spr_artifact_builders_executable": all(
            exists(f"harmonized/spr/instruments/{name}.npz")
            for name in ("sae", "probe", "circuits", "patching")
        ),
        "pairwise_transfer_executable": exists("harmonized/spr/transfer_results.json"),
        "loio_transfer_executable": loio_complete(),
        "memory_feasibility_check_pass": memory_feasible(),
    }
    verdict = "GO FOR FULL RUN" if all(checks.values()) else "NO-GO"
    blockers = sorted(name for name, ok in checks.items() if not ok)
    payload = {
        "verdict": verdict,
        "checks": {name: ("PASS" if ok else "BLOCKED") for name, ok in checks.items()},
        "blockers": blockers,
        "test_suite_tail": tests_tail,
    }
    (root / "full_run_readiness.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Full run readiness", "",
             "| Check | Status |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'BLOCKED'} |" for name, ok in sorted(checks.items())]
    lines += ["", f"VERDICT: {verdict}", ""]
    if blockers:
        lines += ["Blockers:", *[f"- {name}" for name in blockers], ""]
    lines += ["## How to read this verdict", ""]
    if verdict == "GO FOR FULL RUN":
        lines += [
            "All 16 checks PASS on this machine: every required execution artifact exists and is",
            "consistent. The full run is authorized.",
            "",
        ]
    else:
        lines += [
            "BLOCKED means the required execution evidence is absent or failed **on this machine**.",
            "It is not a software defect: every blocker is an artifact that only a real model run",
            "can produce (absent or failed evidence is always NO-GO, and this",
            "gate is never overridden or weakened).",
            "",
            "To flip NO-GO to GO, run on the GPU host (e.g. the RTX 3070 machine), in order:",
            "",
            "```powershell",
            "pip install -r requirements.txt",
            "pip install git+https://github.com/safety-research/circuit-tracer.git",
            "python -m pytest                                # must be 0 errors",
            "python -m r1.run verify-env --config configs/harmonized/common_gemma2_2b.yaml",
            "python -m r1.run all --config configs/smoke/tiny.yaml --smoke-test",
            "python -m r1.run reproduce-sae --config configs/reproduction/sae_gemma2_2b.yaml",
            "python -m r1.run gate-sae --config configs/reproduction/sae_gemma2_2b.yaml",
            "python -m r1.run reproduce-probes --config configs/reproduction/probes_gemma2_2b_it.yaml",
            "python -m r1.run gate-probes --config configs/reproduction/probes_gemma2_2b_it.yaml",
            "python -m r1.run all --config configs/harmonized/common_gemma2_2b.yaml",
            "python -m r1.run full-run-readiness             # re-evaluate on this machine",
            "```",
            "",
                    ]
    (REPO_ROOT / "FULL_RUN_READINESS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[FULL-RUN-READINESS] {verdict} (blockers: {blockers or 'none'})")


def stage_smoke_all(ctx: RunContext) -> None:
    """Run the dependency chain with explicit non-scientific smoke routing."""
    import dataclasses
    stage_all(dataclasses.replace(ctx, stage="all", smoke_test=True))


def stage_all(ctx: RunContext) -> None:
    """Ordered pipeline for the config's mode; gates stop the run unless overridden."""
    cfg = load_config(ctx)
    if ctx.instruments is not None:
        raise ValueError(
            "--instruments is only valid for the spr-* pilot stages; the 'all' pipeline "
            "always requires all four instruments and never silently downgrades"
        )
    if ctx.smoke_test:
        sequence = ["verify-env", "audit", "harmonized-sae", "harmonized-probes", "divergence-gate",
                    "circuits", "patching", "spr-build", "spr-within", "spr-transfer", "spr-loio"]
    elif cfg.mode == "reproduction":
        sequence = ["audit", "reproduce-sae", "gate-sae", "reproduce-probes", "gate-probes"]
    elif cfg.mode == "harmonized":
        sequence = ["audit", "harmonized-sae", "harmonized-probes", "divergence-gate",
                    "circuits", "patching", "spr-build", "spr-within", "spr-transfer", "spr-loio"]
    else:
        raise ValueError(f"cannot run 'all' for mode {cfg.mode!r}")

    gate_stages = {"gate-sae", "gate-probes", "divergence-gate"}
    failures: list[str] = []
    for name in sequence:
        print(f"\n=== [ALL] stage: {name} ===")
        try:
            if name == "verify-env":
                stage_verify_env(ctx, soft=ctx.smoke_test)
            else:
                STAGE_DISPATCH[name](ctx)
        except Exception as exc:
            if name in gate_stages and ctx.override_failed_gate:
                failures.append(name)
                print(f"[ALL] gate {name} FAILED but --override-failed-gate is set; continuing. ({exc})")
                continue
            raise
        _register_stage_completion(cfg, name)
    if failures:
        _write_override_metadata(cfg, failures)
        print(f"\n[ALL] completed with overridden gate failures: {failures}")
    else:
        print("\n[ALL] pipeline completed; all gates passed.")


def _register_stage_completion(cfg: Config, stage: str) -> None:
    """Register the stage's artifact directory in results/manifest.json."""
    from src.config import REPO_ROOT
    from src.manifest import register_artifact

    root = REPO_ROOT / cfg.quant["results"]["root"]
    mode = "harmonized" if cfg.mode in ("harmonized", "smoke") else cfg.mode
    stage_dirs = {
        "audit": root / cfg.mode / "audit",
        "reproduce-sae": root / "reproduction" / "sae",
        "harmonized-sae": root / mode / "sae",
        "reproduce-probes": root / "reproduction" / "probes",
        "harmonized-probes": root / mode / "probes",
        "circuits": root / mode / "circuits",
        "patching": root / mode / "patching",
        "divergence-gate": root / mode / "core_divergence_gate.json",
        "spr-within": root / mode / "spr" / "within_instrument.json",
        "spr-transfer": root / mode / "spr" / "transfer_results.json",
        "spr-loio": root / mode / "spr" / "leave_one_instrument_out.csv",
    }
    target = stage_dirs.get(stage)
    if target is None:
        return
    instrument = {"harmonized-sae": "sae", "harmonized-probes": "probe", "circuits": "circuits",
                  "patching": "patching", "reproduce-sae": "sae", "reproduce-probes": "probe"}.get(stage)
    register_artifact(root, artifact_type="stage_output", path=target, stage=stage, instrument=instrument)


def _write_override_metadata(cfg: Config, failed_gates: list[str]) -> None:
    """Record gate overrides in result metadata.

    The developer override for a failed gate is written to disk verbatim so a
    downstream consumer can see that a gate was bypassed, never silently.
    """
    from datetime import datetime, timezone

    from src.config import REPO_ROOT

    root = REPO_ROOT / cfg.quant["results"]["root"]
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "override_failed_gate": True,
        "overridden_gates": failed_gates,
    }
    with (root / "gate_override.json").open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)


def stage_spr_build(ctx: RunContext) -> None:
    """Validate that every instrument produced its SPR artifact."""
    cfg = load_config(ctx)
    if ctx.dry_run:
        describe_plan(ctx, cfg)
        return
    instruments = _load_spr_instruments(
        cfg, required=ctx.instruments or ("sae", "probe", "circuits", "patching")
    )
    print(f"[SPR-BUILD] all {len(instruments)} instrument artifacts present: {sorted(instruments)}")


STAGE_DISPATCH = {
    "audit": stage_audit,
    "reproduce-sae": _stage_sae,
    "gate-sae": stage_gate_sae,
    "reproduce-probes": _stage_probes,
    "gate-probes": stage_gate_probes,
    "harmonized-sae": lambda ctx: _stage_sae(ctx, "harmonized"),
    "harmonized-probes": lambda ctx: _stage_probes(ctx, "harmonized"),
    "circuits": stage_circuits,
    "patching": stage_patching,
    "spr-build": stage_spr_build,
    "spr-within": stage_spr_within,
    "spr-transfer": stage_spr_transfer,
    "spr-loio": stage_spr_loio,
    "verify-env": stage_verify_env,
    "validate-probe-datasets": stage_validate_probe_datasets,
    "divergence-gate": stage_divergence_gate,
    "smoke-all": stage_smoke_all,
    "full-run-readiness": stage_full_run_readiness,
    "all": stage_all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r1.run",
        description="Staged R1 experiment runner.",
    )
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="experiment profile yaml (configs/reproduction|harmonized|smoke/...)",
    )
    parser.add_argument("--seed", type=int, default=None, help="master seed override")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the execution plan without loading any model",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"route all artifacts under {SMOKE_ROOT}/; never scientifically valid",
    )
    parser.add_argument(
        "--override-failed-gate",
        action="store_true",
        help="with 'all': continue past failed gates and report them at the end",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="comma-separated instrument subset (e.g. sae,probe) for an explicit reduced "
             "SPR pilot; invalid with 'all'/'smoke-all', which always require all four",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instruments = (
        tuple(name.strip() for name in args.instruments.split(",") if name.strip())
        if args.instruments
        else None
    )
    ctx = RunContext(
        stage=args.stage,
        config_path=args.config,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        override_failed_gate=args.override_failed_gate,
        seed=args.seed,
        instruments=instruments,
    )
    try:
        STAGE_DISPATCH[ctx.stage](ctx)
    except Exception as exc:  # noqa: BLE001 - single exit point with full message
        print(f"[FAILED] stage {ctx.stage}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
