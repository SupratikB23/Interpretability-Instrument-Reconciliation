"""Configuration loader for R1.

Config lives in ``configs/*.yaml`` only — nothing hardcoded in scripts. This module loads the three fixed config files and merges
optional argparse overrides on top. Plain yaml + argparse (not hydra) is the
single config mechanism for ALL arms.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

DEFAULT_CONFIG_FILES = {
    "model": CONFIG_DIR / "model_gemma2_2b.yaml",
    "quant": CONFIG_DIR / "quant_rtn_sweep.yaml",
}
DEFAULT_EXPERIMENT_FILES = {
    "reproduction": CONFIG_DIR / "reproduction" / "sae_gemma2_2b.yaml",
    "harmonized": CONFIG_DIR / "harmonized" / "common_gemma2_2b.yaml",
    "smoke": CONFIG_DIR / "smoke" / "tiny.yaml",
}
ExperimentMode = Literal["reproduction", "harmonized", "smoke"]


class ConfigValidationError(ValueError):
    """Raised when a scientific configuration violates the experiment contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML config file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Config:
    """Merged R1 configuration.

    Attributes:
        model: primary model config (default Gemma-2-2B).
        quant: RTN sweep + gate target + taxonomy config.
        secondary_model: Pythia-70M config (loaded on demand).
    """

    model: dict[str, Any]
    quant: dict[str, Any]
    secondary_model: dict[str, Any]
    experiment: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @classmethod
    def from_yaml(
        cls,
        model_file: Path | str | None = None,
        quant_file: Path | str | None = None,
        secondary_file: Path | str | None = None,
    ) -> Config:
        return cls(
            model=load_yaml(Path(model_file) if model_file else DEFAULT_CONFIG_FILES["model"]),
            quant=load_yaml(Path(quant_file) if quant_file else DEFAULT_CONFIG_FILES["quant"]),
            secondary_model=load_yaml(
                Path(secondary_file)
                if secondary_file
                else CONFIG_DIR / "model_pythia70m.yaml"
            ),
        )

    @classmethod
    def from_experiment(cls, path: Path | str) -> Config:
        """Load and validate a mode-specific experiment profile.

        Profiles overlay the legacy model and quant files, so existing call sites can
        continue to consume ``cfg.model`` and ``cfg.quant`` unchanged.
        """
        profile_path = Path(path)
        raw = load_yaml(profile_path)
        if not isinstance(raw, dict):
            raise ConfigValidationError(f"config {profile_path} must contain a YAML mapping")
        cfg = cls.from_yaml()
        cfg.experiment = copy.deepcopy(raw)
        cfg.config_path = profile_path.resolve()
        if isinstance(raw.get("model_config"), dict):
            cfg.model = deep_merge(cfg.model, raw["model_config"])
        if isinstance(raw.get("quant_config"), dict):
            cfg.quant = deep_merge(cfg.quant, raw["quant_config"])
        cfg.validate()
        return cfg

    @property
    def mode(self) -> str:
        return str(self.experiment.get("mode", "legacy"))

    def snapshot(self) -> dict[str, Any]:
        """Return the complete JSON-serializable configuration snapshot."""
        return {
            "model": copy.deepcopy(self.model),
            "quant": copy.deepcopy(self.quant),
            "secondary_model": copy.deepcopy(self.secondary_model),
            "experiment": copy.deepcopy(self.experiment),
        }

    def config_hash(self) -> str:
        """Stable SHA-256 over the complete resolved configuration."""
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self, expected_mode: ExperimentMode | None = None) -> None:
        """Validate explicit provenance, budgets, RTN, SPR, and statistics fields."""
        if not self.experiment:
            _validate_legacy_config(self)
            return
        _validate_experiment(self.experiment, expected_mode)


def _require(mapping: dict[str, Any], path: str, expected: type | tuple[type, ...]) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ConfigValidationError(f"missing required config field: {path}")
        value = value[component]
    if not isinstance(value, expected) or isinstance(value, bool) and expected is int:
        names = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
        raise ConfigValidationError(f"config field {path} must be {names}, got {type(value).__name__}")
    return value


def _positive_int(mapping: dict[str, Any], path: str, *, allow_zero: bool = False) -> int:
    value = _require(mapping, path, int)
    if value < (0 if allow_zero else 1):
        raise ConfigValidationError(f"config field {path} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _validate_provenance(name: str, spec: Any) -> None:
    if not isinstance(spec, dict):
        raise ConfigValidationError(f"dataset/corpus {name!r} must be an explicit provenance mapping")
    for key in ("identifier", "revision", "split", "source"):
        value = spec.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(f"dataset/corpus {name!r} requires non-empty provenance field {key!r}")


_PLACEHOLDER_DATASET_IDS = frozenset({"should/should", "synthetic/high-stakes-monitor"})


def _reject_placeholder_dataset_specs(raw: dict[str, Any]) -> None:
    """Refuse identifiers which previously made a long probe run non-reproducible.

    Dataset execution uses the nested ``model_config.probe.datasets`` contract,
    while the top-level block records experiment provenance.  Validate both so
    neither metadata nor the actual loader can retain a cosmetic placeholder.
    """
    candidates: list[tuple[str, Any]] = list((raw.get("datasets") or {}).items())
    probe = (raw.get("model_config") or {}).get("probe") or {}
    candidates.extend((probe.get("datasets") or {}).items())
    for name, spec in candidates:
        identifier = spec.get("dataset_id", spec.get("identifier")) if isinstance(spec, dict) else None
        if isinstance(identifier, str) and identifier.strip().lower() in _PLACEHOLDER_DATASET_IDS:
            raise ConfigValidationError(
                f"dataset {name!r} uses unresolved placeholder identifier {identifier!r}; "
                "supply a verified source dataset or a versioned local generator before running probes"
            )


def _validate_experiment(raw: dict[str, Any], expected_mode: ExperimentMode | None) -> None:
    mode = _require(raw, "mode", str)
    if mode not in DEFAULT_EXPERIMENT_FILES:
        raise ConfigValidationError(f"mode must be one of {sorted(DEFAULT_EXPERIMENT_FILES)}, got {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ConfigValidationError(f"expected mode {expected_mode!r}, config declares {mode!r}")
    _require(raw, "schema_version", int)
    seeds = _require(raw, "seeds", list)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ConfigValidationError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ConfigValidationError("seeds must not contain duplicates")
    for path in (
        "model.identifier", "model.tokenizer_identifier", "model.revision",
        "readout.hook", "compression.method", "compression.scheme",
        "spr.signal_statistic", "spr.perturbation_statistic",
        "statistics.bootstrap.unit", "statistics.multiple_testing.method",
    ):
        value = _require(raw, path, str)
        if not value.strip():
            raise ConfigValidationError(f"config field {path} must not be empty")
    _positive_int(raw, "readout.layer", allow_zero=True)
    bits = _require(raw, "compression.bit_widths", list)
    if not bits or any(isinstance(bit, bool) or not isinstance(bit, int) or not 2 <= bit <= 16 for bit in bits):
        raise ConfigValidationError("compression.bit_widths must contain integers in [2, 16]")
    modules = _require(raw, "compression.quantized_modules", list)
    excluded = _require(raw, "compression.excluded_modules", list)
    if not modules or not all(isinstance(item, str) and item for item in modules):
        raise ConfigValidationError("compression.quantized_modules must be a non-empty string list")
    if set(modules) & set(excluded):
        raise ConfigValidationError("quantized_modules and excluded_modules must be disjoint")
    counts = _require(raw, "compression.expected_rtn_module_counts", dict)
    if not counts or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ConfigValidationError("expected_rtn_module_counts values must be non-negative integers")
    datasets = _require(raw, "datasets", dict)
    corpora = _require(raw, "corpora", dict)
    if not datasets or not corpora:
        raise ConfigValidationError("datasets and corpora must each declare at least one provenance entry")
    for name, spec in {**datasets, **corpora}.items():
        _validate_provenance(name, spec)
    _reject_placeholder_dataset_specs(raw)
    for budget in ("dry_run", "smoke", "full"):
        _positive_int(raw, f"budgets.{budget}.max_examples")
        _positive_int(raw, f"budgets.{budget}.max_tokens")
    if raw["budgets"]["smoke"]["max_tokens"] >= raw["budgets"]["full"]["max_tokens"]:
        raise ConfigValidationError("smoke max_tokens must be smaller than full max_tokens")
    _positive_int(raw, "statistics.bootstrap.n_resamples")
    _positive_int(raw, "statistics.permutation.n_resamples")
    confidence = _require(raw, "statistics.confidence_level", (int, float))
    if not 0 < float(confidence) < 1:
        raise ConfigValidationError("statistics.confidence_level must be between 0 and 1")
    estimator = _require(raw, "spr.preregistered_estimator", str)
    signal_stat = raw["spr"]["signal_statistic"]
    perturbation_stat = raw["spr"]["perturbation_statistic"]
    _ALLOWED_SIGNAL_STATS = ("peak_abs_projection", "mean_abs_projection", "rms_projection")
    if signal_stat not in _ALLOWED_SIGNAL_STATS:
        raise ConfigValidationError(
            f"spr.signal_statistic must be one of {_ALLOWED_SIGNAL_STATS}, got {signal_stat!r}"
        )
    if perturbation_stat != "mean_abs_directional_delta":
        raise ConfigValidationError(
            "spr.perturbation_statistic must be 'mean_abs_directional_delta', "
            f"got {perturbation_stat!r}"
        )
    expected_estimator = f"{signal_stat}/{perturbation_stat}"
    if estimator != expected_estimator:
        raise ConfigValidationError(
            f"spr.preregistered_estimator must equal {expected_estimator!r}, got {estimator!r}"
        )


def _validate_legacy_config(cfg: Config) -> None:
    """Minimal compatibility validation for historical callers."""
    _require(cfg.model, "model.name", str)
    _require(cfg.model, "readout.hook", str)
    _require(cfg.quant, "compression.bit_widths", list)
    _require(cfg.quant, "compression.quant_modules", dict)
    _require(cfg.quant, "results.root", str)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the standard config-file arguments to an argparse parser.

    Every pipeline entry point uses these so that a sweep change is a one-line
    config edit or one CLI flag, never a code edit.
    """
    parser.add_argument(
        "--model-config", type=Path, default=None, help="Path to model yaml (default: configs/model_gemma2_2b.yaml)"
    )
    parser.add_argument(
        "--quant-config", type=Path, default=None, help="Path to quant/sweep yaml (default: configs/quant_rtn_sweep.yaml)"
    )
    parser.add_argument("--model-name", type=str, default=None, help="Override model name (rare; config-driven otherwise)")
    parser.add_argument("--bits", type=int, nargs="*", default=None, help="Override bit-width sweep")
    parser.add_argument("--seed", type=int, default=None, help="Master seed override; per-step seeds derived and logged")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override results root")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config from parsed args, applying CLI overrides."""
    cfg = Config.from_yaml(args.model_config, args.quant_config)
    if args.model_name is not None:
        cfg.model["model"]["name"] = args.model_name
    if args.bits is not None:
        cfg.quant["compression"]["bit_widths"] = list(args.bits)
    if args.seed is not None:
        cfg.quant["seed"] = int(args.seed)
    if args.output_dir is not None:
        # Every artifact key holds a full path beginning with the old root, so
        # rewriting only `root` would leave all of them pointing at results/.
        results = cfg.quant["results"]
        old_root = results["root"]
        new_root = str(args.output_dir)
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
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def bit_widths(cfg: Config) -> list[int]:
    """The RTN bit-width sweep {8,7,6,5,4} from config."""
    return list(cfg.quant["compression"]["bit_widths"])


def results_path(cfg: Config, key: str) -> Path:
    """Resolve a results artifact path from config against REPO_ROOT."""
    return REPO_ROOT / cfg.quant["results"][key]
