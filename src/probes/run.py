"""Probe sweep driver with explicit reproduction and harmonized R1 modes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..compression.rtn import quantize_state_dict
from ..config import Config
from .gates import assert_probe_big_drop_gate, probe_gate_report
from .train_eval import (
    EXPERIMENT_MODES,
    FrozenProbe,
    ProbeCell,
    big_drop_rate,
    evaluate_frozen_probe,
    fit_frozen_probe,
    load_probe_dataset,
    operational_failure_rate,
    quantization_cells,
    write_cells_csv,
)

ProbeMode = Literal["reproduction", "harmonized"]


def _validate_mode(mode: str) -> ProbeMode:
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"probe mode must be one of {sorted(EXPERIMENT_MODES)}, got {mode!r}")
    return cast(ProbeMode, mode)


def _probe_config(cfg: Config, mode: ProbeMode) -> dict[str, Any]:
    """Resolve either a mode-nested probe config or the legacy flat config."""
    probe = cfg.model["probe"]
    mode_config = probe.get(mode)
    if isinstance(mode_config, dict):
        common = {key: value for key, value in probe.items() if key not in EXPERIMENT_MODES}
        return {**common, **mode_config}
    return probe


def _extract_activations(model: Any, texts: list[str], layer: int, seq_len: int) -> NDArray[np.generic]:
    """Extract residual-stream activations at the final non-padding token."""
    return _extract_activations_multi(model, texts, [layer], seq_len)[layer]


def _extract_activations_multi(
    model: Any, texts: list[str], layers: list[int], seq_len: int
) -> dict[int, NDArray[np.generic]]:
    """Extract all requested residual hooks in one model forward per example."""
    import torch

    if not layers or len(set(layers)) != len(layers):
        raise ValueError("requested probe layers must be a non-empty duplicate-free list")
    hooks = {layer: f"blocks.{layer}.hook_resid_post" for layer in layers}
    # Stop the forward right after the deepest layer we read. Probes never use
    # logits; running to the end would materialize the full [batch, seq, vocab]
    # tensor — Gemma-2's 256k vocab makes that ~8 GB for a 32x512 batch and OOMs
    # an 8 GB card. stop_at_layer is exclusive, so max(layers)+1 keeps the
    # deepest requested resid_post while skipping every later block + the unembed.
    stop_at_layer = max(layers) + 1
    rows: dict[int, list[NDArray[np.generic]]] = {layer: [] for layer in layers}
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True, truncate=True)[:, :seq_len]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=list(hooks.values()), stop_at_layer=stop_at_layer
            )
        for layer, hook in hooks.items():
            rows[layer].append(cache[hook][:, -1, :].detach().float().cpu().numpy())
    return {layer: np.concatenate(values, axis=0) for layer, values in rows.items()}


def _pad_token_id(model: Any) -> int:
    """Resolve a real pad token id; never silently reuse BOS (it would shift attention)."""
    tokenizer = model.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("model tokenizer exposes neither pad_token_id nor eos_token_id")
    return int(pad_id)


def _extract_activations_batched(
    model: Any, texts: list[str], layers: list[int], seq_len: int, batch_size: int,
    *, desc: str = "",
) -> dict[int, NDArray[np.generic]]:
    """Padded-batch extraction of all requested layers in one forward per batch.

    Each row is read at its own final non-padding token index, which under causal
    attention is exactly the activation single-example extraction would produce:
    trailing pad tokens are never attended by earlier positions. The
    equivalence test pins this against ``_extract_activations_multi``. ``desc`` is
    an optional label for the per-batch progress log (a long Gemma run is
    otherwise silent for hours).
    """
    import torch

    if batch_size <= 0:
        raise ValueError(f"probe batch_size must be positive, got {batch_size}")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("requested probe layers must be a non-empty duplicate-free list")
    hooks = {layer: f"blocks.{layer}.hook_resid_post" for layer in layers}
    # See _extract_activations_multi: stop_at_layer avoids the full-vocab logit
    # tensor (Gemma-2 vocab 256k), which is the dominant activation-extraction
    # allocation and OOMs an 8 GB card on a 32x512 batch. Exclusive index.
    stop_at_layer = max(layers) + 1
    pad_id = _pad_token_id(model)
    n_batches = (len(texts) + batch_size - 1) // batch_size
    rows: dict[int, list[NDArray[np.generic]]] = {layer: [] for layer in layers}
    for batch_i, start in enumerate(range(0, len(texts), batch_size)):
        if desc and (batch_i % 25 == 0 or batch_i == n_batches - 1):
            print(f"[PROBES]   {desc}: batch {batch_i + 1}/{n_batches} "
                  f"({100 * (batch_i + 1) / n_batches:.0f}%)", flush=True)
        batch_texts = texts[start : start + batch_size]
        token_rows = [
            model.to_tokens(text, prepend_bos=True, truncate=True)[:, :seq_len].reshape(-1)
            for text in batch_texts
        ]
        lengths = [int(row.numel()) for row in token_rows]
        width = max(lengths)
        batch = torch.full((len(token_rows), width), pad_id, dtype=torch.long, device=model.cfg.device)
        for index, row in enumerate(token_rows):
            batch[index, : lengths[index]] = row
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch, names_filter=list(hooks.values()), stop_at_layer=stop_at_layer
            )
        last_index = torch.as_tensor(lengths, device=batch.device) - 1
        for layer, hook in hooks.items():
            acts = cache[hook]  # [batch, width, d_model]
            selected = acts[torch.arange(len(token_rows), device=acts.device), last_index, :]
            rows[layer].append(selected.detach().float().cpu().numpy())
    return {layer: np.concatenate(values, axis=0) for layer, values in rows.items()}


def _apply_state_dict(model: Any, state_dict: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state_dict[name].to(param.dtype))


def _fit_monitor_probes(
    x_fp16: NDArray[np.generic],
    labels: NDArray[np.generic],
    *,
    n_seeds: int,
    base_auc_eligibility: float,
    test_size: float,
    ece_bins: int,
) -> list[FrozenProbe]:
    """Train exactly one FP16 probe for each seed."""
    return [
        fit_frozen_probe(
            x_fp16,
            labels,
            seed,
            base_auc_eligibility,
            test_size,
            ece_bins,
        )
        for seed in range(n_seeds)
    ]


def _evaluate_condition(
    frozen_probes: list[FrozenProbe],
    activations: NDArray[np.generic],
    *,
    bits: int,
    monitor: str,
    layer: int,
    mode: ProbeMode,
) -> list[ProbeCell]:
    """Evaluate already-fitted probes without any retraining."""
    return [
        evaluate_frozen_probe(
            frozen,
            activations,
            bits=bits,
            monitor=monitor,
            layer=layer,
            mode=mode,
        )
        for frozen in frozen_probes
    ]


def _write_metadata(
    path: Path,
    cfg: Config,
    probe_cfg: dict[str, Any],
    mode: ProbeMode,
    cells: list[ProbeCell],
) -> None:
    """Persist mode and protocol metadata beside the result table."""
    model_cfg = cfg.model["model"]
    compression = cfg.quant["compression"]
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": mode,
        "check_semantics": (
            "exact_reproduction" if mode == "reproduction" and probe_cfg.get("source_protocol_exact", True)
            else "source_reference_validation" if mode == "reproduction" else "directional_harmonized_r1"
        ),
        "instrument": "linear_probe",
        "model_identifier": model_cfg["name"],
        "model_revision": model_cfg.get("revision"),
        "tokenizer_identifier": model_cfg.get("tokenizer", model_cfg["name"]),
        "layers": probe_cfg["layers"],
        "datasets": probe_cfg["datasets"],
        "dataset_splits": {
            monitor: spec.get("split") if isinstance(spec, dict) else None
            for monitor, spec in probe_cfg["datasets"].items()
        },
        "seeds": list(range(probe_cfg["n_seeds"])),
        "sequence_length": cfg.model["perplexity"]["seq_len"],
        "number_of_cells": len(cells),
        "compression_method": compression.get("method", "RTN"),
        "bit_widths": compression["bit_widths"],
        "quantized_modules": compression["quant_modules"],
        "excluded_modules": compression["skip_modules"],
        "probe_training_precision": "FP16",
        "probe_retrained_per_condition": False,
        "metrics": ["auc", "brier", "log_loss", "ece"],
        "operational_failure_criterion": {
            "metric": "post_update_auc",
            "operator": "<",
            "threshold": probe_cfg.get("operational_failure_auc", 0.70),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def _write_probe_direction(
    directory: Path, *, monitor: str, layer: int, frozen: FrozenProbe
) -> Path:
    """Persist the exact FP16-trained readout direction used at every bit width."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{monitor}_layer{layer}_seed{frozen.seed}.npz"
    np.savez_compressed(
        path,
        w_probe=np.asarray(frozen.estimator.coef_, dtype=np.float64).reshape(-1),
        bias=np.asarray(frozen.estimator.intercept_, dtype=np.float64),
        train_test_split="stratified_fp16",
        test_indices=np.asarray(frozen.test_indices, dtype=np.int64),
        layer=np.asarray(layer),
        monitor=np.asarray(monitor),
        seed=np.asarray(frozen.seed),
        preprocessing=np.asarray("none"),
    )
    return path


def _append_probe_spr_rows(
    rows: list[dict[str, object]], *, frozen: FrozenProbe, x_fp16: NDArray[np.generic],
    x_quant: NDArray[np.generic], cell: ProbeCell, damage_scale: float,
) -> None:
    """Project real aligned activations on the persisted frozen-probe direction."""
    direction = np.asarray(frozen.estimator.coef_, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"probe {cell.monitor}/layer{cell.layer}/seed{cell.seed} has a zero/non-finite direction")
    direction /= norm
    indices = frozen.test_indices
    fp = np.asarray(x_fp16[indices], dtype=np.float64)
    quant = np.asarray(x_quant[indices], dtype=np.float64)
    signal = float(np.abs(fp @ direction).mean())
    perturbation = float(np.abs((quant - fp) @ direction).mean())
    rows.append({
        "instrument": "probe", "sample_id": f"{cell.monitor}:L{cell.layer}:S{cell.seed}:B{cell.bits}",
        "readout_id": f"{cell.monitor}:L{cell.layer}:S{cell.seed}", "condition": "fp16" if cell.bits == 16 else f"int{cell.bits}",
        "bit_width": cell.bits, "group_id": f"{cell.monitor}:L{cell.layer}:S{cell.seed}",
        "signal": signal, "perturbation": perturbation, "spr": signal / max(perturbation, 1e-8),
        "raw_target": cell.delta_auc, "normalized_damage": float(np.clip(-cell.delta_auc / damage_scale, 0.0, 1.0)),
        "target_kind": "continuous", "model_id": "configured", "layer": cell.layer,
        "direction_source": "fp16_logistic_regression", "seed": cell.seed,
    })


def _write_probe_spr_artifact(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write the transfer-compatible projection of the shared SPR row contract."""
    if not rows:
        raise RuntimeError("probe SPR artifact cannot be written without evaluated frozen probes")
    directory = output_dir / "harmonized" / "spr" / "instruments"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "probe.npz"
    np.savez_compressed(
        path,
        spr=np.asarray([row["spr"] for row in rows], dtype=np.float64),
        target=np.asarray([row["normalized_damage"] for row in rows], dtype=np.float64),
        bit_width=np.asarray([row["bit_width"] for row in rows], dtype=np.int64),
        groups=np.asarray([row["group_id"] for row in rows]),
        signal=np.asarray([row["signal"] for row in rows], dtype=np.float64),
        perturbation=np.asarray([row["perturbation"] for row in rows], dtype=np.float64),
        target_kind=np.asarray("continuous"),
    )
    with (directory / "probe_rows.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    return path


def _stratified_budget_selection(
    texts: list[str], labels: NDArray[np.generic], budget: int, *, seed: int
) -> list[int]:
    """Deterministically select at most ``budget`` example indices, class-balanced.

    A naive ``texts[:budget]`` prefix can leave a smoke budget with a single
    class, which would crash the frozen-probe contract downstream. Selection is
    seeded and class-stratified, and the caller persists the chosen hashes so the
    exact smoke subset is auditable.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    per_class = budget // 2
    chosen: list[int] = []
    counts = {0: 0, 1: 0}
    for index in order:
        label = int(labels[index])
        if counts[label] < per_class:
            chosen.append(int(index))
            counts[label] += 1
        if len(chosen) >= budget:
            break
    if len(chosen) < budget:  # one class exhausted: fill from the other class
        # Maintain the membership set ONCE; rebuilding ``set(chosen)`` inside the
        # loop is O(n * budget) and can reach tens of billions of ops on a large
        # imbalanced monitor (e.g. 400k texts, budget 100k).
        chosen_set = set(chosen)
        for index in order:
            idx = int(index)
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) >= budget:
                break
    chosen.sort()
    if len(chosen) < 2:
        raise ValueError(f"budget selection produced {len(chosen)} examples; need at least 2")
    return chosen


def _persist_budget_selection(
    artifact_dir: Path, monitor: str, texts: list[str], indices: list[int]
) -> None:
    """Persist the exact selected example hashes for a budgeted probe run."""
    import hashlib

    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "monitor": monitor,
        "n_selected": len(indices),
        "selected_indices": indices,
        "selected_text_sha256": [
            hashlib.sha256(" ".join(texts[i].split()).encode("utf-8")).hexdigest()
            for i in indices
        ],
    }
    with (artifact_dir / f"budget_selection_{monitor}.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def run_probe_sweep(
    cfg: Config,
    *,
    mode: ProbeMode = "harmonized",
    texts: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[ProbeCell]:
    """Run a frozen-probe sweep and write mode-separated results.

    ``mode`` is part of the API and every row/metadata artifact. The default is
    ``harmonized`` because the legacy implementation used the shared RTN sweep;
    callers claiming reproduction must opt into ``mode="reproduction"``.
    ``output_dir`` is the common results root, beneath which this writes
    ``<mode>/probes/``.
    """
    import torch

    transformer_lens = import_module("transformer_lens")
    mode = _validate_mode(mode)
    model_cfg = cfg.model["model"]
    probe_cfg = _probe_config(cfg, mode)
    compression = cfg.quant["compression"]
    bit_widths = compression["bit_widths"]
    if output_dir is None:
        output_dir = _results_root(cfg)
    artifact_dir = output_dir / mode / "probes"

    model = transformer_lens.HookedTransformer.from_pretrained(
        model_cfg["name"],
        device=model_cfg["device"],
        dtype=torch.float16,
        revision=model_cfg.get("revision"),
        trust_remote_code=False,
    )
    # ``.detach().cpu().clone()`` (not ``.clone().to("cpu")``): move to CPU first,
    # so we never allocate a full-parameter clone on the GPU. Cloning on-device
    # first would spike GPU memory by the largest tensor (Gemma-2 embed is
    # ~1.18 GB) on top of the resident model — needless pressure on an 8 GB card.
    # ``.cpu()`` already copies off-device; the trailing ``.clone()`` guarantees a
    # standalone tensor even on a CPU-only box where ``.cpu()`` is a no-op.
    fp16_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }

    monitor_data: dict[str, tuple[list[str], NDArray[np.generic]]] = {}
    budget = int((cfg.experiment.get("budgets") or {}).get("smoke" if cfg.mode == "smoke" else "full", {}).get("max_examples", 0))
    for monitor in probe_cfg["monitors"]:
        monitor_texts, labels = load_probe_dataset(monitor, probe_cfg["datasets"])
        if texts is not None:
            if len(texts) != len(labels):
                raise ValueError(
                    f"monitor {monitor!r}: override has {len(texts)} texts but dataset has "
                    f"{len(labels)} labels"
                )
            monitor_texts = texts
        labels = np.asarray(labels)
        if budget and len(monitor_texts) > budget:
            indices = _stratified_budget_selection(monitor_texts, labels, budget, seed=int(cfg.experiment.get("seeds", [0])[0]))
            _persist_budget_selection(artifact_dir, monitor, monitor_texts, indices)
            monitor_texts = [monitor_texts[i] for i in indices]
            labels = labels[indices]
        if len(monitor_texts) != len(labels) or len(monitor_texts) < 2:
            raise ValueError(f"monitor {monitor!r}: smoke/full budget left too few aligned examples")
        monitor_data[monitor] = (monitor_texts, labels)

    cells: list[ProbeCell] = []
    spr_rows: list[dict[str, object]] = []
    ece_bins = probe_cfg.get("ece_bins", 10)
    seq_len = cfg.model["perplexity"]["seq_len"]
    batch_size = int(probe_cfg.get("batch_size", 16))
    for monitor, (monitor_texts, labels) in monitor_data.items():
        print(f"[PROBES] monitor={monitor}: {len(monitor_texts)} examples x "
              f"{1 + len(bit_widths)} conditions (fp16 + {bit_widths}), batch={batch_size}", flush=True)
        _apply_state_dict(model, fp16_state)
        fp16_by_layer = _extract_activations_batched(model, monitor_texts, list(probe_cfg["layers"]), seq_len, batch_size, desc=f"{monitor} fp16")
        frozen_by_layer: dict[int, list[FrozenProbe]] = {}
        for layer in probe_cfg["layers"]:
            x_fp16 = fp16_by_layer[layer]
            frozen_probes = _fit_monitor_probes(
                x_fp16,
                labels,
                n_seeds=probe_cfg["n_seeds"],
                base_auc_eligibility=probe_cfg["base_auc_eligibility"],
                test_size=probe_cfg["test_size"],
                ece_bins=ece_bins,
            )
            frozen_by_layer[layer] = frozen_probes
            direction_dir = artifact_dir / "readout_directions"
            for frozen in frozen_probes:
                _write_probe_direction(direction_dir, monitor=monitor, layer=layer, frozen=frozen)
            fp_cells = _evaluate_condition(
                    frozen_probes,
                    x_fp16,
                    bits=16,
                    monitor=monitor,
                    layer=layer,
                    mode=mode,
                )
            cells.extend(fp_cells)
            # FP16 rows carry zero quantization perturbation by construction, so
            # their SPR is undefined (division by eps); the SPR artifact contains
            # quantized conditions only, like the SAE/circuits/patching arms.
        for bits in bit_widths:
            quantized_state = quantize_state_dict(
                fp16_state, bits, compression["quant_modules"], compression["skip_modules"],
            )
            _apply_state_dict(model, quantized_state)
            quantized_by_layer = _extract_activations_batched(model, monitor_texts, list(probe_cfg["layers"]), seq_len, batch_size, desc=f"{monitor} int{bits}")
            for layer in probe_cfg["layers"]:
                x_fp16 = fp16_by_layer[layer]
                x_quantized = quantized_by_layer[layer]
                frozen_probes = frozen_by_layer[layer]
                condition_cells = _evaluate_condition(
                        frozen_probes,
                        x_quantized,
                        bits=bits,
                        monitor=monitor,
                        layer=layer,
                        mode=mode,
                    )
                cells.extend(condition_cells)
                for frozen, cell in zip(frozen_probes, condition_cells, strict=True):
                    _append_probe_spr_rows(spr_rows, frozen=frozen, x_fp16=x_fp16, x_quant=x_quantized, cell=cell, damage_scale=float(probe_cfg.get("spr_damage_scale", 0.05)))
        _apply_state_dict(model, fp16_state)

    write_cells_csv(cells, artifact_dir / "auc_calibration.csv")
    _write_metadata(artifact_dir / "metadata.json", cfg, probe_cfg, mode, cells)
    if mode == "harmonized":
        _write_probe_spr_artifact(spr_rows, output_dir)
    return cells


def _results_root(cfg: Config) -> Path:
    from ..config import REPO_ROOT

    return REPO_ROOT / cfg.quant["results"]["root"]


def report_rates(cells: list[ProbeCell], cfg: Config, *, mode: ProbeMode = "harmonized") -> dict[str, float]:
    """Report rates over eligible quantized cells using absolute operational failure."""
    probe_cfg = _probe_config(cfg, _validate_mode(mode))
    quantized = quantization_cells(cells)
    eligible = [cell for cell in quantized if cell.eligible]
    return {
        "big_drop_rate": big_drop_rate(quantized, probe_cfg["big_drop_auc_threshold"]),
        "operational_failure_rate": operational_failure_rate(
            quantized, probe_cfg.get("operational_failure_auc", 0.70)
        ),
        "n_cells": float(len(quantized)),
        "n_eligible": float(len(eligible)),
        "eligible_coverage": len(eligible) / len(quantized) if quantized else 0.0,
    }


def run_probe_gate(
    cells: list[ProbeCell],
    cfg: Config,
    *,
    mode: ProbeMode = "harmonized",
) -> dict[str, Any]:
    """Enforce an exact reproduction gate or labelled harmonized directional gate."""
    mode = _validate_mode(mode)
    probe_cfg = _probe_config(cfg, mode)
    gate_cfg = cfg.quant["gate_probe"]
    kwargs = {
        "big_drop_auc_threshold": probe_cfg["big_drop_auc_threshold"],
        "operational_failure_auc": probe_cfg.get("operational_failure_auc", 0.70),
        "min_eligible_count": gate_cfg.get("min_eligible_count", 1),
        "min_eligible_coverage": gate_cfg.get("min_eligible_coverage", 0.5),
        "max_operational_failures": gate_cfg.get("max_operational_failures", 0),
        "require_reference_interval": mode == "reproduction" and bool(probe_cfg.get("source_protocol_exact", True)),
    }
    assert_probe_big_drop_gate(
        cells,
        gate_cfg["reference_big_drop_rate_min"],
        gate_cfg["reference_big_drop_rate_max"],
        **kwargs,
    )
    return probe_gate_report(
        cells,
        gate_cfg["reference_big_drop_rate_min"],
        gate_cfg["reference_big_drop_rate_max"],
        **kwargs,
    )
