"""Circuit-tracer integration boundary and quantized CIRCUS sweep.

No external tracer API is guessed here. Callers must inject a real model factory
and an adapter that returns the normalized :class:`AttributionGraph` schema.
"""

from __future__ import annotations

import csv
import gc
import json
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..config import Config, bit_widths
from ..sae.protocol import ExperimentMode
from .stability import (
    AttributionGraph,
    CircusConfig,
    classify_edges,
    consensus_circuit,
    graph_fingerprint,
    inclusion_frequencies,
    stability_configs,
    stability_shift,
)


@dataclass(frozen=True)
class TracerModelState:
    """One concrete model instance supplied by the caller."""

    label: str
    state_id: str
    model: Any


class TracerAdapter(Protocol):
    """Versioned caller-owned adapter around the real tracer API."""

    def __call__(
        self,
        *,
        model: Any,
        state_id: str,
        prompts: Sequence[str],
        device: str,
        seed: int,
    ) -> AttributionGraph: ...


ModelFactory = Callable[[str, int | None], TracerModelState]


def _safe_weakref(obj: Any) -> weakref.ref | None:
    """Weak reference to ``obj``, or ``None`` if it is not weak-referenceable.

    Real models are ``nn.Module`` instances (weak-referenceable), so the object
    identity guard stays fully active in production. A freed model's weakref
    goes dead and never matches, so releasing each condition's model between
    factory calls cannot produce a false "reused model" error from CPython
    recycling the freed object's address. Test stubs that are plain ``dict``s
    are not weak-referenceable; identity tracking is simply skipped for them
    (their distinctness is covered by the state_id and graph-fingerprint guards).
    """
    try:
        return weakref.ref(obj)
    except TypeError:
        return None


def _trim_noise_tail(graph: AttributionGraph, retain_cumulative: float) -> AttributionGraph:
    """Drop edges the CIRCUS ensemble can never include (lossless noise-tail prune).

    A raw circuit-tracer attribution graph is a near-dense ``[node, node]``
    adjacency, so its edge count is ``O(n_total^2)`` — tens of millions of edges,
    almost all in the vanishing-influence tail. CIRCUS includes an edge in a
    stability view only when its cumulative influence is ``<=`` that view's edge
    threshold, and ``stability_configs`` never uses an edge threshold above 0.95.
    So every edge with ``cumulative_influence`` above that can never be included
    in any view → ``s(e) = 0`` → NOISE by definition. Removing them changes no
    retained edge's inclusion frequency or class (cumulative influence is left
    untouched — no renormalization), it only discards the artifact tail.
    ``retain_cumulative`` sits just above the ensemble maximum (default 0.98 >
    0.95) so the prune stays provably lossless for core/contingent structure.
    """
    kept_edges = [edge for edge in graph.edges if edge.cumulative_influence <= retain_cumulative]
    if not kept_edges:
        raise RuntimeError(
            f"noise-tail trim at cumulative <= {retain_cumulative} removed every edge; "
            "the attribution graph has no influential head (check the tracer output)"
        )
    if len(kept_edges) == len(graph.edges):
        return graph
    return AttributionGraph(
        state_id=graph.state_id,
        nodes=graph.nodes,
        edges=kept_edges,
        graph_metadata=graph.graph_metadata,
    )


def _empty_cuda_cache() -> None:
    """Best-effort release of freed CUDA blocks back to the allocator.

    The circuit sweep reloads a full model per condition (fp16 + one per bit
    width). On an 8 GB GPU two copies of Gemma-2-2B do not fit, so each
    condition's model must be dropped *and* its cached blocks returned before
    the next ``model_factory`` call — otherwise peak memory spans two models
    at the reassignment point and OOMs (observed 2026-09-06). Import is guarded
    so the module stays usable on a CPU-only ``device``.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch always present in real runs
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class CircusResult:
    label: str
    state_id: str
    configs: Sequence[CircusConfig]
    core_edges: set[tuple[str, str]]
    inclusion: dict[tuple[str, str], float]
    graph_fingerprint: str

    @property
    def n_configs(self) -> int:
        return len(self.configs)


def build_attribution_graph(
    model_state: TracerModelState,
    prompts: Sequence[str],
    device: str,
    seed: int,
    *,
    tracer: TracerAdapter,
) -> AttributionGraph:
    """Invoke a caller-provided real tracer adapter and validate its normalized graph."""
    if not prompts:
        raise ValueError("circuit tracing requires at least one prompt")
    if not callable(tracer):
        raise TypeError("tracer must be a caller-provided real TracerAdapter")
    graph = tracer(
        model=model_state.model,
        state_id=model_state.state_id,
        prompts=prompts,
        device=device,
        seed=seed,
    )
    if not isinstance(graph, AttributionGraph):
        raise TypeError(
            "tracer adapter must return src.circuits.stability.AttributionGraph; "
            "normalize the real backend output in the caller-owned adapter"
        )
    if graph.state_id != model_state.state_id:
        raise ValueError(
            f"tracer returned graph state_id {graph.state_id!r} for model state "
            f"{model_state.state_id!r}; refusing mislabeled tracing"
        )
    backend = graph.graph_metadata.get("tracer_backend")
    version = graph.graph_metadata.get("tracer_version")
    if not backend or not version:
        raise ValueError(
            "normalized graph must record tracer_backend and tracer_version; "
            "placeholder or unverifiable graph output is refused"
        )
    if graph.graph_metadata.get("synthetic", False):
        raise ValueError("synthetic/fake graphs cannot be used in the circuit sweep")
    return graph


def run_circus(
    graph: AttributionGraph,
    n_configs: int,
    seed: int,
    *,
    label: str = "",
) -> CircusResult:
    """Run an explicit cumulative-influence stability ensemble on one graph."""
    configs = stability_configs(n_configs, seed)
    inclusion = inclusion_frequencies(graph, configs)
    return CircusResult(
        label=label,
        state_id=graph.state_id,
        configs=configs,
        core_edges=consensus_circuit(inclusion),
        inclusion=inclusion,
        graph_fingerprint=graph_fingerprint(graph),
    )


def _validate_model_state(
    expected_label: str,
    model_state: TracerModelState,
    state_ids: set[str],
    seen_models: list[weakref.ref],
) -> None:
    if not isinstance(model_state, TracerModelState):
        raise TypeError("model_factory must return TracerModelState")
    if model_state.label != expected_label:
        raise ValueError(
            f"model_factory returned label {model_state.label!r}, expected {expected_label!r}"
        )
    if not model_state.state_id:
        raise ValueError(f"model condition {expected_label!r} has an empty state_id")
    reused_object = any(ref() is model_state.model for ref in seen_models)
    if model_state.state_id in state_ids or reused_object:
        raise ValueError(
            f"model condition {expected_label!r} reuses another condition's model state; "
            "each bit width must be genuinely distinct"
        )
    state_ids.add(model_state.state_id)
    ref = _safe_weakref(model_state.model)
    if ref is not None:
        seen_models.append(ref)


def run_circus_sweep(
    cfg: Config,
    *,
    prompts: Sequence[str],
    seed: int,
    output_dir: Path,
    model_factory: ModelFactory,
    tracer: TracerAdapter,
    device: str = "cpu",
    mode: ExperimentMode | str = "harmonized",
) -> dict[str, CircusResult]:
    """Trace FP16 and every injected quantized model, rejecting reused/fake output.

    Artifacts are written under ``<output_dir>/<mode>/circuits/``: the experiment mode is part of the API and of every artifact path,
    never a cosmetic label.
    """
    normalized_mode = ExperimentMode(mode)
    n_configs = int(cfg.quant["circus"]["n_configs_min"])
    # Lossless prune (see _trim_noise_tail): keeps every edge the ensemble could
    # include, drops the O(n^2) vanishing-influence tail that would otherwise
    # bloat downstream tables to millions of rows. Above the max threshold (0.95).
    retain_cumulative = float(cfg.quant["circus"].get("edge_retain_cumulative", 0.98))
    contingent_min = float(cfg.quant["circus"]["contingent_min"])
    requested = [("fp16", None)] + [(f"int{bits}", bits) for bits in bit_widths(cfg)]
    state_ids: set[str] = set()
    seen_models: list[weakref.ref] = []
    fingerprints: dict[str, str] = {}
    results: dict[str, CircusResult] = {}

    for offset, (label, bits) in enumerate(requested):
        model_state = model_factory(label, bits)
        _validate_model_state(label, model_state, state_ids, seen_models)
        graph = build_attribution_graph(
            model_state,
            prompts,
            device,
            seed + offset,
            tracer=tracer,
        )
        graph = _trim_noise_tail(graph, retain_cumulative)
        fingerprint = graph_fingerprint(graph)
        if fingerprint in fingerprints:
            other = fingerprints[fingerprint]
            raise ValueError(
                f"conditions {other!r} and {label!r} produced identical attribution graphs; "
                "refusing reused/fake quantized tracing"
            )
        fingerprints[fingerprint] = label
        results[label] = run_circus(graph, n_configs, seed, label=label)

        # Free this condition's model before the next factory call so at most one
        # occupies the GPU (8 GB cannot hold two Gemma-2-2B copies; OOM 2026-09-06).
        # The reuse guard uses weakrefs, so a freed model drops out on its own and
        # cannot trigger a false "reused model" error on address recycling.
        model_state = None  # type: ignore[assignment]  # drop last GPU reference
        graph = None  # type: ignore[assignment]
        gc.collect()
        _empty_cuda_cache()

    artifact_root = output_dir / normalized_mode.value
    write_stability_configs_csv(results, artifact_root)
    write_edge_stability_csv(results, cfg, artifact_root)
    fp16 = results["fp16"]
    for label, result in results.items():
        if label != "fp16":
            edge_stability_shift_csv(
                fp16,
                result,
                artifact_root / "circuits" / f"stability_shift_{label}.csv",
            )
    _write_circus_metadata(artifact_root / "circuits" / "metadata.json", normalized_mode, seed, n_configs, results)
    if normalized_mode is ExperimentMode.HARMONIZED_R1:
        _write_circuit_spr_artifact(results, output_dir, contingent_min=contingent_min)
    return results


def _write_circuit_spr_artifact(
    results: Mapping[str, CircusResult], output_dir: Path, *, contingent_min: float
) -> Path:
    """Write real CIRCUS-stability rows as an explicitly labelled SPR approximation.

    SPR conditions on the instrument's FP16 readout structure, so only edges that
    are *non-noise in FP16* under CIRCUS — inclusion frequency ``>= contingent_min``
    (core ``s=1`` and contingent ``0.5<=s<1``) — are valid SPR samples. Edges below
    that are threshold-noise (Parekh 2603.00523), not structure the instrument
    reads; scoring them both corrupts the SPR fit and inflates the artifact to tens
    of millions of rows (the whole O(n^2) edge tail × every bit width).
    """
    fp16 = results.get("fp16")
    if fp16 is None:
        raise RuntimeError("circuit SPR requires an fp16 CIRCUS result")
    scored_edges = [edge for edge, stability in fp16.inclusion.items() if stability >= contingent_min]
    if not scored_edges:
        raise RuntimeError(
            f"circuit SPR artifact has no non-noise FP16 edges (inclusion >= {contingent_min}); "
            "cannot build the circuit SPR instrument"
        )
    rows: list[tuple[int, str, float, float, float]] = []
    for label, result in results.items():
        if label == "fp16":
            continue
        if not label.startswith("int"):
            raise ValueError(f"unexpected circuit condition label {label!r}")
        bits = int(label.removeprefix("int"))
        for edge in scored_edges:
            baseline = float(fp16.inclusion.get(edge, 0.0))
            quantized = float(result.inclusion.get(edge, 0.0))
            rows.append((bits, f"{edge[0]}->{edge[1]}", baseline, abs(quantized - baseline), 1.0 - quantized))
    if not rows:
        raise RuntimeError("circuit SPR artifact has no quantized edge rows")
    directory = output_dir / "harmonized" / "spr" / "instruments"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "circuits.npz"
    bits, edges, signal, perturbation, damage = map(list, zip(*rows, strict=True))
    import numpy as np

    # Perturbation floor = the instrument's resolution 1/B: inclusion frequency
    # s(e) is discretised to multiples of 1/B, so the smallest observable change
    # is 1/B. A 1e-8 floor turns every stable core edge (Δs=0) into an SPR≈1e8
    # outlier that wrecks the transfer metric (RMSE≈1e5); the resolution floor
    # caps a stable edge's SPR at signal*B. B is config-driven (no magic constant).
    resolution = 1.0 / float(fp16.n_configs)
    perturbation_arr = np.maximum(np.asarray(perturbation), resolution)
    np.savez_compressed(
        path, spr=np.asarray(signal) / perturbation_arr,
        target=np.asarray(damage), bit_width=np.asarray(bits, dtype=np.int64), groups=np.asarray(edges),
        signal=np.asarray(signal), perturbation=np.asarray(perturbation), target_kind=np.asarray("continuous"),
        direction_source=np.asarray("CIRCUS_stability_approximation"),
        perturbation_floor=np.asarray(resolution),
    )
    return path


def _write_circus_metadata(
    path: Path,
    mode: ExperimentMode,
    seed: int,
    n_configs: int,
    results: dict[str, CircusResult],
) -> None:
    """Persist mode/protocol metadata beside the circuit stability tables."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": mode.value,
        "instrument": "circuit_tracer_circus",
        "seed": seed,
        "n_ensemble_configs": n_configs,
        "conditions": {
            label: {"state_id": result.state_id, "graph_fingerprint": result.graph_fingerprint}
            for label, result in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def write_stability_configs_csv(
    results: dict[str, CircusResult], output_dir: Path
) -> Path:
    """Persist every explicit ensemble configuration used by every condition."""
    out = output_dir / "circuits"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "stability_configs.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "state_id",
                "config_id",
                "node_cumulative_influence",
                "edge_cumulative_influence",
                "tie_break_seed",
            ]
        )
        for label, result in results.items():
            for config in result.configs:
                writer.writerow(
                    [
                        label,
                        result.state_id,
                        config.config_id,
                        config.node_cumulative_influence,
                        config.edge_cumulative_influence,
                        config.tie_break_seed,
                    ]
                )
    return path


def write_edge_stability_csv(
    results: dict[str, CircusResult], cfg: Config, output_dir: Path
) -> Path:
    out = output_dir / "circuits"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "edge_stability.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "state_id", "source", "target", "inclusion", "class"])
        for label, result in results.items():
            classes = classify_edges(
                result.inclusion,
                cfg.quant["circus"]["core_min"],
                cfg.quant["circus"]["contingent_min"],
            )
            for edge_id, stability in sorted(result.inclusion.items()):
                writer.writerow(
                    [label, result.state_id, edge_id[0], edge_id[1], stability, classes[edge_id]]
                )
    return path


def edge_stability_shift_csv(fp16: CircusResult, quant: CircusResult, path: Path) -> None:
    shifts = stability_shift(fp16.inclusion, quant.inclusion)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target", "s_fp16", "s_quant", "shift"])
        for edge_id, shift in sorted(shifts.items()):
            writer.writerow(
                [
                    edge_id[0],
                    edge_id[1],
                    fp16.inclusion.get(edge_id, 0.0),
                    quant.inclusion.get(edge_id, 0.0),
                    shift,
                ]
            )
