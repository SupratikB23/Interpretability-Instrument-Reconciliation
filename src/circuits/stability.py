"""Faithful cumulative-influence stability ensembles for attribution graphs.

This module owns only a normalized graph schema and CIRCUS stability logic. It
contains no guessed circuit-tracer API and never converts raw score quantiles into
purported CIRCUS configurations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

CORE = "core"
CONTINGENT = "contingent"
NOISE = "noise"


@dataclass(frozen=True)
class AttributionNode:
    node_id: str
    layer: int | None
    component: str
    influence: float
    rank: int
    cumulative_influence: float
    feature_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttributionEdge:
    source: str
    target: str
    influence: float
    rank: int
    cumulative_influence: float
    sign: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def edge_id(self) -> tuple[str, str]:
        return (self.source, self.target)


@dataclass(frozen=True)
class AttributionGraph:
    """Normalized tracer output tied to one concrete model state."""

    state_id: str
    nodes: Sequence[AttributionNode]
    edges: Sequence[AttributionEdge]
    graph_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("attribution graph requires the traced model state_id")
        if not self.nodes or not self.edges:
            raise ValueError("attribution graph must contain nodes and edges; fake/empty graphs are refused")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("attribution graph contains duplicate node IDs")
        known = set(node_ids)
        edge_ids: set[tuple[str, str]] = set()
        for node in self.nodes:
            _validate_ranked_influence("node", node.node_id, node.rank, node.cumulative_influence)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge {edge.edge_id!r} references an unknown node")
            if edge.edge_id in edge_ids:
                raise ValueError(f"duplicate edge {edge.edge_id!r} is ambiguous")
            edge_ids.add(edge.edge_id)
            _validate_ranked_influence("edge", repr(edge.edge_id), edge.rank, edge.cumulative_influence)
            if edge.sign not in (-1, 0, 1, None):
                raise ValueError(f"edge {edge.edge_id!r} has invalid sign {edge.sign!r}")

    @property
    def edge_ids(self) -> list[tuple[str, str]]:
        return [edge.edge_id for edge in self.edges]

    @property
    def node_scores(self) -> dict[str, float]:
        return {node.node_id: node.influence for node in self.nodes}


def _validate_ranked_influence(kind: str, identifier: str, rank: int, cumulative: float) -> None:
    if rank < 1:
        raise ValueError(f"{kind} {identifier!r} must have a positive influence rank")
    if not 0.0 <= cumulative <= 1.0:
        raise ValueError(
            f"{kind} {identifier!r} cumulative influence must be in [0, 1], got {cumulative}"
        )


@dataclass(frozen=True)
class CircusConfig:
    """One explicit cumulative-influence pruning view."""

    config_id: str
    node_cumulative_influence: float
    edge_cumulative_influence: float
    tie_break_seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("node_cumulative_influence", self.node_cumulative_influence),
            ("edge_cumulative_influence", self.edge_cumulative_influence),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")


def stability_configs(n_configs: int, seed: int) -> list[CircusConfig]:
    """Build B explicit crossed cumulative-influence configurations.

    Node influence retained increases while edge influence retained decreases, so
    no view is a strict threshold refinement of all preceding views. The values
    are cumulative-influence targets, not quantiles of raw graph scores.
    """
    if n_configs < 9:
        raise ValueError(f"CIRCUS requires B >= 9 non-nested configs, got {n_configs}")
    if n_configs == 1:  # unreachable, documents the denominator below
        levels = [0.75]
    else:
        levels = [0.55 + (0.40 * index / (n_configs - 1)) for index in range(n_configs)]
    return [
        CircusConfig(
            config_id=f"ci-{index + 1:02d}",
            node_cumulative_influence=node_level,
            edge_cumulative_influence=edge_level,
            tie_break_seed=seed + index,
        )
        for index, (node_level, edge_level) in enumerate(zip(levels, reversed(levels), strict=True))
    ]


def non_nested_configs(
    graph: AttributionGraph, n_configs: int, seed: int
) -> list[CircusConfig]:
    """Compatibility name for explicit cumulative-influence configs."""
    del graph
    return stability_configs(n_configs, seed)


def prune_graph(graph: AttributionGraph, config: CircusConfig) -> set[tuple[str, str]]:
    """Prune using tracer-supplied cumulative node and edge influence metadata."""
    kept_nodes = {
        node.node_id
        for node in graph.nodes
        if node.cumulative_influence <= config.node_cumulative_influence
    }
    return {
        edge.edge_id
        for edge in graph.edges
        if edge.cumulative_influence <= config.edge_cumulative_influence
        and edge.source in kept_nodes
        and edge.target in kept_nodes
    }


def inclusion_frequencies(
    graph: AttributionGraph, configs: Iterable[CircusConfig]
) -> dict[tuple[str, str], float]:
    """Compute s(e), the edge inclusion frequency across explicit views."""
    counts = {edge_id: 0 for edge_id in graph.edge_ids}
    n_views = 0
    for config in configs:
        if not isinstance(config, CircusConfig):
            raise TypeError("CIRCUS views must be explicit CircusConfig objects")
        for edge_id in prune_graph(graph, config):
            counts[edge_id] += 1
        n_views += 1
    if n_views < 9:
        raise ValueError(f"CIRCUS stability requires at least 9 views, got {n_views}")
    return {edge_id: count / n_views for edge_id, count in counts.items()}


def consensus_circuit(
    inclusion: Mapping[tuple[str, str], float], consensus_threshold: float = 1.0
) -> set[tuple[str, str]]:
    return {edge_id for edge_id, stability in inclusion.items() if stability >= consensus_threshold}


def classify_edges(
    inclusion: Mapping[tuple[str, str], float],
    core_min: float = 1.0,
    contingent_min: float = 0.5,
) -> dict[tuple[str, str], str]:
    if core_min != 1.0 or contingent_min != 0.5:
        raise ValueError("faithful CIRCUS taxonomy is fixed at core=1 and contingent>=0.5")
    return {
        edge_id: CORE if stability == 1.0 else CONTINGENT if stability >= 0.5 else NOISE
        for edge_id, stability in inclusion.items()
    }


def stability_shift(
    inclusion_fp16: Mapping[tuple[str, str], float],
    inclusion_quant: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    all_edges = set(inclusion_fp16) | set(inclusion_quant)
    return {
        edge_id: inclusion_quant.get(edge_id, 0.0) - inclusion_fp16.get(edge_id, 0.0)
        for edge_id in all_edges
    }


@dataclass(frozen=True)
class StabilityComparison:
    delta_stability: Mapping[tuple[str, str], float]
    core_retention: float
    transitions: Mapping[tuple[str, str], tuple[str, str]]
    weighted_overlap: float
    stability_adjusted_jaccard: float


def compare_stability(
    fp16: Mapping[tuple[str, str], float], quantized: Mapping[tuple[str, str], float]
) -> StabilityComparison:
    """Return preregisterable stability comparisons without binary-only collapse."""
    all_edges = set(fp16) | set(quantized)
    fp_classes = classify_edges(fp16)
    quant_classes = classify_edges(quantized)
    fp_core = {edge for edge, value in fp16.items() if value == 1.0}
    retained = sum(quantized.get(edge, 0.0) == 1.0 for edge in fp_core)
    minima = sum(min(fp16.get(edge, 0.0), quantized.get(edge, 0.0)) for edge in all_edges)
    maxima = sum(max(fp16.get(edge, 0.0), quantized.get(edge, 0.0)) for edge in all_edges)
    fp_total = sum(fp16.values())
    quant_total = sum(quantized.values())
    weighted_overlap = 0.0 if max(fp_total, quant_total) == 0 else minima / max(fp_total, quant_total)
    return StabilityComparison(
        delta_stability=stability_shift(fp16, quantized),
        core_retention=1.0 if not fp_core else retained / len(fp_core),
        transitions={
            edge: (fp_classes.get(edge, NOISE), quant_classes.get(edge, NOISE))
            for edge in all_edges
        },
        weighted_overlap=weighted_overlap,
        stability_adjusted_jaccard=1.0 if maxima == 0 else minima / maxima,
    )


def graph_fingerprint(graph: AttributionGraph) -> str:
    """Fingerprint graph content while deliberately excluding condition labels/state IDs."""
    payload = {
        "nodes": [
            (node.node_id, node.layer, node.component, node.influence, node.rank, node.cumulative_influence)
            for node in sorted(graph.nodes, key=lambda item: item.node_id)
        ],
        "edges": [
            (edge.source, edge.target, edge.influence, edge.rank, edge.cumulative_influence, edge.sign)
            for edge in sorted(graph.edges, key=lambda item: item.edge_id)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
