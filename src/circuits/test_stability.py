"""Tests for cumulative-influence CIRCUS ensembles and tracing boundaries."""

from pathlib import Path

import numpy as np
import pytest

from src.circuits.extract import (
    TracerModelState,
    _trim_noise_tail,
    run_circus_sweep,
)
from src.circuits.stability import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
    CircusConfig,
    classify_edges,
    compare_stability,
    consensus_circuit,
    inclusion_frequencies,
    non_nested_configs,
    prune_graph,
    stability_configs,
)
from src.config import Config


def _graph(state_id: str = "fp16", damage: float = 0.0) -> AttributionGraph:
    nodes = [
        AttributionNode("n0", 0, "feature-0", 1.0 - damage, 1, 0.20, {"feature_id": 0}),
        AttributionNode("n1", 1, "feature-1", 0.8 - damage, 2, 0.45, {"feature_id": 1}),
        AttributionNode("n2", 2, "feature-2", 0.3, 3, 0.80, {"feature_id": 2}),
    ]
    edges = [
        AttributionEdge("n0", "n1", 1.0 - damage, 1, 0.30, 1, {"kind": "clt"}),
        AttributionEdge("n1", "n2", 0.5 - damage, 2, 0.65, 1, {"kind": "clt"}),
        AttributionEdge("n0", "n2", 0.2 - damage, 3, 0.92, -1, {"kind": "clt"}),
    ]
    return AttributionGraph(
        state_id,
        nodes,
        edges,
        {"tracer_backend": "real-adapter-test", "tracer_version": "1.2.3"},
    )


def _config(bits=(8, 4)) -> Config:
    return Config(
        model={"model": {"name": "toy"}},
        quant={
            "compression": {"bit_widths": list(bits)},
            "circus": {"n_configs_min": 9, "core_min": 1.0, "contingent_min": 0.5},
        },
        secondary_model={},
    )


def test_configs_are_explicit_crossed_and_reproducible():
    first = stability_configs(9, seed=17)
    second = stability_configs(9, seed=17)
    assert first == second
    assert all(isinstance(config, CircusConfig) for config in first)
    assert [config.node_cumulative_influence for config in first] == sorted(
        config.node_cumulative_influence for config in first
    )
    assert [config.edge_cumulative_influence for config in first] == sorted(
        (config.edge_cumulative_influence for config in first), reverse=True
    )
    assert non_nested_configs(_graph(), 9, 17) == first


def test_requires_at_least_9_configs():
    with pytest.raises(ValueError, match="B >= 9"):
        stability_configs(8, seed=0)


def test_pruning_uses_cumulative_influence_not_raw_score():
    graph = _graph()
    config = CircusConfig("view", 0.50, 0.70, 0)
    kept = prune_graph(graph, config)
    assert kept == {("n0", "n1")}


def test_stability_taxonomy_and_comparison():
    graph = _graph()
    inclusion = inclusion_frequencies(graph, stability_configs(9, seed=0))
    labels = classify_edges(inclusion)
    core = consensus_circuit(inclusion)
    assert core == {("n0", "n1")}
    assert labels[("n0", "n1")] == "core"
    assert labels[("n0", "n2")] == "noise"

    quantized = dict(inclusion)
    quantized[("n0", "n1")] = 0.5
    comparison = compare_stability(inclusion, quantized)
    assert comparison.delta_stability[("n0", "n1")] == pytest.approx(-0.5)
    assert comparison.core_retention == 0.0
    assert comparison.transitions[("n0", "n1")] == ("core", "contingent")
    assert 0.0 <= comparison.stability_adjusted_jaccard < 1.0


def test_quantized_sweep_traces_each_genuine_model_state(tmp_path: Path):
    traced_states = []

    def factory(label, bits):
        model = {"label": label, "bits": bits, "damage": 0.0 if bits is None else (9 - bits) / 100}
        return TracerModelState(label, f"state-{label}", model)

    def tracer(*, model, state_id, prompts, device, seed):
        del prompts, device, seed
        # Retain the model object so its identity stays valid for the
        # cross-sweep distinctness check below. The sweep frees each condition's
        # model between factory calls (GPU headroom), which would otherwise let
        # CPython recycle a freed object's address into the next condition and
        # collapse the id() comparison. Real tracers do not retain the model, so
        # the production freeing is unaffected.
        traced_states.append((state_id, model))
        return _graph(state_id, model["damage"])

    results = run_circus_sweep(
        _config(),
        prompts=["A real prompt"],
        seed=7,
        output_dir=tmp_path,
        model_factory=factory,
        tracer=tracer,
    )
    assert list(results) == ["fp16", "int8", "int4"]
    assert len({state_id for state_id, _ in traced_states}) == 3
    assert len({id(model) for _, model in traced_states}) == 3
    assert len({result.graph_fingerprint for result in results.values()}) == 3
    assert (tmp_path / "harmonized" / "circuits" / "stability_configs.csv").exists()
    assert (tmp_path / "harmonized" / "circuits" / "edge_stability.csv").exists()
    assert (tmp_path / "harmonized" / "circuits" / "metadata.json").exists()


def test_identical_quantized_graph_is_refused(tmp_path: Path):
    def factory(label, bits):
        return TracerModelState(label, f"state-{label}", {"bits": bits})

    def fake_tracer(*, model, state_id, prompts, device, seed):
        del model, prompts, device, seed
        graph = _graph("template", 0.0)
        return AttributionGraph(state_id, graph.nodes, graph.edges, graph.graph_metadata)

    with pytest.raises(ValueError, match="identical attribution graphs"):
        run_circus_sweep(
            _config(bits=(4,)),
            prompts=["prompt"],
            seed=0,
            output_dir=tmp_path,
            model_factory=factory,
            tracer=fake_tracer,
        )


def test_reused_model_instance_is_refused(tmp_path: Path):
    # A realistic model stub: real models are class instances (weak-referenceable),
    # never a bare ``object()`` (which cannot be weak-referenced). The sweep's
    # identity guard tracks models by weakref so a freed condition's model drops
    # out on its own; the closure below keeps this one alive across conditions,
    # so returning it twice is still refused.
    class _FakeModel:
        pass

    shared_model = _FakeModel()

    def factory(label, bits):
        del bits
        return TracerModelState(label, f"state-{label}", shared_model)

    def tracer(*, model, state_id, prompts, device, seed):
        del model, prompts, device, seed
        return _graph(state_id, 0.0 if state_id == "state-fp16" else 0.1)

    with pytest.raises(ValueError, match="reuses another condition's model state"):
        run_circus_sweep(
            _config(bits=(4,)),
            prompts=["prompt"],
            seed=0,
            output_dir=tmp_path,
            model_factory=factory,
            tracer=tracer,
        )


def test_trim_noise_tail_is_lossless_for_included_edges():
    # The tail edge (cumulative 0.995) can never be included by any ensemble view
    # (max edge threshold 0.95), so trimming it must not change any retained
    # edge's inclusion frequency.
    nodes = [
        AttributionNode("n0", 0, "feature-0", 1.0, 1, 0.10),
        AttributionNode("n1", 1, "feature-1", 0.8, 2, 0.40),
        AttributionNode("n2", 2, "feature-2", 0.05, 3, 0.70),
    ]
    edges = [
        AttributionEdge("n0", "n1", 1.0, 1, 0.30),
        AttributionEdge("n1", "n2", 0.4, 2, 0.60),
        AttributionEdge("n0", "n2", 0.001, 3, 0.995),  # vanishing-influence noise tail
    ]
    graph = AttributionGraph("fp16", nodes, edges, {"tracer_backend": "t", "tracer_version": "1"})
    configs = stability_configs(9, seed=0)
    full_inclusion = inclusion_frequencies(graph, configs)

    trimmed = _trim_noise_tail(graph, 0.98)
    assert {edge.edge_id for edge in trimmed.edges} == {("n0", "n1"), ("n1", "n2")}
    trimmed_inclusion = inclusion_frequencies(trimmed, configs)
    for edge_id, stability in trimmed_inclusion.items():
        assert full_inclusion[edge_id] == pytest.approx(stability)
    assert full_inclusion[("n0", "n2")] < 0.5  # the dropped edge was noise anyway


def test_circuit_spr_artifact_scores_only_non_noise_edges(tmp_path: Path):
    def factory(label, bits):
        return TracerModelState(label, f"state-{label}", {"bits": bits})

    def tracer(*, model, state_id, prompts, device, seed):
        del model, prompts, device, seed
        return _graph(state_id, 0.0 if state_id == "state-fp16" else 0.1)

    run_circus_sweep(
        _config(bits=(4,)),
        prompts=["p"],
        seed=0,
        output_dir=tmp_path,
        model_factory=factory,
        tracer=tracer,
    )
    with np.load(
        tmp_path / "harmonized" / "spr" / "instruments" / "circuits.npz", allow_pickle=False
    ) as data:
        groups = set(data["groups"].tolist())
        bits = set(data["bit_width"].tolist())
    # Only n0->n1 reaches s>=0.5 in _graph; n1->n2 (s<0.5) and n0->n2 (s=0) are
    # threshold-noise and must not appear as SPR samples.
    assert groups == {"n0->n1"}
    assert bits == {4}


def test_synthetic_graph_is_refused(tmp_path: Path):
    def factory(label, bits):
        return TracerModelState(label, f"state-{label}", {"bits": bits})

    def tracer(*, model, state_id, prompts, device, seed):
        del model, prompts, device, seed
        graph = _graph(state_id)
        metadata = dict(graph.graph_metadata, synthetic=True)
        return AttributionGraph(state_id, graph.nodes, graph.edges, metadata)

    with pytest.raises(ValueError, match="synthetic/fake"):
        run_circus_sweep(
            _config(bits=()),
            prompts=["prompt"],
            seed=0,
            output_dir=tmp_path,
            model_factory=factory,
            tracer=tracer,
        )
