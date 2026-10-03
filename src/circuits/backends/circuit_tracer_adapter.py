"""Adapter layer around Anthropic's ``circuit-tracer`` backend.

This module isolates the external tracer behind a versioned, normalized-graph
boundary. It implements the :class:`~src.circuits.extract.TracerAdapter`
protocol: it accepts a live :class:`circuit_tracer.ReplacementModel` (already a
``TransformerLensReplacementModel``), runs real attribution on it, and returns a
normalized :class:`~src.circuits.stability.AttributionGraph`.

No tracer API is guessed here; the real package is imported lazily inside each
function so this module stays importable when ``circuit-tracer`` is not
installed (in which case the import raises a clear message rather than a
``ModuleNotFoundError``).

Scientific mapping (circuit-tracer ``Graph`` -> ``AttributionGraph``):

- The tracer's ``adjacency_matrix`` is ``[target, source]`` over nodes ordered as
  ``[feature nodes, error nodes, token nodes, logit nodes]`` (see
  ``circuit_tracer.graph.Graph`` docstring). Feature node ``k`` corresponds to
  ``graph.active_features[k] == (layer, position, feature_idx)``.
- ``compute_node_influence`` / ``compute_edge_influence`` are the tracer's own
  cumulative-influence functions, which is exactly the cumulative-influence
  (not raw-score-quantile) signal that CIRCUS requires.
- Node and edge ``rank`` are assigned by sorting influence descending (rank 1 =
  most influential); ``cumulative_influence`` is the normalized running-sum of
  influence in that ordering, so it lies in ``[0, 1]`` as ``stability.py``
  expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Imported lazily to keep the module importable without circuit-tracer installed.
_ADAPTER_VERSION = "circuit-tracer-0.5.x"


def _import_circuit_tracer():
    """Import the real backend or raise an actionable error."""
    try:
        import circuit_tracer  # noqa: F401
        from circuit_tracer import attribute
        from circuit_tracer.graph import (
            compute_edge_influence,
            compute_node_influence,
            normalize_matrix,
        )
    except ImportError as exc:  # pragma: no cover - depends on install env
        raise ImportError(
            "circuit-tracer is not installed. It is not on PyPI; download it from GitHub: "
            "`pip install git+https://github.com/safety-research/circuit-tracer.git`."
        ) from exc
    return attribute, compute_node_influence, compute_edge_influence, normalize_matrix


def _influence_rank_and_cumulative(values: Sequence[float]) -> tuple[list[int], list[float]]:
    """Return descending ranks and normalized cumulative influence for a value list.

    ``rank`` is 1-based with 1 = largest value. ``cumulative_influence[i]`` is the
    running sum of ``value[i]`` over the descending ordering, normalized to
    ``[0, 1]`` by the total. Ties are broken by original position for
    determinism.
    """
    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    total = sum(abs(values[i]) for i in order)
    if total <= 0.0:
        # All-zero influence: still produce a valid monotone layout.
        n = len(values)
        return [i + 1 for i in range(n)], [1.0] * n
    ranks = [0] * len(values)
    cumulative = [0.0] * len(values)
    running = 0.0
    for position, index in enumerate(order, start=1):
        ranks[index] = position
        running += abs(values[index])
        cumulative[index] = running / total
    return ranks, cumulative


def _feature_node_id(layer: int, pos: int, feature_idx: int) -> str:
    """Unique id for a circuit-tracer feature node.

    circuit-tracer treats each ``(layer, position, feature_idx)`` triple as a
    distinct node, so the token POSITION must be part of the id — the same
    feature firing at two positions is two nodes. Omitting position collapsed
    them and raised "attribution graph contains duplicate node IDs". Used for
    both node creation and the adjacency-index -> id map so edges still resolve.
    """
    return f"L{layer}:P{pos}:feature-{layer}-{feature_idx}"


def _build_graph(graph, state_id: str) -> Any:
    """Convert a circuit-tracer ``Graph`` to a normalized ``AttributionGraph``."""
    from circuit_tracer.graph import (
        compute_edge_influence as _edge_fluence,
    )
    from circuit_tracer.graph import (
        compute_node_influence as _node_fluence,
    )
    from circuit_tracer.graph import (
        normalize_matrix,
    )

    from ..stability import AttributionEdge, AttributionGraph, AttributionNode

    adjacency = graph.adjacency_matrix
    n_logits = len(graph.logit_targets)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    n_total = adjacency.shape[0]

    # Logit node weights, matching circuit_tracer.graph.prune_graph semantics.
    logit_weights = graph.adjacency_matrix.new_zeros(n_total)
    logit_weights[-n_logits:] = graph.logit_probabilities

    node_influence = _node_fluence(adjacency, logit_weights)  # length n_total
    node_influence = node_influence.detach().float().cpu().numpy()

    nodes = []
    for index, (layer, _pos, feature_idx) in enumerate(graph.active_features.tolist()):
        layer_i, pos_i, fidx_i = int(layer), int(_pos), int(feature_idx)
        component = f"feature-{layer_i}-{fidx_i}"
        nodes.append(
            AttributionNode(
                node_id=_feature_node_id(layer_i, pos_i, fidx_i),
                layer=layer_i,
                component=component,
                influence=float(node_influence[index]),
                rank=0,  # filled below
                cumulative_influence=0.0,  # filled below
                feature_metadata={"position": pos_i, "feature_idx": fidx_i},
            )
        )
    # Token and logit nodes (kept for edge endpoints), plus error nodes.
    # token nodes
    for pos in range(n_tokens):
        nodes.append(
            AttributionNode(
                node_id=f"token:{pos}", layer=None, component=f"token-{pos}",
                influence=float(node_influence[n_features + n_tokens * graph.cfg.n_layers + pos]),
                rank=0, cumulative_influence=0.0,
            )
        )
    # logit nodes
    for idx, _target in enumerate(graph.logit_targets):
        nodes.append(
            AttributionNode(
                node_id=f"logit:{idx}", layer=None, component=f"logit-{idx}",
                influence=float(node_influence[-n_logits + idx]),
                rank=0, cumulative_influence=0.0,
            )
        )

    node_of_index = {}
    for k in range(n_features):
        layer = int(graph.active_features[k][0])
        pos = int(graph.active_features[k][1])
        fidx = int(graph.active_features[k][2])
        node_of_index[k] = _feature_node_id(layer, pos, fidx)
    for pos in range(n_tokens):
        node_of_index[n_features + n_tokens * graph.cfg.n_layers + pos] = f"token:{pos}"
    for idx in range(n_logits):
        node_of_index[-n_logits + idx] = f"logit:{idx}"

    # Assign node ranks/cumulative influence from the influence list.
    node_ranks, node_cum = _influence_rank_and_cumulative([n.influence for n in nodes])
    nodes = [
        AttributionNode(
            node_id=n.node_id, layer=n.layer, component=n.component, influence=n.influence,
            rank=node_ranks[i], cumulative_influence=node_cum[i],
            feature_metadata=n.feature_metadata,
        )
        for i, n in enumerate(nodes)
    ]

    # Edge influence over the normalized adjacency (rows=target, cols=source).
    pruned = adjacency.clone()
    edge_scores = _edge_fluence(normalize_matrix(pruned), logit_weights)  # [target, source]
    edge_scores = edge_scores.detach().float().cpu().numpy()

    edges = []
    edge_records = []  # (source_idx, target_idx, influence)
    rows, cols = edge_scores.shape
    for r in range(rows):
        for c in range(cols):
            val = float(edge_scores[r, c])
            if val != 0.0 and r in node_of_index and c in node_of_index:
                edge_records.append((c, r, val))  # source=c, target=r
    edge_ranks, edge_cum = _influence_rank_and_cumulative([e[2] for e in edge_records])
    for i, (src, tgt, val) in enumerate(edge_records):
        edges.append(
            AttributionEdge(
                source=node_of_index[src],
                target=node_of_index[tgt],
                influence=val,
                rank=edge_ranks[i],
                cumulative_influence=edge_cum[i],
                sign=None,
            )
        )

    metadata = {
        "tracer_backend": "circuit-tracer",
        "tracer_version": _ADAPTER_VERSION,
        "scan_name": graph.scan_name if graph.scan_name else None,
        "n_logits": n_logits,
        "n_tokens": n_tokens,
        "replacement_score": None,
        "completeness_score": None,
    }
    return AttributionGraph(state_id=state_id, nodes=nodes, edges=edges, graph_metadata=metadata)


def attribute_fn(
    *,
    model,
    state_id: str,
    prompts: Sequence[str],
    device: str,
    seed: int,
) -> Any:
    """TracerAdapter-compatible entry point (src/circuits/extract.py).

    Runs real attribution on the supplied :class:`circuit_tracer.ReplacementModel`
    for each prompt and merges the resulting graphs into one normalized
    :class:`AttributionGraph` bound to ``state_id``.
    """
    del device  # the ReplacementModel already carries its own device
    if not isinstance(prompts, Sequence) or not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError("adapter requires a non-empty sequence of prompt strings")
    attribute, *_ = _import_circuit_tracer()

    graphs = []
    for prompt in prompts:
        raw = attribute(prompt, model)
        graphs.append(_build_graph(raw, state_id))

    return _merge(graphs, state_id)


def _merge(graphs: Sequence[Any], state_id: str) -> Any:
    """Merge several per-prompt graphs into one graph for one model state.

    Node/edge identity is preserved by node id; an edge's influence is summed
    across prompts. When only one graph is supplied it is returned unchanged
    (after re-tagging to ``state_id``).
    """
    from ..stability import AttributionEdge, AttributionGraph, AttributionNode

    if len(graphs) == 1:
        g = graphs[0]
        return AttributionGraph(
            state_id=state_id, nodes=g.nodes, edges=g.edges, graph_metadata=g.graph_metadata
        )

    node_ids = {}
    nodes = []
    for g in graphs:
        for node in g.nodes:
            if node.node_id not in node_ids:
                node_ids[node.node_id] = len(nodes)
                nodes.append(AttributionNode(
                    node_id=node.node_id, layer=node.layer, component=node.component,
                    influence=node.influence, rank=0, cumulative_influence=0.0,
                    feature_metadata=node.feature_metadata,
                ))

    merged_edges = {}
    for g in graphs:
        for edge in g.edges:
            key = (edge.source, edge.target)
            if key not in merged_edges:
                merged_edges[key] = 0.0
            merged_edges[key] += edge.influence

    edge_records = [(s, t, v) for (s, t), v in merged_edges.items() if v != 0.0]
    edge_records.sort(key=lambda e: (-abs(e[2]), e[0], e[1]))
    edge_ranks, edge_cum = _influence_rank_and_cumulative([e[2] for e in edge_records])
    edges = [
        AttributionEdge(
            source=s, target=t, influence=v, rank=edge_ranks[i],
            cumulative_influence=edge_cum[i], sign=None,
        )
        for i, (s, t, v) in enumerate(edge_records)
    ]

    # Re-rank nodes on summed influence (only feature/logit/token nodes matter).
    node_records = sorted(range(len(nodes)), key=lambda i: (-abs(nodes[i].influence), i))
    node_ranks = [0] * len(nodes)
    node_cum = [0.0] * len(nodes)
    total = sum(abs(nodes[i].influence) for i in node_records)
    running = 0.0
    for pos, i in enumerate(node_records, start=1):
        node_ranks[i] = pos
        running += abs(nodes[i].influence)
        node_cum[i] = (running / total) if total > 0 else 1.0

    nodes = [
        AttributionNode(
            node_id=n.node_id, layer=n.layer, component=n.component, influence=n.influence,
            rank=node_ranks[i], cumulative_influence=node_cum[i], feature_metadata=n.feature_metadata,
        )
        for i, n in enumerate(nodes)
    ]

    metadata = dict(graphs[0].graph_metadata)
    metadata["n_prompts_merged"] = len(graphs)
    return AttributionGraph(state_id=state_id, nodes=nodes, edges=edges, graph_metadata=metadata)