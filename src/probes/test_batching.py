"""Probe extraction equivalence contracts.

Uses a deterministic causal fake model so the equivalence between
single-example extraction and padded-batch extraction is checked without a
model download. Causal masking guarantees trailing pad tokens never influence
earlier positions; these tests pin the plumbing that relies on it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.probes.run import (
    _extract_activations_batched,
    _extract_activations_multi,
    _stratified_budget_selection,
)

D_MODEL = 8
D_VOCAB = 64
LAYERS = [0, 1]


class _FakeCausalModel:
    """Activations depend only on tokens up to each position (causal)."""

    def __init__(self):
        self.cfg = SimpleNamespace(device="cpu")
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        ids = [2] if prepend_bos else []
        ids += [(ord(c) % (D_VOCAB - 3)) + 3 for c in text]
        return torch.tensor([ids])

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None):
        b, t = tokens.shape
        cums = torch.cumsum(tokens.double(), dim=1)  # causal prefix summary
        positions = torch.arange(t, dtype=torch.float64).unsqueeze(0).expand(b, t)
        acts = torch.stack(
            [torch.sin(cums * (i + 1)) + torch.cos(positions * (i + 1)) for i in range(D_MODEL)],
            dim=-1,
        )
        return None, {hook: acts for hook in names_filter}


TEXTS = ["ab", "the longer probe prompt body", "x", "medium length example here", "y"]


def test_batched_extraction_matches_single_example_extraction():
    """Padded batches must reproduce single-example readouts exactly."""
    model = _FakeCausalModel()
    single = _extract_activations_multi(model, TEXTS, LAYERS, seq_len=32)
    batched = _extract_activations_batched(model, TEXTS, LAYERS, seq_len=32, batch_size=2)
    for layer in LAYERS:
        assert single[layer].shape == batched[layer].shape
        assert np.allclose(single[layer], batched[layer], atol=1e-12), (
            "batched extraction diverged from single-example extraction — padding or "
            "final-non-padding indexing is wrong"
        )


def test_batched_extraction_matches_across_batch_sizes():
    model = _FakeCausalModel()
    reference = _extract_activations_batched(model, TEXTS, LAYERS, seq_len=32, batch_size=1)
    for batch_size in (2, 3, len(TEXTS)):
        out = _extract_activations_batched(model, TEXTS, LAYERS, seq_len=32, batch_size=batch_size)
        for layer in LAYERS:
            assert np.allclose(reference[layer], out[layer], atol=1e-12)


def test_multi_layer_extraction_matches_per_layer():
    """One multi-layer forward must equal separate per-layer forwards."""
    model = _FakeCausalModel()
    combined = _extract_activations_batched(model, TEXTS, LAYERS, seq_len=32, batch_size=2)
    for layer in LAYERS:
        solo = _extract_activations_batched(model, TEXTS, [layer], seq_len=32, batch_size=2)
        assert np.allclose(combined[layer], solo[layer], atol=1e-12)


def test_batch_size_must_be_positive():
    model = _FakeCausalModel()
    with pytest.raises(ValueError, match="batch_size"):
        _extract_activations_batched(model, TEXTS, LAYERS, 32, 0)


def test_stratified_budget_selection_is_class_balanced_and_deterministic():
    """Budget selection must keep both classes and persist a stable subset."""
    texts = [f"text-{i}" for i in range(40)]
    labels = np.array([0] * 30 + [1] * 10, dtype=np.int64)
    first = _stratified_budget_selection(texts, labels, 8, seed=42)
    second = _stratified_budget_selection(texts, labels, 8, seed=42)
    assert first == second, "budget selection must be deterministic for a fixed seed"
    assert len(first) == 8
    chosen = sorted(int(labels[i]) for i in first)
    assert chosen.count(0) == 4 and chosen.count(1) == 4, "selection must be class-stratified"


def test_stratified_budget_selection_fills_when_a_class_is_exhausted():
    texts = [f"text-{i}" for i in range(10)]
    labels = np.array([0] * 2 + [1] * 8, dtype=np.int64)
    chosen = _stratified_budget_selection(texts, labels, 6, seed=0)
    assert len(chosen) == 6
    assert len(set(chosen)) == 6
