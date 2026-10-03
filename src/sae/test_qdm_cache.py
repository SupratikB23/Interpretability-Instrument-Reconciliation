"""Equivalence tests: disk-cached two-phase sweep vs the reference streamers.

The cache path must produce the same correlations, firing rates, and
directional statistics as the interleaved two-model implementation — the same
tokens seeing bit-identical weights — while loading each state dict once.
"""

import numpy as np
import torch

from src.sae.qdm import (
    _fp16_feature_cache_pass,
    _qdm_statistics_from_cache,
    _stream_qdm_statistics,
)

HOOK = "blocks.12.hook_resid_post"
TEXTS = ["aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee"]


class _DeterministicSwapModel:
    """Activations are a pure function of (tokens, scale) so re-encoding a text
    in a later pass reproduces exactly the values the FP16 pass cached."""

    def __init__(self, hook: str, d_model: int = 8):
        self.hook = hook
        self.d_model = d_model
        self.scale = torch.zeros(1)

    def named_parameters(self):
        return [("scale", self.scale)]

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        return torch.arange(len(text), dtype=torch.long).reshape(1, -1)

    def run_with_cache(self, tokens, names_filter=None):
        n = tokens.shape[1]
        base = np.linspace(-1, 1, n * self.d_model).reshape(1, n, self.d_model)
        s = float(self.scale.sum())
        pattern = np.sin(np.arange(base.size) * 0.7).reshape(base.shape)
        acts = base + s * (0.3 * pattern + 0.1)
        return None, {self.hook: torch.tensor(acts, dtype=torch.float32)}


class _ReluSAE:
    def eval(self):
        return self

    def encode(self, flat):
        return torch.relu(flat)

    W_dec = torch.eye(8)


def _states():
    return {"scale": torch.zeros(1)}, {"scale": torch.ones(1)}


def test_cache_path_matches_two_model_streamer(tmp_path):
    fp16_state, quant_state = _states()
    model = _DeterministicSwapModel(HOOK)
    sae = _ReluSAE()

    ref_corr, ref_firing, ref_n = _two_model_reference(fp16_state, quant_state, sae)

    meta = _fp16_feature_cache_pass(
        model, fp16_state, sae, TEXTS, HOOK, max_tokens=1000, seq_len=64,
        chunk_texts=2, cache_dir=tmp_path, decoder_directions=None,
        activation_stats=None,
    )
    corr, firing, n, directional = _qdm_statistics_from_cache(
        model, quant_state, sae, TEXTS, HOOK, seq_len=64,
        cache_dir=tmp_path, meta=meta, decoder_directions=None,
    )

    assert n == ref_n
    np.testing.assert_allclose(firing, ref_firing, atol=1e-6)
    finite = np.isfinite(ref_corr) & np.isfinite(corr)
    assert finite.any()
    np.testing.assert_allclose(corr[finite], ref_corr[finite], atol=1e-5)
    assert directional is None


def test_cache_path_matches_with_directions_and_truncated_budget(tmp_path):
    fp16_state, quant_state = _states()
    model = _DeterministicSwapModel(HOOK)
    sae = _ReluSAE()
    directions = np.eye(8)

    ref = _stream_qdm_statistics(
        _StateProxy(fp16_state), _StateProxy(quant_state), sae, TEXTS, HOOK,
        max_tokens=13, seq_len=64, decoder_directions=directions,
    )
    ref_corr, ref_firing, ref_n, ref_dir = ref

    act: dict = {}
    meta = _fp16_feature_cache_pass(
        model, fp16_state, sae, TEXTS, HOOK, max_tokens=13, seq_len=64,
        chunk_texts=2, cache_dir=tmp_path, decoder_directions=directions,
        activation_stats=act,
    )
    corr, firing, n, directional = _qdm_statistics_from_cache(
        model, quant_state, sae, TEXTS, HOOK, seq_len=64,
        cache_dir=tmp_path, meta=meta, decoder_directions=directions,
    )

    assert n == ref_n == 13
    np.testing.assert_allclose(firing, ref_firing, atol=1e-6)
    finite = np.isfinite(ref_corr) & np.isfinite(corr)
    np.testing.assert_allclose(corr[finite], ref_corr[finite], atol=1e-5)
    assert directional is not None and ref_dir is not None
    np.testing.assert_allclose(directional[0], ref_dir[0], atol=1e-5)
    np.testing.assert_allclose(directional[1], ref_dir[1], atol=1e-5)
    assert "sum" in act and "max" in act


def test_cache_pass_rebuilds_over_stale_files_from_a_killed_run(tmp_path):
    """Regression: a killed mid-sweep run leaves orphaned chunk .npy files; the
    next FP16 pass must rebuild the cache directory fresh instead of mixing
    generations on disk."""
    cache_dir = tmp_path / "_fp16_feature_cache"
    cache_dir.mkdir()
    stale = cache_dir / "chunk_000999_x.npy"
    np.save(stale, np.zeros((4, 8), dtype=np.float32))

    fp16_state, _quant_state = _states()
    model = _DeterministicSwapModel(HOOK)
    sae = _ReluSAE()

    meta = _fp16_feature_cache_pass(
        model, fp16_state, sae, TEXTS, HOOK, max_tokens=1000, seq_len=64,
        chunk_texts=2, cache_dir=cache_dir, decoder_directions=None,
        activation_stats=None,
    )

    assert not stale.exists(), "stale chunk files from a killed run must be cleared"
    on_disk = {path.name for path in cache_dir.glob("chunk_*_x.npy")}
    expected = {f"chunk_{chunk['index']:06d}_x.npy" for chunk in meta["chunks"]}
    assert on_disk == expected, "cache directory must contain exactly this run's chunks"


def _two_model_reference(fp16_state, quant_state, sae):
    return _stream_qdm_statistics(
        _StateProxy(fp16_state), _StateProxy(quant_state), sae, TEXTS, HOOK,
        max_tokens=1000, seq_len=64,
    )


class _StateProxy:
    """Presents a state dict as a model whose activations follow that state."""

    def __init__(self, state):
        self.state = state
        self.hook = HOOK
        self.scale = state["scale"]

    def named_parameters(self):
        return [("scale", self.scale)]

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        return torch.arange(len(text), dtype=torch.long).reshape(1, -1)

    def run_with_cache(self, tokens, names_filter=None):
        proxy = _DeterministicSwapModel(HOOK)
        proxy.scale = self.state["scale"]
        return proxy.run_with_cache(tokens, names_filter)
