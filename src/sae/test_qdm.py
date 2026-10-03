"""Regression tests for the QDM streaming pipeline (2606.03002 §2.2, App. I).

These use lightweight fakes standing in for TransformerLens/SAELens so the
sequencing logic is covered without a GPU, a real SAE, or a model download.
"""

import numpy as np
import torch

from src.sae.qdm import _result_from_statistics, _stream_qdm_statistics


class _FakeModel:
    """Minimal HookedTransformer stand-in emitting deterministic activations.

    ``noise`` simulates the activation shift a quantized model produces relative
    to the FP16 reference on identical tokens.
    """

    def __init__(self, hook: str, d_model: int = 8, noise: float = 0.0, seed: int = 0):
        self.hook = hook
        self.d_model = d_model
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        return torch.arange(len(text)).reshape(1, -1)

    def run_with_cache(self, tokens, names_filter=None):
        n = tokens.shape[1]
        base = np.linspace(-1, 1, n * self.d_model).reshape(1, n, self.d_model)
        if self.noise:
            base = base + self.noise * self.rng.normal(0, 1, base.shape)
        return None, {self.hook: torch.tensor(base, dtype=torch.float32)}


class _FakeSAE:
    """Identity-ish encoder: features are the residual acts passed through ReLU."""

    def eval(self):
        return self

    def encode(self, flat):
        return torch.relu(flat)


HOOK = "blocks.12.hook_resid_post"
TEXTS = ["aaaaaaaa", "bbbbbbbb", "cccccccc"]


def test_identical_models_give_unit_correlation():
    """Sanity check on the estimator itself: a model against itself correlates 1.0."""
    model = _FakeModel(HOOK)
    corr, _firing, n = _stream_qdm_statistics(
        model, model, _FakeSAE(), TEXTS, HOOK, max_tokens=1000, seq_len=64
    )
    finite = np.isfinite(corr)
    assert n > 0
    assert np.allclose(corr[finite], 1.0, atol=1e-6)


def test_distinct_models_do_not_report_perfect_survival():
    """Regression: the FP16 and quantized passes must come from DIFFERENT models.

    The original pipeline quantized one model in place and then encoded both the
    "FP16" and "quantized" features from it, so every c_j came out as exactly 1.0
    and the arm reported 100% survival at every bit-width — a green-looking result
    that silently erases the entire effect under study.
    """
    fp16 = _FakeModel(HOOK, noise=0.0)
    quant = _FakeModel(HOOK, noise=0.5, seed=1)
    corr, _firing, _n = _stream_qdm_statistics(
        fp16, quant, _FakeSAE(), TEXTS, HOOK, max_tokens=1000, seq_len=64
    )
    finite = np.isfinite(corr)
    assert finite.any()
    assert not np.allclose(corr[finite], 1.0, atol=1e-6), (
        "perturbed model still reported perfect correlation — "
        "the two passes are reading the same weights"
    )


def test_token_budget_is_respected():
    """The streaming pass must stop at max_tokens, not run the whole corpus."""
    model = _FakeModel(HOOK)
    _, _, n = _stream_qdm_statistics(
        model, model, _FakeSAE(), TEXTS, HOOK, max_tokens=10, seq_len=64
    )
    assert n == 10


def test_survival_rates_use_fp16_active_gate():
    """Only FP16-active features (f_j > 0.001) enter the survival denominator."""
    corr = np.array([0.95, 0.7, 0.2, 0.99])
    firing = np.array([0.5, 0.5, 0.5, 0.0])  # last feature never fires in FP16
    res = _result_from_statistics(
        6, corr, firing,
        active_firing_rate_min=0.001, survived_corr_min=0.9, degraded_corr_min=0.5,
    )
    assert res.n_active == 3
    assert res.survived_pct == 100.0 / 3
    assert res.degraded_pct == 100.0 / 3
    assert res.damaged_pct == 100.0 / 3


class _StatefulFakeModel(_FakeModel):
    """Fake model whose noise scale is set by the applied state dict.

    ``named_parameters`` exposes one tensor; ``_apply_state_dict`` copies the
    quantized state into it, and ``run_with_cache`` reads it as the noise scale.
    """

    def __init__(self, hook: str, seed: int = 0):
        super().__init__(hook, seed=seed)
        self._scale = torch.tensor(0.0)
        self._noise_rng = None

    def named_parameters(self):
        return [("scale", self._scale)]

    def run_with_cache(self, tokens, names_filter=None):
        n = tokens.shape[1]
        base = np.linspace(-1, 1, n * self.d_model).reshape(1, n, self.d_model)
        if float(self._scale) > 0:
            if self._noise_rng is None:
                self._noise_rng = np.random.default_rng(123)
            base = base + float(self._scale) * self._noise_rng.normal(0, 1, base.shape)
        return None, {self.hook: torch.tensor(base, dtype=torch.float32)}


def test_single_model_streamer_matches_two_model_streamer():
    """Chunked state-swapping must reproduce the paired statistics exactly."""
    from src.sae.qdm import _apply_state_dict, _stream_qdm_statistics_single_model

    hook = "blocks.12.hook_resid_post"

    def make_state(scale: float) -> dict:
        return {"scale": torch.tensor(scale)}

    # Two-model reference.
    fp16_ref = _StatefulFakeModel(hook)
    quant_ref = _StatefulFakeModel(hook)
    _apply_state_dict(fp16_ref, make_state(0.0))
    _apply_state_dict(quant_ref, make_state(0.0))
    ref_corr, ref_firing, ref_n = _stream_qdm_statistics(
        fp16_ref, quant_ref, _FakeSAE(), TEXTS, hook, max_tokens=1000, seq_len=64
    )

    # Single-model swapped variant: fp16 pass (0.0 noise) vs quant pass (0.0 noise)
    # gives the identical paired statistics.
    solo = _StatefulFakeModel(hook)
    corr, firing, n = _stream_qdm_statistics_single_model(
        solo, make_state(0.0), make_state(0.0), _FakeSAE(), TEXTS, hook,
        max_tokens=1000, seq_len=64, chunk_texts=2,
    )
    assert n == ref_n
    assert np.allclose(corr, ref_corr, atol=1e-6)
    assert np.allclose(firing, ref_firing, atol=1e-6)


def test_single_model_streamer_collects_activation_stats():
    """The FP16 pass must accumulate mean/peak activations for the SPR artifact."""
    from src.sae.qdm import _stream_qdm_statistics_single_model

    hook = "blocks.12.hook_resid_post"
    stats: dict = {}
    solo = _StatefulFakeModel(hook)
    _stream_qdm_statistics_single_model(
        solo, {"scale": torch.tensor(0.0)}, {"scale": torch.tensor(0.0)},
        _FakeSAE(), TEXTS, hook, max_tokens=1000, seq_len=64,
        chunk_texts=2, activation_stats=stats,
    )
    assert "sum" in stats and "max" in stats
    assert np.all(stats["max"] >= stats["sum"] / stats["max"].size)  # sane ordering
    assert np.all(np.isfinite(stats["sum"])) and np.all(np.isfinite(stats["max"]))
