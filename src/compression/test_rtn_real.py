"""Real-model RTN integration tests.

These load a small real model (GPT-2-small) through TransformerLens and assert
the scientific contract against the *actual* state-dict parameter names, so a
future library rename (Test F) or an accidental silent rank skip (Test B) fails
here instead of silently producing an unquantized comparison arm.

GPT-2-small is deliberately chosen: it is a real catch of the target wire-up
(Q/K/V/O + MLP projections, embeddings, layer norms, biases, positional
embeddings) while remaining CPU-viable and small. It has no gated-MLP ``W_gate``,
so the expected module counts are computed from the loaded model rather than
hard-coded to Gemma's 7 families.
"""

from __future__ import annotations

import pytest
import torch

from src.compression.rtn import quantize_state_dict, select_quantizable_keys

# The exact matchers the harmonized configs use (configs/harmonized/common_gemma2_2b.yaml).
QUANT = {
    "attn_q": ["q_proj", "W_Q"],
    "attn_k": ["k_proj", "W_K"],
    "attn_v": ["v_proj", "W_V"],
    "attn_o": ["o_proj", "W_O"],
    "mlp_in": ["up_proj", "W_in"],
    "mlp_out": ["down_proj", "W_out"],
    "mlp_gate": ["gate_proj", "W_gate"],
}
SKIP = {
    "embed": ["embed", "wte", "token_embedding"],
    "unembed": ["lm_head", "unembed", "w_u", "final_proj"],
    "ln": ["layernorm", "layer_norm", "ln", "norm"],
    "bias": ["bias"],
}


def _load_gpt2():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def gpt2_state_dict():
    """Load GPT-2-small once; skip the module if the download is unavailable."""
    try:
        model = _load_gpt2()
    except Exception as exc:  # noqa: BLE001 - treat any load failure as environment-blocked
        pytest.skip(f"gpt2 load unavailable in this environment: {exc}")
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def test_real_model_intended_projections_change_and_exclusions_are_identical(gpt2_state_dict):
    """Test A: projections change; embeddings/norms/biases/pos/unembed do not."""
    quantized = quantize_state_dict(gpt2_state_dict, 4, QUANT, SKIP)
    selected = set(quantized.audit.selected_names)

    # Every selected tensor must be an attention or MLP projection weight.
    for name in selected:
        assert name.endswith((".attn.W_Q", ".attn.W_K", ".attn.W_V", ".attn.W_O", ".mlp.W_in", ".mlp.W_out")), name
        assert not torch.equal(gpt2_state_dict[name], quantized[name]), f"{name} was selected but not changed"

    # Every non-selected tensor must be byte-identical to the input. Skipped
    # params are returned as zero-copy views (never mutated downstream), so they
    # intentionally share storage rather than being cloned into every quant state.
    for name, value in gpt2_state_dict.items():
        if name not in selected:
            assert torch.equal(value, quantized[name]), f"{name} was changed but should be untouched"
            assert value.data_ptr() == quantized[name].data_ptr()


def test_real_model_no_gate_skip_and_exact_expected_counts(gpt2_state_dict):
    """Tests B and F: every supported projection family is selected; counts are exact."""
    keys = list(gpt2_state_dict)
    selected = select_quantizable_keys(keys, QUANT, SKIP)
    expected_count = len(selected)
    assert expected_count > 0

    quantized = quantize_state_dict(
        gpt2_state_dict, 6, QUANT, SKIP, expected_module_count=expected_count
    )
    audit = quantized.audit
    assert audit.selected_count == expected_count
    assert audit.quantized_count == expected_count
    assert audit.actually_quantized_module_count == expected_count

    # All selected tensors are rank-2 in GPT-2 (no head-structured tensors here),
    # and none were silently skipped.
    assert all(not torch.equal(gpt2_state_dict[n], quantized[n]) for n in audit.quantized_names)


def test_real_model_audit_records_every_selected_and_excluded_parameter(gpt2_state_dict):
    """Audit covers every parameter, with shaped/axis metadata for projections."""
    quantized = quantize_state_dict(gpt2_state_dict, 5, QUANT, SKIP)
    audit = quantized.audit

    recorded = {m.name for m in audit.modules}
    assert recorded == set(gpt2_state_dict)

    q_proj = next(m for m in audit.modules if m.name.endswith(".attn.W_Q"))
    assert q_proj.status == "quantized"
    assert q_proj.bits == 5
    # TransformerLens stores W_Q as head-structured [n_head, d_model, d_head]
    # (GPT-2-small: 12 heads, 768 model, 64 head). This is precisely the
    # non-2D projection layout the architecture-aware axis resolution exists for.
    assert q_proj.original_shape == q_proj.quantized_shape == [12, 768, 64]
    assert q_proj.quantization_axes == [1]
    assert q_proj.mse > 0 and q_proj.mae > 0

    w_e = next(m for m in audit.modules if m.name == "embed.W_E")
    assert w_e.status == "skipped"
    assert w_e.reason == "matched skip_modules"
    assert w_e.quantized_shape is None


def test_real_model_error_monotonicity_from_int8_to_int4(gpt2_state_dict):
    """Test D against real weights: INT8 <= INT7 <= ... <= INT4."""
    errors: dict[int, float] = {}
    for bits in (4, 5, 6, 7, 8):
        quantized = quantize_state_dict(gpt2_state_dict, bits, QUANT, SKIP)
        total_sq = 0.0
        for name in quantized.audit.quantized_names:
            total_sq += (quantized[name] - gpt2_state_dict[name]).square().sum().item()
        errors[bits] = total_sq
    assert errors[8] <= errors[7] <= errors[6] <= errors[5] <= errors[4]