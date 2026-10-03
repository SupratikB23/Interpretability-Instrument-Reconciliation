"""Tests for architecture-aware per-output-channel RTN quantization."""

import json

import pytest
import torch

from src.compression.rtn import (
    QuantizedStateDict,
    quantize_state_dict,
    rtn_quantize_tensor,
    select_quantizable_keys,
)

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
    "embed": ["embed"],
    "unembed": ["lm_head", "W_U"],
    "ln": ["layernorm", "norm", "ln"],
    "bias": ["bias"],
}


def _reference_rtn(weight: torch.Tensor, bits: int, axes: tuple[int, ...]) -> torch.Tensor:
    """Independent, loop-based reference with no production helper reuse."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    output_axes = [axis for axis in range(weight.ndim) if axis not in axes]
    result = torch.empty_like(weight)
    output_shape = [weight.shape[axis] for axis in output_axes]
    for output_index in torch.cartesian_prod(
        *[torch.arange(size) for size in output_shape]
    ).reshape(-1, len(output_shape)):
        index = [slice(None)] * weight.ndim
        for axis, coordinate in zip(output_axes, output_index.tolist()):
            index[axis] = coordinate
        channel = weight[tuple(index)]
        maximum = channel.abs().max()
        if maximum == 0:
            result[tuple(index)] = 0
        else:
            scale = maximum / qmax
            result[tuple(index)] = torch.round(channel / scale).clamp(qmin, qmax) * scale
    return result


def _model_like_state_dict() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(18)
    shapes = {
        # HF [out, in].
        "hf.layers.0.self_attn.q_proj.weight": (6, 5),
        # TransformerLens [...head/group, input, output].
        "blocks.0.attn.W_K": (2, 5, 3),
        "blocks.0.attn.W_V": (2, 2, 5, 3),
        # TransformerLens [...head/group, d_head, output].
        "blocks.0.attn.W_O": (2, 3, 5),
        "blocks.0.mlp.W_in": (5, 7),
        "blocks.0.mlp.W_out": (2, 4, 7, 5),
        "blocks.0.mlp.W_gate": (2, 5, 7),
        "embed.W_E": (11, 5),
        "blocks.0.ln1.weight": (5,),
        "unembed.W_U": (5, 11),
    }
    return {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}


def test_independent_numerical_reference_for_all_supported_layouts():
    state = _model_like_state_dict()
    quantized = quantize_state_dict(state, 5, QUANT, SKIP, expected_module_count=7)
    expected_axes = {
        "hf.layers.0.self_attn.q_proj.weight": (1,),
        "blocks.0.attn.W_K": (1,),
        "blocks.0.attn.W_V": (2,),
        "blocks.0.attn.W_O": (0, 1),
        "blocks.0.mlp.W_in": (0,),
        "blocks.0.mlp.W_out": (2,),
        "blocks.0.mlp.W_gate": (1,),
    }
    for name, axes in expected_axes.items():
        expected = _reference_rtn(state[name], 5, axes)
        torch.testing.assert_close(quantized[name], expected, rtol=0, atol=0)


def test_intended_modules_change_and_exclusions_are_identical():
    state = _model_like_state_dict()
    quantized = quantize_state_dict(state, 4, QUANT, SKIP, expected_module_count=7)
    intended = set(state) - {"embed.W_E", "blocks.0.ln1.weight", "unembed.W_U"}

    assert {name for name in state if not torch.equal(state[name], quantized[name])} == intended
    for name in set(state) - intended:
        assert torch.equal(state[name], quantized[name])
        # Skipped params are returned as zero-copy views of the input (they are
        # never mutated by any consumer), so they intentionally share storage —
        # cloning e.g. a large embedding into every quantized state wastes RAM.
        assert state[name].data_ptr() == quantized[name].data_ptr()


def test_no_silent_rank_skipping_for_2d_3d_and_4d_projections():
    state = _model_like_state_dict()
    quantized = quantize_state_dict(state, 4, QUANT, SKIP)

    selected_ranks = {state[name].ndim for name in quantized.audit.selected_names}
    assert selected_ranks == {2, 3, 4}
    assert quantized.audit.selected_count == 7
    assert quantized.audit.quantized_count == 7
    assert quantized.audit.actually_quantized_module_count == 7
    assert all(not torch.equal(state[name], quantized[name]) for name in quantized.audit.selected_names)


def test_error_monotonicity_from_int8_through_int4():
    weight = torch.randn((4, 47), generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    errors = {
        bits: rtn_quantize_tensor(weight, bits).dequantized.sub(weight).square().mean().item()
        for bits in (4, 5, 6, 7, 8)
    }
    assert errors[8] <= errors[7] <= errors[6] <= errors[5] <= errors[4]


def test_reapplication_from_pristine_weights_is_idempotent():
    state = _model_like_state_dict()
    first = quantize_state_dict(state, 6, QUANT, SKIP, expected_module_count=7)
    second = quantize_state_dict(state, 6, QUANT, SKIP, expected_module_count=7)

    assert first.audit.to_dict() == second.audit.to_dict()
    for name in state:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_quantized_values_do_not_receive_a_second_perturbation():
    weight = torch.randn((5, 13), generator=torch.Generator().manual_seed(4))
    first = rtn_quantize_tensor(weight, 6).dequantized
    second = rtn_quantize_tensor(first, 6).dequantized
    torch.testing.assert_close(second, first, rtol=1e-6, atol=1e-7)


def test_module_and_parameter_count_gates_fail_loudly():
    state = _model_like_state_dict()
    selected_elements = sum(
        state[name].numel() for name in select_quantizable_keys(state, QUANT, SKIP)
    )
    quantized = quantize_state_dict(
        state,
        6,
        QUANT,
        SKIP,
        expected_module_count=7,
        expected_quantizable_parameter_count=selected_elements,
    )
    assert quantized.audit.expected_module_count == 7
    assert quantized.audit.expected_quantizable_parameter_count == selected_elements

    with pytest.raises(RuntimeError, match="module-count gate failed"):
        quantize_state_dict(state, 6, QUANT, SKIP, expected_module_count=8)
    with pytest.raises(RuntimeError, match="parameter-count gate failed"):
        quantize_state_dict(
            state,
            6,
            QUANT,
            SKIP,
            expected_quantizable_parameter_count=selected_elements + 1,
        )


def test_audit_contains_metrics_reasons_shapes_and_writes_json(tmp_path):
    state = _model_like_state_dict()
    quantized = quantize_state_dict(state, 5, QUANT, SKIP)
    assert isinstance(quantized, QuantizedStateDict)
    audit = quantized.audit
    assert audit.selected_count == audit.quantized_count == 7
    assert audit.skipped_count == 3
    assert audit.selected_parameter_count == audit.quantized_parameter_count

    q_record = next(module for module in audit.modules if module.name.endswith("W_O"))
    assert q_record.original_shape == q_record.quantized_shape == [2, 3, 5]
    assert q_record.quantization_axes == [0, 1]
    assert q_record.status == "quantized"
    assert q_record.reason is None
    assert q_record.changed_elements > 0
    assert 0 < q_record.percent_changed <= 100
    assert q_record.mse > 0 and q_record.mae > 0 and q_record.max_absolute_error > 0
    assert -1 <= q_record.cosine_similarity <= 1
    assert q_record.norm_ratio > 0

    excluded = next(module for module in audit.modules if module.name == "embed.W_E")
    assert excluded.status == "skipped"
    assert excluded.reason == "matched skip_modules"
    assert excluded.quantized_shape is None

    path = tmp_path / "quantization_audit.json"
    audit.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["quantized_names"] == audit.quantized_names
    assert payload["modules"][0]["bits"] == 5


def test_zero_matches_fail_loudly():
    with pytest.raises(ValueError, match="selected zero parameters"):
        quantize_state_dict({"embed.weight": torch.randn(3, 4)}, 4, {"q": ["q_proj"]}, SKIP)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("blocks.0.attn.W_Q", torch.randn(5), "rank"),
        ("hf.q_proj.weight", torch.randn(2, 3, 4), "expected.*out_features"),
        ("custom.projection", torch.randn(3, 4), "no supported projection layout"),
        ("blocks.0.attn.W_Q", torch.empty(2, 0, 3), "non-zero"),
        ("blocks.0.attn.W_Q", torch.tensor([[[float("nan")]]]), "finite"),
        ("blocks.0.attn.W_Q", torch.ones(2, 3, dtype=torch.int64), "floating dtype"),
    ],
)
def test_selected_unsupported_or_invalid_tensors_fail_loudly(name, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        quantize_state_dict({name: value}, 4, {"selected": [name]}, {})


@pytest.mark.parametrize("bits", [0, 1, 17, -4, True, 4.5])
def test_invalid_bit_width_fails(bits):
    with pytest.raises((TypeError, ValueError), match="bits"):
        rtn_quantize_tensor(torch.randn(2, 3), bits)


def test_all_zero_channels_are_exact_and_have_zero_scale():
    weight = torch.tensor([[0.0, 0.0, 0.0], [1.0, -0.25, 0.5]])
    result = rtn_quantize_tensor(weight, 4)
    assert torch.equal(result.dequantized[0], torch.zeros(3))
    assert result.scale[0, 0].item() == 0
    assert torch.isfinite(result.dequantized).all()


def test_selection_quantizes_only_projections():
    keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.bias",
    ]
    selected = select_quantizable_keys(keys, QUANT, SKIP)
    assert len(selected) == 7
    assert all("proj" in key and "weight" in key for key in selected)
