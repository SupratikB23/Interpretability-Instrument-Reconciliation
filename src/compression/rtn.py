"""Architecture-aware symmetric round-to-nearest (RTN) quantization.

Hugging Face linear weights are stored as ``[out, in]``. TransformerLens stores
attention and MLP projections with their semantic dimensions exposed (for
example, ``W_Q`` is ``[head, d_model, d_head]`` and ``W_O`` is
``[head, d_head, d_model]``). This module resolves those layouts by parameter
name instead of using tensor rank as a selection rule.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_MIN_BITS = 2
_MAX_BITS = 16
_SUPPORTED_RANKS = (2, 3, 4)


@dataclass(frozen=True)
class QuantResult:
    """Dequantized weights plus the symmetric per-channel scale tensor."""

    dequantized: torch.Tensor
    scale: torch.Tensor
    bits: int
    qmin: int
    qmax: int


@dataclass(frozen=True)
class ModuleQuantizationAudit:
    """Machine-readable audit information for one state-dict parameter."""

    name: str
    original_shape: list[int]
    quantized_shape: list[int] | None
    bits: int
    selected: bool
    status: str
    reason: str | None
    quantization_axes: list[int] | None
    changed_elements: int
    percent_changed: float
    mse: float
    mae: float
    max_absolute_error: float
    cosine_similarity: float
    norm_ratio: float


@dataclass(frozen=True)
class QuantizationAudit:
    """Complete RTN coverage report, suitable for JSON persistence."""

    bits: int
    selected_count: int
    quantized_count: int
    skipped_count: int
    selected_names: list[str]
    quantized_names: list[str]
    skipped_names: list[str]
    selected_parameter_count: int
    quantized_parameter_count: int
    expected_quantizable_parameter_count: int
    expected_module_count: int
    actually_quantized_module_count: int
    modules: list[ModuleQuantizationAudit]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )


class QuantizedStateDict(dict[str, torch.Tensor]):
    """Backwards-compatible state dict that also carries its audit report."""

    audit: QuantizationAudit

    def __init__(self, values: dict[str, torch.Tensor], audit: QuantizationAudit):
        super().__init__(values)
        self.audit = audit


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(f"bits must be an integer, got {type(bits).__name__}")
    if not _MIN_BITS <= bits <= _MAX_BITS:
        raise ValueError(
            f"bits must be in [{_MIN_BITS}, {_MAX_BITS}], got {bits}"
        )


def _validate_weight(weight: torch.Tensor, input_axes: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if not (weight.is_floating_point() or weight.is_complex()):
        raise TypeError(f"weight must have a floating dtype, got {weight.dtype}")
    if weight.is_complex():
        raise TypeError(f"complex weights are unsupported, got {weight.dtype}")
    if weight.ndim not in _SUPPORTED_RANKS:
        raise ValueError(
            f"projection weight rank must be one of {_SUPPORTED_RANKS}, "
            f"got shape {tuple(weight.shape)}"
        )
    if any(size == 0 for size in weight.shape):
        raise ValueError(f"projection weight dimensions must be non-zero, got {tuple(weight.shape)}")
    if not torch.isfinite(weight).all().item():
        raise ValueError("projection weight must contain only finite values")

    normalized = tuple(sorted({axis % weight.ndim for axis in input_axes}))
    if not normalized:
        raise ValueError("at least one input/quantization axis is required")
    if any(not -weight.ndim <= axis < weight.ndim for axis in input_axes):
        raise ValueError(
            f"input axis out of range for shape {tuple(weight.shape)}: {tuple(input_axes)}"
        )
    if len(normalized) >= weight.ndim:
        raise ValueError("quantization axes cannot consume every dimension; no output channels remain")
    return normalized


def _round_nearest(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest with ties handled by ``torch.round`` (half-to-even)."""
    return torch.round(x)


def rtn_quantize_tensor(
    weight: torch.Tensor,
    bits: int,
    per_output_channel: bool = True,
    eps: float = 1e-12,
    input_axes: int | Sequence[int] = -1,
) -> QuantResult:
    """Symmetric RTN quantization over explicit semantic input axes.

    ``input_axes=-1`` preserves the historical ``[out, in]`` behavior. Callers
    quantizing TransformerLens tensors should pass axes resolved from the
    projection layout. All-zero output channels remain exactly zero.
    """
    _validate_bits(bits)
    if not isinstance(per_output_channel, bool):
        raise TypeError("per_output_channel must be a bool")
    if not isinstance(eps, (int, float)) or not 0 < float(eps) < float("inf"):
        raise ValueError(f"eps must be finite and positive, got {eps}")
    axes_arg = (input_axes,) if isinstance(input_axes, int) else tuple(input_axes)
    axes = _validate_weight(weight, axes_arg)

    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)
    if per_output_channel:
        maxima = weight.abs().amax(dim=axes, keepdim=True)
    else:
        maxima = weight.abs().amax()
    raw_scale = maxima / qmax
    safe_scale = raw_scale.clamp_min(float(eps))
    q = torch.clamp(_round_nearest(weight / safe_scale), qmin, qmax)
    dequantized = q * safe_scale
    # This explicit mask guarantees exact zeros even for unusual low-precision eps.
    dequantized = torch.where(maxima == 0, torch.zeros_like(dequantized), dequantized)
    scale = torch.where(maxima == 0, torch.zeros_like(raw_scale), raw_scale)
    return QuantResult(dequantized, scale, bits, qmin, qmax)


def _matches(name: str, patterns: Iterable[str]) -> bool:
    low = name.lower()
    return any(pattern.lower() in low for pattern in patterns)


def select_quantizable_keys(
    keys: Iterable[str],
    quant_modules: Mapping[str, Iterable[str]],
    skip_modules: Mapping[str, Iterable[str]],
) -> list[str]:
    """Select configured projection keys, with exclusions taking precedence."""
    quant_patterns = [pattern for patterns in quant_modules.values() for pattern in patterns]
    skip_patterns = [pattern for patterns in skip_modules.values() for pattern in patterns]
    selected: list[str] = []
    for key in keys:
        if _matches(key, skip_patterns):
            continue
        if quant_patterns and not _matches(key, quant_patterns):
            continue
        selected.append(key)
    return selected


def _transformer_lens_projection(name: str) -> str | None:
    low = name.lower()
    for projection in ("w_q", "w_k", "w_v", "w_o", "w_in", "w_out", "w_gate"):
        if low.endswith((projection, f"{projection}.weight")) or f".{projection}." in low:
            return projection
    return None


def _hugging_face_projection(name: str) -> bool:
    low = name.lower()
    return any(
        marker in low
        for marker in (
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"
        )
    )


def _resolve_input_axes(name: str, weight: torch.Tensor) -> tuple[int, ...]:
    """Resolve semantic input axes for supported HF and TransformerLens layouts."""
    tl_projection = _transformer_lens_projection(name)
    if tl_projection is not None:
        if weight.ndim not in _SUPPORTED_RANKS:
            raise ValueError(
                f"unsupported TransformerLens {tl_projection} rank {weight.ndim} with "
                f"shape {tuple(weight.shape)}; supported ranks are {_SUPPORTED_RANKS}"
            )
        if tl_projection == "w_o" and weight.ndim >= 3:
            # [...head/group, d_head, d_model]: all axes before d_model are input.
            return tuple(range(weight.ndim - 1))
        # W_Q/K/V: [...head/group, d_model, d_head]. W_in/gate/out:
        # [...expert/group, input, output]. Preserve leading independent groups.
        return (weight.ndim - 2,)

    if _hugging_face_projection(name):
        if weight.ndim != 2:
            raise ValueError(
                f"unsupported Hugging Face projection shape {tuple(weight.shape)} for {name}; "
                "expected [out_features, in_features]"
            )
        return (1,)

    raise ValueError(
        f"selected parameter {name!r} has no supported projection layout; "
        "use Hugging Face *_proj weights or TransformerLens W_Q/W_K/W_V/W_O/"
        "W_in/W_out/W_gate tensors"
    )


def _metrics(
    original: torch.Tensor, quantized: torch.Tensor
) -> tuple[int, float, float, float, float, float, float]:
    error = quantized - original
    changed = int(torch.count_nonzero(error).item())
    percent = 100.0 * changed / original.numel()
    original64 = original.detach().to(torch.float64).reshape(-1)
    quantized64 = quantized.detach().to(torch.float64).reshape(-1)
    error64 = quantized64 - original64
    original_norm = torch.linalg.vector_norm(original64)
    quantized_norm = torch.linalg.vector_norm(quantized64)
    if original_norm.item() == 0.0:
        cosine = 1.0 if quantized_norm.item() == 0.0 else 0.0
        norm_ratio = 1.0 if quantized_norm.item() == 0.0 else float("inf")
    else:
        cosine = float(torch.dot(original64, quantized64) / (original_norm * quantized_norm)) if quantized_norm.item() else 0.0
        norm_ratio = float(quantized_norm / original_norm)
    return (
        changed,
        percent,
        float(error64.square().mean()),
        float(error64.abs().mean()),
        float(error64.abs().max()),
        cosine,
        norm_ratio,
    )


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    bits: int,
    quant_modules: Mapping[str, Iterable[str]],
    skip_modules: Mapping[str, Iterable[str]],
    per_output_channel: bool = True,
    *,
    expected_module_count: int | None = None,
    expected_quantizable_parameter_count: int | None = None,
) -> QuantizedStateDict:
    """Quantize configured projections and attach a complete coverage audit.

    Selected unsupported tensors, empty selection, and expected-count mismatches
    raise instead of silently producing a partially quantized model. The result
    remains a ``dict`` subclass for compatibility; access the report as
    ``result.audit`` and persist it with ``result.audit.write_json(path)``.
    """
    _validate_bits(bits)
    selected = select_quantizable_keys(state_dict, quant_modules, skip_modules)
    if not selected:
        raise ValueError(
            "RTN selected zero parameters; check quant_modules against the model's "
            "state-dict naming convention"
        )
    selected_set = set(selected)
    selected_parameters = sum(state_dict[name].numel() for name in selected)
    expected_modules = len(selected) if expected_module_count is None else expected_module_count
    expected_parameters = (
        selected_parameters
        if expected_quantizable_parameter_count is None
        else expected_quantizable_parameter_count
    )
    if expected_modules < 0 or expected_parameters < 0:
        raise ValueError("expected counts must be non-negative")
    if len(selected) != expected_modules:
        raise RuntimeError(
            f"RTN module-count gate failed: expected {expected_modules}, selected {len(selected)}; "
            f"selected={selected}"
        )
    if selected_parameters != expected_parameters:
        raise RuntimeError(
            "RTN quantizable-parameter-count gate failed: "
            f"expected {expected_parameters}, selected {selected_parameters}"
        )

    quantized_state: dict[str, torch.Tensor] = {}
    audits: list[ModuleQuantizationAudit] = []
    quantized_names: list[str] = []
    skipped_names: list[str] = []
    skip_patterns = [pattern for patterns in skip_modules.values() for pattern in patterns]
    quant_patterns = [pattern for patterns in quant_modules.values() for pattern in patterns]

    for name, value in state_dict.items():
        shape = list(value.shape)
        if name in selected_set:
            try:
                axes = _resolve_input_axes(name, value)
                result = rtn_quantize_tensor(
                    value, bits, per_output_channel=per_output_channel, input_axes=axes
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"cannot quantize selected parameter {name!r} with shape {tuple(value.shape)}: {error}"
                ) from error
            quantized = result.dequantized
            quantized_state[name] = quantized
            quantized_names.append(name)
            changed, percent, mse, mae, max_error, cosine, norm_ratio = _metrics(value, quantized)
            audits.append(
                ModuleQuantizationAudit(
                    name, shape, list(quantized.shape), bits, True, "quantized", None,
                    list(axes), changed, percent, mse, mae, max_error, cosine, norm_ratio
                )
            )
        else:
            # Skipped params (embeddings, unembed, norms, biases) are byte-identical
            # to the input — quantization never touches them. Return a detached
            # *view* of the input tensor instead of a full clone: every consumer
            # here only reads the returned dict (``_apply_state_dict`` copies it
            # *into* the model; the audit reads shapes), so cloning e.g. Gemma-2's
            # ~1.18 GB embedding into every one of the {8,7,6,5,4} quantized states
            # is pure wasted RAM/copy time. The view shares storage with the input,
            # which is never mutated in any call site.
            quantized_state[name] = value.detach()
            reason = (
                "matched skip_modules"
                if _matches(name, skip_patterns)
                else "did not match quant_modules" if quant_patterns else "not selected"
            )
            skipped_names.append(name)
            audits.append(
                ModuleQuantizationAudit(
                    name, shape, None, bits, False, "skipped", reason, None,
                    0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0
                )
            )

    if len(quantized_names) != expected_modules:
        raise RuntimeError(
            f"RTN module-count gate failed: expected {expected_modules}, quantized {len(quantized_names)}"
        )
    quantized_parameters = sum(state_dict[name].numel() for name in quantized_names)
    if quantized_parameters != expected_parameters:
        raise RuntimeError(
            "RTN quantizable-parameter-count gate failed after quantization: "
            f"expected {expected_parameters}, quantized {quantized_parameters}"
        )

    audit = QuantizationAudit(
        bits=bits,
        selected_count=len(selected),
        quantized_count=len(quantized_names),
        skipped_count=len(skipped_names),
        selected_names=selected,
        quantized_names=quantized_names,
        skipped_names=skipped_names,
        selected_parameter_count=selected_parameters,
        quantized_parameter_count=quantized_parameters,
        expected_quantizable_parameter_count=expected_parameters,
        expected_module_count=expected_modules,
        actually_quantized_module_count=len(quantized_names),
        modules=audits,
    )
    return QuantizedStateDict(quantized_state, audit)
