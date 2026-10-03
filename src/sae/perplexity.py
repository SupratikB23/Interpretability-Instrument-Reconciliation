"""Strictly aligned sliding-window perplexity for SAE quantization runs."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerplexityResult:
    condition: str
    bits: int
    ppl: float
    delta_ppl_percent: float
    protocol: str
    n_tokens: int
    seq_len: int
    stride: int
    seed: int
    token_digest: str
    scored_token_indices: tuple[int, ...]


def _token_digest(input_ids: Any) -> str:
    values = input_ids.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def sliding_window_perplexity(
    model: Any,
    input_ids: Any,
    *,
    bits: int,
    window_size: int,
    stride: int,
    token_budget: int,
    seed: int,
    condition: str | None = None,
) -> PerplexityResult:
    """Evaluate causal-LM PPL once per target token with overlapping context.

    ``input_ids`` is tokenized once by the caller and reused for every model
    state. Overlap supplies context only: each global target index contributes
    exactly once, preventing low-bit and FP16 conditions from silently scoring
    different tokens.
    """
    import torch  # lazy heavy dependency
    from torch.nn import functional

    if window_size < 2:
        raise ValueError(f"window_size must be at least 2, got {window_size}")
    if stride < 1 or stride > window_size:
        raise ValueError(f"stride must be in [1, window_size], got {stride}")
    if token_budget < 1:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if bits < 2 or bits > 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")

    tokens = torch.as_tensor(input_ids)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [tokens] or [1, tokens], got {tuple(tokens.shape)}")
    usable = min(int(tokens.shape[1]), token_budget + 1)
    if usable < 2:
        raise ValueError("perplexity requires at least two aligned input tokens")
    tokens = tokens[:, :usable]
    if not torch.isfinite(tokens).all().item():
        raise ValueError("input_ids contains non-finite values")

    total_nll = 0.0
    scored: list[int] = []
    previous_end = 0
    device = getattr(model, "device", tokens.device)

    with torch.no_grad():
        for begin in range(0, usable, stride):
            end = min(begin + window_size, usable)
            target_start = max(previous_end, begin + 1)
            if target_start >= end:
                if end == usable:
                    break
                continue
            window = tokens[:, begin:end].to(device)
            output = model(window, return_type="logits")
            logits = output.logits if hasattr(output, "logits") else output[0] if isinstance(output, tuple) else output
            if logits.ndim != 3 or logits.shape[:2] != window.shape:
                raise ValueError(
                    "model logits must align with input as [1, window, vocab], got "
                    f"{tuple(logits.shape)} for tokens {tuple(window.shape)}"
                )

            local_targets = window[:, 1:]
            local_logits = logits[:, :-1, :]
            local_start = target_start - begin - 1
            local_end = end - begin - 1
            selected_logits = local_logits[:, local_start:local_end, :]
            selected_targets = local_targets[:, local_start:local_end]
            losses = functional.cross_entropy(
                selected_logits.reshape(-1, selected_logits.shape[-1]),
                selected_targets.reshape(-1),
                reduction="sum",
            )
            if not torch.isfinite(losses).item():
                raise FloatingPointError(f"non-finite perplexity loss in token window [{begin}, {end})")
            total_nll += float(losses)
            scored.extend(range(target_start, end))
            previous_end = end
            if end == usable:
                break

    expected = tuple(range(1, usable))
    scored_tuple = tuple(scored)
    if scored_tuple != expected:
        raise RuntimeError(
            "sliding-window alignment failure: expected every target token exactly once; "
            f"expected {len(expected)}, scored {len(scored_tuple)}"
        )
    ppl = math.exp(total_nll / len(scored_tuple))
    return PerplexityResult(
        condition=condition or ("FP16" if bits == 16 else f"INT{bits}"),
        bits=bits,
        ppl=ppl,
        delta_ppl_percent=0.0,
        protocol="sliding_window",
        n_tokens=len(scored_tuple),
        seq_len=window_size,
        stride=stride,
        seed=seed,
        token_digest=_token_digest(tokens),
        scored_token_indices=scored_tuple,
    )


def evaluate_aligned_perplexities(
    models_by_bits: Mapping[int, Any],
    input_ids: Any,
    *,
    window_size: int,
    stride: int,
    token_budget: int,
    seed: int,
) -> dict[int, PerplexityResult]:
    """Evaluate FP16 and quantized states and enforce identical scored tokens."""
    if 16 not in models_by_bits:
        raise ValueError("aligned perplexity evaluation requires an FP16 (bits=16) reference")
    raw = {
        bits: sliding_window_perplexity(
            model,
            input_ids,
            bits=bits,
            window_size=window_size,
            stride=stride,
            token_budget=token_budget,
            seed=seed,
        )
        for bits, model in models_by_bits.items()
    }
    reference = raw[16]
    for bits, result in raw.items():
        if (
            result.token_digest != reference.token_digest
            or result.scored_token_indices != reference.scored_token_indices
        ):
            raise RuntimeError(f"INT{bits} perplexity tokens are not aligned with FP16")
    if not math.isfinite(reference.ppl) or reference.ppl <= 0:
        raise FloatingPointError(f"invalid FP16 perplexity reference: {reference.ppl}")
    return {
        bits: replace(
            result,
            delta_ppl_percent=100.0 * (result.ppl - reference.ppl) / reference.ppl,
        )
        for bits, result in raw.items()
    }


def write_perplexity_csv(path: str | Path, results: Mapping[int, PerplexityResult]) -> None:
    """Persist the Priority-2 perplexity table with a stable scientific schema."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "condition",
        "bits",
        "ppl",
        "delta_ppl_percent",
        "protocol",
        "n_tokens",
        "seq_len",
        "stride",
        "seed",
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bits in sorted(results, reverse=True):
            result = results[bits]
            writer.writerow({name: getattr(result, name) for name in fields})
