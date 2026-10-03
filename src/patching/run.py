"""Memory-safe real TransformerLens activation-patching stage driver."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np

from src.compression.rtn import quantize_state_dict
from src.patching.nie_pie import PatchingCondition, run_patching_arm

_DTYPE_MAP = {"float16": "float16", "float32": "float32", "bfloat16": "bfloat16"}


def _load_model(cfg, bits: int | None):
    import torch
    import transformer_lens

    model_spec = (cfg.experiment or {}).get("model") or {}
    dtype_name = str(cfg.model["model"].get("dtype", "float16"))
    dtype = getattr(torch, _DTYPE_MAP.get(dtype_name, "float16"))
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_spec.get("identifier") or cfg.model["model"]["name"],
        device=cfg.model["model"].get("device", "cuda"), dtype=dtype,
        revision=model_spec.get("revision"), trust_remote_code=False,
    )
    if bits is not None:
        compression = cfg.quant["compression"]
        original = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        quantized = quantize_state_dict(original, bits, compression["quant_modules"], compression["skip_modules"])
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(quantized[name].to(value.device, dtype=value.dtype))
    return model


class _SingleSlotModelCache:
    """Holds at most one loaded model; evicts on condition switch.

    ``decompose_component`` issues four consecutive metric calls per condition, so
    a last-used slot turns 24 full model loads per sweep into one per condition
    without ever holding two Gemma copies on the GPU.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._key: int | None | str = None
        self._model = None

    def get(self, bits: int | None):
        import torch

        key = "fp16" if bits is None else bits
        if self._key != key or self._model is None:
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model = _load_model(self._cfg, bits)
            self._key = key
        return self._model


def _metric(model, tokens, positive: str, negative: str, hook: str | None = None, activation=None) -> float:
    import torch
    hooks = []
    if hook is not None:
        tensor = torch.as_tensor(
            activation, device=tokens.device, dtype=next(model.parameters()).dtype
        )
        # TransformerLens calls forward hooks as ``fn(activation, hook=hook_point)``
        # — ``hook`` is a keyword arg — so the replacement fn must accept it (or
        # any keyword). ``**_kw`` absorbs it; the hook just overwrites the
        # activation with the patched tensor.
        hooks = [(hook, lambda _value, **_kw: tensor)]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
    return float(logits[0, -1, model.to_single_token(positive)] - logits[0, -1, model.to_single_token(negative)])


def _condition(cfg, *, label: str, bits: int | None, clean_prompt: str, corrupted_prompt: str, components: list[str], positive: str, negative: str, cache: _SingleSlotModelCache) -> PatchingCondition:
    """Capture condition-owned activations, then patch through the cached condition model."""
    model = cache.get(bits)
    clean_tokens = model.to_tokens(clean_prompt, prepend_bos=True, truncate=True)
    corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True, truncate=True)
    with __import__("torch").no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=components)
        _, corrupt_cache = model.run_with_cache(corrupted_tokens, names_filter=components)
    clean = {component: clean_cache[component].detach().float().cpu().numpy() for component in components}
    corrupted = {component: corrupt_cache[component].detach().float().cpu().numpy() for component in components}
    if any(clean[component].shape != corrupted[component].shape for component in components):
        raise ValueError("patching clean/corrupted prompts tokenize to incompatible hook shapes; choose equal-length prompt pairs")
    digest = hashlib.sha256((label + repr(bits)).encode()).hexdigest()[:16]
    del clean_cache, corrupt_cache

    def model_fn(component: str, activation: np.ndarray, baseline: str) -> float:
        active = cache.get(bits)
        tokens = active.to_tokens(clean_prompt if baseline == "clean" else corrupted_prompt, prepend_bos=True, truncate=True)
        return _metric(active, tokens, positive, negative, component, activation)

    return PatchingCondition(label=label, state_id=f"{label}:{digest}", state={"label": label, "bits": bits}, model_fn=model_fn, clean_acts=clean, corrupted_acts=corrupted)


def _write_patching_spr(results, conditions, output_dir: Path) -> Path:
    rows = []
    for result in results:
        if result.label == "fp16":
            continue
        clean = np.asarray(conditions[result.label].clean_acts[result.component], dtype=np.float64)
        reference = np.asarray(conditions["fp16"].clean_acts[result.component], dtype=np.float64)
        signal = float(np.abs(reference).mean())
        perturbation = float(np.abs(clean - reference).mean())
        rows.append((result, signal, perturbation))
    if not rows:
        raise RuntimeError("patching SPR requires at least one quantized result")
    directory = output_dir / "harmonized" / "spr" / "instruments"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "patching.npz"
    np.savez_compressed(
        path,
        spr=np.asarray([signal / max(perturbation, 1e-8) for _, signal, perturbation in rows]),
        target=np.asarray([np.clip(abs(result.normalized_pie_damage), 0., 1.) for result, _, _ in rows]),
        bit_width=np.asarray([int(result.label.removeprefix("int")) for result, _, _ in rows]),
        groups=np.asarray([result.component for result, _, _ in rows]),
        signal=np.asarray([signal for _, signal, _ in rows]), perturbation=np.asarray([perturbation for _, _, perturbation in rows]),
        target_kind=np.asarray("continuous"), direction_source=np.asarray("activation_magnitude_approximation"),
    )
    return path


def run_patching_stage(cfg, *, output_dir: Path) -> list:
    patch_cfg = (cfg.experiment or {}).get("patching")
    if not isinstance(patch_cfg, dict):
        raise ValueError("patching stage requires explicit 'patching:' config")  # noqa: TRY004 - config-contract violation is deliberately ValueError
    components = patch_cfg.get("components")
    prompts = patch_cfg.get("prompts") or {}
    if not isinstance(components, list) or not components or not all(isinstance(c, str) for c in components):
        raise ValueError("patching.components must be a non-empty list of real hook names")
    clean_prompt, corrupted_prompt = prompts.get("clean"), prompts.get("corrupted")
    if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
        raise ValueError("patching.prompts.clean and .corrupted must be explicit strings")  # noqa: TRY004 - config-contract violation is deliberately ValueError
    positive, negative = patch_cfg.get("positive_token"), patch_cfg.get("negative_token")
    if not isinstance(positive, str) or not isinstance(negative, str):
        raise ValueError("patching requires positive_token and negative_token")  # noqa: TRY004 - config-contract violation is deliberately ValueError
    cache = _SingleSlotModelCache(cfg)
    conditions = {"fp16": _condition(cfg, label="fp16", bits=None, clean_prompt=clean_prompt, corrupted_prompt=corrupted_prompt, components=components, positive=positive, negative=negative, cache=cache)}
    for bits in cfg.quant["compression"]["bit_widths"]:
        label = f"int{bits}"
        conditions[label] = _condition(cfg, label=label, bits=bits, clean_prompt=clean_prompt, corrupted_prompt=corrupted_prompt, components=components, positive=positive, negative=negative, cache=cache)
    results = run_patching_arm(components, conditions, output_dir=output_dir, mode="harmonized")
    _write_patching_spr(results, conditions, output_dir)
    return results
