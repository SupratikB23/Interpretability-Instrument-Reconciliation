"""Real circuit-tracer stage driver for the harmonized experiment."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

from src.circuits.backends.circuit_tracer_adapter import attribute_fn
from src.circuits.extract import TracerModelState, run_circus_sweep
from src.compression.rtn import quantize_state_dict


def _state_id(label: str, model) -> str:
    """Condition-specific fingerprint from actual loaded parameters."""
    digest = hashlib.sha256(label.encode("utf-8"))
    for name, parameter in model.named_parameters():
        if getattr(parameter, "is_meta", False):
            continue  # lazily-loaded transcoder decoders carry no values to hash
        digest.update(name.encode("utf-8"))
        values = parameter.detach().reshape(-1)
        # A full hash would duplicate large weights on CPU.  Shape plus stable
        # endpoint samples is enough to reject accidental FP16 reuse; the RTN
        # audit remains the full coverage proof.
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        if values.numel():
            digest.update(values[: min(32, values.numel())].float().cpu().numpy().tobytes())
    return f"{label}:{digest.hexdigest()[:20]}"


def run_circuits_stage(cfg, *, output_dir: Path, seed: int) -> dict:
    """Load every actual RTN condition, trace it, and write CIRCUS artifacts."""
    import torch
    try:
        from circuit_tracer import ReplacementModel
    except ImportError as exc:
        raise RuntimeError(
            "real tracer adapter/backend is unavailable: download circuit-tracer "
            "from GitHub on the GPU machine with `pip install git+https://github.com/safety-research/circuit-tracer.git`"
        ) from exc

    circuit_cfg = (cfg.experiment or {}).get("circuits")
    if not isinstance(circuit_cfg, dict):
        raise ValueError("circuits stage requires a config 'circuits:' block with prompts and transcoder_set")  # noqa: TRY004 - config-contract violation is deliberately ValueError
    prompts = circuit_cfg.get("prompts")
    transcoder_set = circuit_cfg.get("transcoder_set")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("circuits.prompts must be a non-empty explicit prompt list")
    if not isinstance(transcoder_set, str) or not transcoder_set.strip():
        raise ValueError("circuits.transcoder_set must identify a real circuit-tracer transcoder release")
    model_spec = (cfg.experiment or {}).get("model") or {}
    # circuit-tracer 0.5.2 ships transcoders for Gemma/Llama only; a smoke profile
    # on Pythia must override circuits.model_identifier with a supported model.
    model_name = (
        circuit_cfg.get("model_identifier")
        or model_spec.get("identifier")
        or cfg.model["model"]["name"]
    )
    revision = circuit_cfg.get("model_revision") or model_spec.get("revision")
    device = circuit_cfg.get("device") or cfg.model["model"].get("device", "cuda")
    compression = cfg.quant["compression"]

    def factory(label: str, bits: int | None) -> TracerModelState:
        model = ReplacementModel.from_pretrained(
            model_name, transcoder_set=transcoder_set,
            backend=circuit_cfg.get("backend", "transformerlens"), device=device,
            dtype=torch.float16, revision=revision,
        )
        if bits is not None:
            # Only the LM's own projection weights are quantized; lazily-loaded
            # transcoder decoder weights live on meta tensors and are excluded.
            state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                     if not getattr(parameter, "is_meta", False)}
            quantized = quantize_state_dict(
                state, bits, compression["quant_modules"], compression["skip_modules"],
            )
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in quantized:
                        parameter.copy_(quantized[name].to(parameter.device, dtype=parameter.dtype))
        state = TracerModelState(label=label, state_id=_state_id(label, model), model=model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return state

    return run_circus_sweep(
        cfg, prompts=prompts, seed=seed, output_dir=output_dir, model_factory=factory,
        tracer=attribute_fn, device=device, mode="harmonized",
    )
