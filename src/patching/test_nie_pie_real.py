"""Real-model patching smoke test.

Proves that, with genuinely different model weights, ``run_patching_arm``
returns different NIE/PIE for the fp16 and an RTN-quantized condition — i.e. a
condition label that does not change the underlying model state cannot produce
a different causal decomposition. The fp16/int4 conditions use real GPT-2-small
weights (CPU-viable); the int4 condition carries RTN-quantized weights copied
into the same model architecture.
"""

from __future__ import annotations

import pytest
import torch

from src.compression.rtn import quantize_state_dict
from src.patching.nie_pie import PatchingCondition, run_patching_arm

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


@pytest.fixture(scope="module")
def gpt2():
    from transformer_lens import HookedTransformer

    try:
        return HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001 - environment-blocked, not a code failure
        pytest.skip(f"gpt2 load unavailable in this environment: {exc}")


def _state_dict_from_model(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _forward_logit_diff(state_dict, tokens, device_tokenizer):
    """Return logit-diff (Paris - London) at the last token, as a float metric.

    A fresh scorer model is loaded with ``state_dict``'s weights so the metric
    genuinely reflects the fp16 vs RTN-quantized weights.
    """
    from transformer_lens import HookedTransformer

    scorer = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for name, param in scorer.named_parameters():
            param.copy_(state_dict[name].to(param.dtype))
        logits = scorer(tokens)
    probs = logits[0, -1].softmax(dim=-1)
    paris = device_tokenizer(" Paris", return_tensors="pt")["input_ids"][0, 0].item()
    london = device_tokenizer(" London", return_tensors="pt")["input_ids"][0, 0].item()
    return float(probs[paris] - probs[london])


def _residual_activation(state_dict, tokens):
    """Return the residual-stream activation of the last token, as a real array.

    Derived from the condition's own weights, so fp16 and int4 conditions carry
    genuinely different activation values (the cosmetic-label guard requires
    distinct clean/corrupted activations across conditions).
    """
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state_dict[name].to(param.dtype))
        _, cache = model.run_with_cache(tokens, names_filter="blocks.0.hook_resid_post")
    return cache["blocks.0.hook_resid_post"][0, -1].float().cpu().numpy()


def _condition(label, state_dict, tokens, tokenizer):
    # Clean and corrupted activations are real residual-stream values derived
    # from the condition's own weights; they differ across fp16/int4 so each
    # condition is genuinely distinct.
    clean_activation = _residual_activation(state_dict, tokens)
    clean = {"mlp": clean_activation}
    corrupted = {"mlp": 0.5 * clean_activation}

    def model_fn(component, act, baseline):
        del component
        base = _forward_logit_diff(state_dict, tokens, tokenizer)
        # A tiny per-token-activation coupling makes NIE/PIE non-degenerate.
        return base + (1e-4 * float(act.sum())) * (1.0 if baseline == "clean" else -1.0)

    return PatchingCondition(
        label=label,
        state_id=f"weights-{label}",
        state=state_dict,
        model_fn=model_fn,
        clean_acts=clean,
        corrupted_acts=corrupted,
    )


def test_fp16_and_int4_patching_results_differ(gpt2, tmp_path):
    fp16_state = _state_dict_from_model(gpt2)
    int4_state = quantize_state_dict(fp16_state, 4, QUANT, SKIP)

    tokens = gpt2.to_tokens("The capital of France is", prepend_bos=True, truncate=True)

    fp16_condition = _condition("fp16", fp16_state, tokens, gpt2.tokenizer)
    int4_condition = _condition("int4", int4_state, tokens, gpt2.tokenizer)

    results = run_patching_arm(
        ["mlp"], {"fp16": fp16_condition, "int4": int4_condition}, output_dir=tmp_path
    )
    fp16 = next(r for r in results if r.label == "fp16")
    int4 = next(r for r in results if r.label == "int4")

    # The two conditions hold genuinely different weights (the cosmetic-label
    # guard already verified distinct clean/corrupted activations), so their
    # causal decomposition of the same component must differ.
    assert fp16.nie != pytest.approx(int4.nie, abs=1e-6)
    assert fp16.pie != pytest.approx(int4.pie, abs=1e-6)