"""QDM (Quantization Damage Measurement) pipeline — SAE arm.

Matches Duan, E. (2026). *Perplexity Can Miss SAE Feature Damage Under
Quantization.* arXiv:2606.03002, the "QDM" pipeline (reuse verbatim): a *frozen* pretrained SAE is used as a fixed measurement basis; full
precision (FP16) and RTN-quantized activations are encoded through the *same* SAE
on identical tokens; per-feature Pearson correlation ``c_j`` between FP16 and
compressed activations is the stability score.

Stack (do not reinvent):
- TransformerLens for activation extraction.
- SAELens against Gemma Scope layer-12 checkpoints for SAE loading.
- RTN from ``src/compression/rtn.py``.
- Streaming correlation from ``src/sae/correlation.py`` (2606.03002 App. I).

The heavy imports (transformer_lens, sae_lens, datasets) are made lazily inside
functions so the pure logic modules stay importable/testable without them.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..compression.rtn import quantize_state_dict
from .correlation import StreamingPearson
from .gates import assert_sae_int6_gate
from .metadata import build_run_metadata, write_metadata
from .perplexity import write_perplexity_csv
from .protocol import (
    ExperimentMode,
    HarmonizedProvenance,
    ReferenceProvenance,
    RunProvenance,
    artifact_directory,
    validate_protocol,
)
from .survival import survival_rates


@dataclass
class QDMResult:
    """Output of one bit-width QDM run.

    Attributes:
        bits: bit-width that produced the quantized activations (16 = FP16 ref).
        survived_pct / degraded_pct / damaged_pct: percentages over active features.
        mean_corr: mean per-feature correlation over active features.
        n_active: number of active features (f_j > active_firing_rate_min).
        correlations: per-feature c_j (d_sae, may contain NaN). Stored separately
            to avoid shipping large arrays in the summary CSV.
    """

    bits: int
    survived_pct: float
    degraded_pct: float
    damaged_pct: float
    mean_corr: float
    n_active: int
    correlations: np.ndarray | None = None


@dataclass
class SAEDirectionalStats:
    """Streaming decoder-direction signal/perturbation sufficient statistics."""

    signal_sum: np.ndarray
    perturbation_sum: np.ndarray
    n_tokens: int = 0

    def update(self, fp16_residual: np.ndarray, quant_residual: np.ndarray, directions: np.ndarray) -> None:
        if fp16_residual.shape != quant_residual.shape:
            raise ValueError("SAE directional statistics require aligned residual activations")
        self.signal_sum += np.abs(fp16_residual @ directions.T).sum(axis=0)
        self.perturbation_sum += np.abs((quant_residual - fp16_residual) @ directions.T).sum(axis=0)
        self.n_tokens += fp16_residual.shape[0]

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_tokens <= 0:
            raise RuntimeError("cannot finalize empty SAE directional statistics")
        return self.signal_sum / self.n_tokens, self.perturbation_sum / self.n_tokens


def _gpu_memory_snapshot(device: str) -> dict[str, float | int | str | None]:
    """Peak/allocated/reserved GPU memory for one stage.

    CPU-only runs record None values rather than fabricating GPU numbers.
    """
    import torch  # lazy heavy dependency

    if device.startswith("cuda") and torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_bytes": float(torch.cuda.memory_allocated()),
            "reserved_bytes": float(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        }
    return {"device": device, "allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}


def _reset_gpu_peak(device: str) -> None:
    import torch  # lazy heavy dependency

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def estimate_model_state_gb(n_parameters: int, *, bytes_per_param: int = 2) -> float:
    """Approximate FP16 model-state footprint in GiB."""
    return n_parameters * bytes_per_param / (1024 ** 3)


def assert_memory_feasible(cfg: object, *, n_copies_on_gpu: int = 1, sae_params: int = 0) -> None:
    """Fail before OOM when the configured experiment cannot fit on the selected GPU.

    The estimate is approximate (weights + SAE + a 2 GiB activation buffer) and is
    only a guard against clearly impossible configurations. Override with
    ``experiment.memory.allow_impossible: true`` when the estimate is wrong.
    """
    import torch  # lazy heavy dependency

    experiment = getattr(cfg, "experiment", {}) or {}
    memory_cfg = experiment.get("memory") or {}
    device = str(getattr(cfg, "model", {}).get("model", {}).get("device", "cpu"))
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"config requests device {device!r} but CUDA is unavailable on this machine; "
            "run on the CUDA experiment machine or set model.device: cpu for tiny smoke profiles"
        )
    n_params = None
    # Prefer the live model's parameter count when the caller passes one via config.
    model_cfg = getattr(cfg, "model", {}).get("model", {})
    n_params = memory_cfg.get("model_parameters") or model_cfg.get("n_parameters")
    if n_params is None:
        return  # no estimate possible without a parameter count; do not fabricate one
    total_gb = estimate_model_state_gb(int(n_params)) * n_copies_on_gpu
    total_gb += estimate_model_state_gb(int(sae_params)) if sae_params else 0.0
    total_gb += 2.0  # activation/logits working buffer, deliberately approximate
    free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    if total_gb > free_gb and not bool(memory_cfg.get("allow_impossible", False)):
        raise RuntimeError(
            f"approximate memory estimate {total_gb:.1f} GiB exceeds free GPU memory "
            f"{free_gb:.1f} GiB; reduce resident model copies or set "
            "experiment.memory.allow_impossible: true to override this approximate check"
        )


def _corpus_texts(corpus: str | Mapping[str, object], n_tokens: int, seed: int) -> list[str]:
    """Load a text corpus, returning a bounded list of documents.

    Uses ``datasets`` for ``wikitext-2-raw`` (behavioral) or ``fineweb-edu-subset``
    (probe/SAE activations). Corpus selection is config-driven.

    Only enough documents to cover ``n_tokens`` are materialized. FineWeb-Edu is
    far larger than memory, so it is read in streaming mode and truncated; the
    caller's token budget (``perplexity.n_tokens``) bounds the pull rather than
    the split size. ``seed`` is accepted so every stochastic step takes an
    explicit seed even where the current selection is a
    deterministic prefix.
    """
    from datasets import load_dataset  # lazy heavy dependency

    # Reproduction corpus is WikiText-2-raw train (Duan 2606.03002v2 App. K
    # Table K.2); FineWeb-Edu below is the non-reproduction default.
    # //8 (~8 tokens/doc) covers WikiText-2-raw's short lines without starving
    # the token budget; the streamer still stops at exactly max_tokens.
    max_docs = max(1, n_tokens // 8)

    if isinstance(corpus, Mapping):
        corpus_name = str(corpus.get("identifier") or corpus.get("id") or "")
        config = corpus.get("config")
        split = corpus.get("split")
        revision = corpus.get("revision")
        if not corpus_name or not isinstance(split, str) or not split:
            raise ValueError("configured corpus requires non-empty identifier and split")
        kwargs: dict[str, object] = {"split": split}
        if config is not None:
            kwargs["name"] = config
        if revision is not None:
            kwargs["revision"] = revision
        # The configured data source, not a friendly alias, controls execution.
        kwargs["streaming"] = bool(corpus.get("streaming", corpus_name == "HuggingFaceFW/fineweb-edu"))
        print(f"[SAE] Loading corpus {corpus_name} (config={config}, split={split}, streaming={kwargs['streaming']})...")
        # Some datasets versions forward unknown kwargs into BuilderConfig and
        # raise (datasets 5.x: **config_kwargs), so pass timeout only when the
        # installed signature declares an explicit named parameter for it.
        try:
            accepts_timeout = any(
                param.name == "timeout" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
                for param in inspect.signature(load_dataset).parameters.values()
            )
        except (TypeError, ValueError):
            accepts_timeout = False
        if accepts_timeout:
            kwargs["timeout"] = 300
        try:
            ds = load_dataset(corpus_name, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"failed to load corpus {corpus_name}/{config}@{split} ({error.__class__.__name__}: {error}). "
                "Check network/proxy access to huggingface.co; if the split is already cached "
                "locally, retry with HF_DATASETS_OFFLINE=1. Do NOT unpin corpus_revision to "
                "work around this."
            ) from error
        print("[SAE] Corpus loaded.")
    elif corpus == "wikitext-2-raw":
        corpus_name = corpus
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif corpus == "fineweb-edu-subset":
        corpus_name = corpus
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    else:
        raise ValueError(f"unknown corpus {corpus!r}")

    texts: list[str] = []
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        texts.append(text)
        if len(texts) >= max_docs:
            break
    if not texts:
        raise ValueError(f"corpus '{corpus_name}' yielded no non-empty documents")
    return texts


def _drop_bos(acts, exclude_bos: bool):
    """Drop position 0 (the prepended BOS) from ``[batch, seq, d_model]`` acts.

    Gemma-2's BOS is a massive attention-sink outlier: it makes thousands of SAE
    features fire (inflating the active-feature count) and, being large and
    quantization-robust, it dominates the per-feature Pearson correlation
    (inflating survival). Excluding it from the *feature analysis* — while keeping
    it in the forward pass so attention is correct — is standard for Gemma SAE
    work and is required to reproduce Duan 2606.03002's active/survival counts.
    The BOS must stay in the model input (it provides the attention sink); only
    its activation row is removed here.
    """
    if not exclude_bos:
        return acts
    return acts[:, 1:, :]


def _encode_batch(
    model, sae, tokens, hook: str, *, exclude_bos: bool = False, encode_float32: bool = False
) -> np.ndarray:
    """Run one token batch through ``model`` and encode ``hook`` acts via the frozen SAE.

    Returns ``[batch*seq, d_sae]`` feature activations as float64. The encoder is
    frozen and applied to *identical* tokens for the FP16 and quantized models so
    the resulting correlation is between aligned activations. With ``exclude_bos``
    the BOS position (0) is dropped from the analysed activations (see ``_drop_bos``).
    ``encode_float32`` casts the residual acts to float32 before ``sae.encode``,
    matching Duan's reference (`sae.encode(acts.float())`); default False keeps the
    prior (native-dtype) behavior byte-identical for existing configs.
    """
    import torch  # lazy heavy dependency

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook])
        acts = _drop_bos(cache[hook], exclude_bos)  # [batch, seq(-1), d_model]
        flat = acts.reshape(-1, acts.shape[-1])
        if encode_float32:
            flat = flat.float()
        encoded = sae.encode(flat)  # [batch*seq, d_sae]
    return encoded.detach().float().cpu().numpy().astype(np.float64)


def _encode_batch_with_residual(
    model, sae, tokens, hook: str, *, exclude_bos: bool = False, encode_float32: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Encode and return aligned flattened residual activations for decoder SPR.

    ``encode_float32`` matches Duan's ``sae.encode(acts.float())`` (default False
    preserves the prior native-dtype behavior).
    """
    import torch

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook])
        acts = _drop_bos(cache[hook], exclude_bos)
        flat = acts.reshape(-1, acts.shape[-1])
        if encode_float32:
            flat = flat.float()
        encoded = sae.encode(flat)
    return (
        encoded.detach().float().cpu().numpy().astype(np.float64),
        flat.detach().float().cpu().numpy().astype(np.float64),
    )


def _bos_token_id(model) -> int:
    """Resolve the tokenizer BOS id (Gemma-2 uses 2)."""
    bos = getattr(getattr(model, "tokenizer", None), "bos_token_id", None)
    if bos is not None:
        return int(bos)
    # Fall back to whatever prepend_bos prepends to an empty string.
    row = model.to_tokens("", prepend_bos=True).reshape(-1)
    if row.numel() == 0:
        raise ValueError("cannot resolve a BOS token id for concatenate packing")
    return int(row[0])


def pack_concatenated_blocks(model, texts, block_len: int, *, bos_policy: str):
    """Duan 2606.03002 App. K tokenization: concatenate non-empty lines, chunk into L.

    The paper's QDM corpus is "WikiText-2-raw. Non-empty lines are concatenated
    and tokenized" into fixed length-L blocks (Gemma L=256). Each source line is
    tokenized WITHOUT a per-line BOS, all ids are concatenated into one flat
    stream, and the stream is split into ``block_len``-token blocks.

    ``bos_policy``:
      - ``prepend_each_block``: every block is ``[BOS] + (block_len - 1)`` content
        tokens, so each independently-encoded block has a valid position 0 (the
        default and the sane choice when blocks are fed to the model separately).
      - ``flat_stream``: blocks are pure ``block_len``-token slices of the stream,
        with no injected BOS (one continuous document view).

    Returns a list of 2-D LongTensors shaped ``[1, block_len]`` ready for
    ``run_with_cache`` — the SAE arm consumes these exactly like tokenized
    strings, and the FP16 and quantized passes reuse the identical block list so
    their features are perfectly aligned by construction.
    """
    import torch

    if block_len < 2:
        raise ValueError(f"block_len must be >= 2, got {block_len}")
    if bos_policy not in ("prepend_each_block", "flat_stream"):
        raise ValueError(
            f"concatenate packing supports bos_policy prepend_each_block|flat_stream, got {bos_policy!r}"
        )
    id_rows: list = []
    for text in texts:
        row = model.to_tokens(text, prepend_bos=False, truncate=False).reshape(-1)
        if row.numel():
            id_rows.append(row)
    if not id_rows:
        raise RuntimeError("concatenate packing produced no tokens from the activation corpus")
    stream = torch.cat(id_rows)
    if bos_policy == "flat_stream":
        step = block_len
        prefix = None
    else:
        step = block_len - 1
        prefix = torch.tensor([_bos_token_id(model)], dtype=stream.dtype, device=stream.device)
    blocks: list = []
    for start in range(0, int(stream.numel()), step):
        content = stream[start : start + step]
        if content.numel() == 0:
            break
        block = content if prefix is None else torch.cat([prefix, content])
        blocks.append(block.unsqueeze(0))
    return blocks


def pack_duan_exact_blocks(model, texts, block_len: int, *, bos_policy: str, max_tokens: int | None = None):
    """EXACT reproduction of Duan 2606.03002's ``build_tokens_2d`` (repo script 03).

    The reference repo (``sae-feature-survival-quantization-0141``,
    ``scripts/03_gemma_phase3_streaming_sweep.py``) builds its Gemma token corpus as::

        full_text  = "\\n\\n".join(x for x in ds["text"] if x.strip())
        token_ids  = tokenizer.encode(full_text, add_special_tokens=False)  # NO BOS
        tokens_2d  = tokens[:n_seqs*seq_len].reshape(n_seqs, seq_len)

    i.e. all non-empty lines are joined with a ``"\\n\\n"`` separator into ONE
    string, tokenized in a SINGLE call with no special tokens, then reshaped into
    full ``block_len`` blocks (the final partial block is dropped). This differs
    from :func:`pack_concatenated_blocks`, which tokenizes each line separately
    and concatenates the id rows with no separator — a different token stream and
    a different active-feature count. This function exists to reproduce the
    paper's numbers byte-for-byte on the token side.

    ``bos_policy`` (the isolation toggle, orthogonal to the tokenization protocol):
      - ``flat_stream``: pure ``block_len``-token slices, no BOS injected — the
        paper's exact protocol. ``exclude_bos`` MUST be False downstream (there is
        no BOS row to drop; dropping position 0 would delete a real content token).
      - ``prepend_each_block``: ``[BOS] + (block_len - 1)`` content tokens per
        block, to isolate the BOS-in-forward-pass lever against the *identical*
        Duan token stream. ``exclude_bos`` should be True downstream.

    ``max_tokens`` caps the content tokens consumed (mirrors the repo's
    ``token_budget``); with the paper's 500_000 / L=256 this yields exactly 1953
    blocks = 499,968 tokens. Returns a list of ``[1, block_len]`` LongTensors, the
    same shape :func:`pack_concatenated_blocks` returns, so the downstream
    FP16/quantized passes consume it unchanged.
    """
    import torch

    if block_len < 2:
        raise ValueError(f"block_len must be >= 2, got {block_len}")
    if bos_policy not in ("prepend_each_block", "flat_stream"):
        raise ValueError(
            f"duan_exact packing supports bos_policy prepend_each_block|flat_stream, got {bos_policy!r}"
        )
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise RuntimeError("duan_exact packing requires model.tokenizer.encode (HF tokenizer)")
    # Their exact separator + single-call tokenization with no special tokens.
    full_text = "\n\n".join(t for t in texts if isinstance(t, str) and t.strip())
    if not full_text:
        raise RuntimeError("duan_exact packing produced no non-empty text from the activation corpus")
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    stream = torch.tensor(ids, dtype=torch.long)
    n_avail = int(stream.numel())
    if n_avail == 0:
        raise RuntimeError("duan_exact packing tokenized to an empty stream")

    if bos_policy == "flat_stream":
        budget = n_avail if max_tokens is None else min(int(max_tokens), n_avail)
        n_blocks = budget // block_len
        if n_blocks == 0:
            raise RuntimeError(f"duan_exact stream ({n_avail} tokens) too short for one L={block_len} block")
        return [stream[i * block_len : (i + 1) * block_len].unsqueeze(0) for i in range(n_blocks)]

    # prepend_each_block: BOS + (block_len - 1) content tokens per block.
    step = block_len - 1
    budget = n_avail if max_tokens is None else min(int(max_tokens), n_avail)
    n_blocks = budget // step
    if n_blocks == 0:
        raise RuntimeError(f"duan_exact stream ({n_avail} tokens) too short for one L={block_len} block")
    prefix = torch.tensor([_bos_token_id(model)], dtype=stream.dtype)
    return [
        torch.cat([prefix, stream[i * step : (i + 1) * step]]).unsqueeze(0)
        for i in range(n_blocks)
    ]


def _prepare_sequence_tokens(model, item, seq_len: int):
    """Return model-ready tokens for a corpus item (pre-packed tensor or raw string).

    Packed-concatenate mode passes pre-tokenized ``[1, L]`` blocks; per-document
    mode passes strings tokenized with a prepended BOS. Both are truncated to
    ``seq_len`` so the two modes share one downstream extraction path.
    """
    import torch

    if isinstance(item, torch.Tensor):
        tokens = item if item.ndim == 2 else item.reshape(1, -1)
        return tokens[:, :seq_len]
    return model.to_tokens(item, prepend_bos=True, truncate=True)[:, :seq_len]


def _stream_qdm_statistics(
    fp16_model,
    quant_model,
    sae,
    texts: list[str],
    hook: str,
    max_tokens: int,
    seq_len: int,
    decoder_directions: np.ndarray | None = None,
    single_active_gpu_model: bool = True,
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, int, tuple[np.ndarray, np.ndarray]]:
    """Single streaming pass computing per-feature correlations and FP16 firing rates.

    Both models see the *same* tokens in the same order. Sufficient statistics are
    folded in per batch (2606.03002 App. I) so peak memory is ``O(d_sae)``, never
    ``O(N * d_sae)`` — materializing 500k tokens x 16,384 features would be ~65GB
    and is explicitly ruled out by the memory budget.

    Firing rates are accumulated from the FP16 features only, since the active
    gate ``f_j > 0.001`` is defined on FP16 (2606.03002 §2.2).

    Returns:
        ``(correlations, fp16_firing_rates, n_tokens)``.
    """
    sae.eval()
    est: StreamingPearson | None = None
    fire_counts: np.ndarray | None = None
    produced = 0
    directional: SAEDirectionalStats | None = None

    for text in texts:
        if produced >= max_tokens:
            break
        tokens = fp16_model.to_tokens(text, prepend_bos=True, truncate=True)[:, :seq_len]
        if tokens.numel() == 0:
            continue

        if decoder_directions is None:
            f_fp16 = _encode_batch(fp16_model, sae, tokens, hook)
            f_quant = _encode_batch(quant_model, sae, tokens, hook)
            h_fp16 = h_quant = None
        else:
            f_fp16, h_fp16 = _encode_batch_with_residual(fp16_model, sae, tokens, hook)
            f_quant, h_quant = _encode_batch_with_residual(quant_model, sae, tokens, hook)

        # Never exceed the token budget: trim the tail of the final batch.
        room = max_tokens - produced
        if f_fp16.shape[0] > room:
            f_fp16 = f_fp16[:room]
            f_quant = f_quant[:room]
            if h_fp16 is not None and h_quant is not None:
                h_fp16, h_quant = h_fp16[:room], h_quant[:room]

        if est is None:
            est = StreamingPearson(d_sae=f_fp16.shape[1])
            fire_counts = np.zeros(f_fp16.shape[1], dtype=np.float64)
        est.update(f_fp16, f_quant)
        fire_counts += (f_fp16 > 0).sum(axis=0)
        if decoder_directions is not None and h_fp16 is not None and h_quant is not None:
            if directional is None:
                directional = SAEDirectionalStats(
                    np.zeros(f_fp16.shape[1], dtype=np.float64),
                    np.zeros(f_fp16.shape[1], dtype=np.float64),
                )
            directional.update(h_fp16, h_quant, decoder_directions)
        produced += f_fp16.shape[0]

    if est is None or fire_counts is None or produced == 0:
        raise RuntimeError(
            "QDM produced no activations — check the corpus, seq_len, and token budget"
        )
    base = (est.finalize(), fire_counts / produced, produced)
    return (*base, directional.finalize()) if decoder_directions is not None and directional is not None else base


def _stream_qdm_statistics_single_model(
    model,
    fp16_state: dict,
    quant_state: dict,
    sae,
    texts: list[str],
    hook: str,
    max_tokens: int,
    seq_len: int,
    decoder_directions: np.ndarray | None = None,
    chunk_texts: int = 16,
    activation_stats: dict[str, np.ndarray] | None = None,
) -> tuple:
    """Memory-safe variant of ``_stream_qdm_statistics``.

    Only ONE model copy is resident on the GPU. The canonical FP16 state and the
    quantized state live on CPU; per chunk of texts the live model is set to FP16,
    the chunk is encoded, then the model is set to the quantized state and the
    chunk is re-encoded. The paired sufficient statistics are therefore identical
    to the two-model interleaved implementation — the same tokens see bit-identical
    weights; only the time ordering changes. FP16 feature activations for one chunk
    are held on CPU and discarded after the fold, so peak memory stays O(chunk),
    never O(N * d_sae).

    Returns the same tuple shape as ``_stream_qdm_statistics``.
    """
    if chunk_texts <= 0:
        raise ValueError(f"chunk_texts must be positive, got {chunk_texts}")
    sae.eval()
    est: StreamingPearson | None = None
    fire_counts: np.ndarray | None = None
    directional: SAEDirectionalStats | None = None
    produced = 0

    for start in range(0, len(texts), chunk_texts):
        if produced >= max_tokens:
            break
        chunk = texts[start : start + chunk_texts]
        fp16_features: list[np.ndarray] = []
        quant_features: list[np.ndarray] = []
        fp16_residuals: list[np.ndarray] = []
        quant_residuals: list[np.ndarray] = []
        room = max_tokens - produced

        _apply_state_dict(model, fp16_state)
        for text in chunk:
            tokens = model.to_tokens(text, prepend_bos=True, truncate=True)[:, :seq_len]
            if tokens.numel() == 0 or room <= 0:
                continue
            if decoder_directions is None:
                feats = _encode_batch(model, sae, tokens, hook)
                fp16_features.append(feats[:room])
            else:
                feats, resid = _encode_batch_with_residual(model, sae, tokens, hook)
                fp16_features.append(feats[:room])
                fp16_residuals.append(resid[:room])
            room -= min(room, fp16_features[-1].shape[0])

        room = max_tokens - produced
        _apply_state_dict(model, quant_state)
        for text in chunk:
            tokens = model.to_tokens(text, prepend_bos=True, truncate=True)[:, :seq_len]
            if tokens.numel() == 0 or room <= 0:
                continue
            if decoder_directions is None:
                feats = _encode_batch(model, sae, tokens, hook)
                quant_features.append(feats[:room])
            else:
                feats, resid = _encode_batch_with_residual(model, sae, tokens, hook)
                quant_features.append(feats[:room])
                quant_residuals.append(resid[:room])
            room -= min(room, quant_features[-1].shape[0])

        if not fp16_features or not quant_features:
            continue
        f_fp16 = np.concatenate(fp16_features, axis=0)
        f_quant = np.concatenate(quant_features, axis=0)
        if f_fp16.shape != f_quant.shape:
            raise RuntimeError(
                f"chunked encoding misaligned: fp16 {f_fp16.shape} vs quant {f_quant.shape}"
            )
        if est is None:
            est = StreamingPearson(d_sae=f_fp16.shape[1])
            fire_counts = np.zeros(f_fp16.shape[1], dtype=np.float64)
        est.update(f_fp16, f_quant)
        fire_counts += (f_fp16 > 0).sum(axis=0)
        if activation_stats is not None:
            if "sum" not in activation_stats:
                activation_stats["sum"] = np.zeros(f_fp16.shape[1], dtype=np.float64)
                activation_stats["max"] = np.zeros(f_fp16.shape[1], dtype=np.float64)
            activation_stats["sum"] += f_fp16.sum(axis=0)
            activation_stats["max"] = np.maximum(activation_stats["max"], f_fp16.max(axis=0))
        if decoder_directions is not None and fp16_residuals and quant_residuals:
            if directional is None:
                directional = SAEDirectionalStats(
                    np.zeros(f_fp16.shape[1], dtype=np.float64),
                    np.zeros(f_fp16.shape[1], dtype=np.float64),
                )
            directional.update(
                np.concatenate(fp16_residuals, axis=0),
                np.concatenate(quant_residuals, axis=0),
                decoder_directions,
            )
        produced += f_fp16.shape[0]

    if est is None or fire_counts is None or produced == 0:
        raise RuntimeError(
            "QDM produced no activations — check the corpus, seq_len, and token budget"
        )
    base = (est.finalize(), fire_counts / produced, produced)
    return (*base, directional.finalize()) if decoder_directions is not None and directional is not None else base


def _apply_state_dict(model, state_dict) -> None:
    """Copy a (FP16 or RTN-dequantized) state dict into a live model in place."""
    import torch  # lazy heavy dependency

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state_dict[name].to(param.dtype).to(param.device))


def _fp16_feature_cache_pass(
    model,
    fp16_state: dict,
    sae,
    texts: list[str],
    hook: str,
    max_tokens: int,
    seq_len: int,
    *,
    chunk_texts: int,
    cache_dir: Path,
    decoder_directions: np.ndarray | None,
    activation_stats: dict[str, np.ndarray] | None,
    exclude_bos: bool = False,
    encode_float32: bool = False,
) -> dict:
    """Encode the FP16 condition ONCE and cache per-chunk features on disk.

    The original single-model design re-uploaded the full ~5 GB state dict to
    the GPU for every 16-text chunk (two PCIe round-trips per chunk per
    bit-width — ~24 TB per condition), which made a 500k-token sweep take
    ~7 h per bit-width on an RTX 3070. The FP16 encode is identical for every
    bit-width, so it runs once here; each quantized condition then needs a
    single state-dict load and one encode pass, reading the cached FP16
    features for the paired sufficient statistics (Duan 2606.03002 App. I).

    Returns the in-memory meta dict consumed by ``_qdm_statistics_from_cache``.
    """
    import json

    if chunk_texts <= 0:
        raise ValueError(f"chunk_texts must be positive, got {chunk_texts}")
    import shutil

    # A killed mid-sweep run leaves orphaned chunk files behind; rebuild fresh.
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _apply_state_dict(model, fp16_state)
    sae.eval()

    total_texts = len(texts)
    n_chunks_est = (min(total_texts, max(1, max_tokens // 8)) + chunk_texts - 1) // chunk_texts
    print(f"[SAE] FP16 feature cache pass: {total_texts} texts, budget={max_tokens} tokens, ~{n_chunks_est} chunks")

    sx = sx2 = fire_counts = signal_sum = None
    produced = 0
    last_reported = 0
    chunks: list[dict] = []
    index = 0
    for start in range(0, len(texts), chunk_texts):
        if produced >= max_tokens:
            break
        if index % 50 == 0 or produced - last_reported >= 50_000:
            print(f"[SAE]   FP16 cache: chunk {index}, {produced}/{max_tokens} tokens ({100*produced/max(max_tokens,1):.1f}%)")
            last_reported = produced
        room = max_tokens - produced
        rows_per_text: list[int] = []
        xs: list[np.ndarray] = []
        hs: list[np.ndarray] = []
        for text in texts[start : start + chunk_texts]:
            tokens = _prepare_sequence_tokens(model, text, seq_len)
            if tokens.numel() == 0 or room <= 0:
                rows_per_text.append(0)
                continue
            if decoder_directions is None:
                x = _encode_batch(model, sae, tokens, hook, exclude_bos=exclude_bos, encode_float32=encode_float32)
            else:
                x, h = _encode_batch_with_residual(
                    model, sae, tokens, hook, exclude_bos=exclude_bos, encode_float32=encode_float32
                )
                hs.append(h[: min(room, x.shape[0])])
            take = min(room, x.shape[0])
            xs.append(x[:take])
            rows_per_text.append(take)
            room -= take
        if not xs:
            index += 1
            continue
        x_chunk = np.concatenate(xs, axis=0)
        np.save(cache_dir / f"chunk_{index:06d}_x.npy", x_chunk.astype(np.float32))
        if decoder_directions is not None:
            h_chunk = np.concatenate(hs, axis=0)
            np.save(cache_dir / f"chunk_{index:06d}_h.npy", h_chunk.astype(np.float32))
            if signal_sum is None:
                signal_sum = np.zeros(x_chunk.shape[1], dtype=np.float64)
            signal_sum += np.abs(h_chunk @ decoder_directions.T).sum(axis=0)
        if sx is None:
            sx = np.zeros(x_chunk.shape[1], dtype=np.float64)
            sx2 = np.zeros(x_chunk.shape[1], dtype=np.float64)
            fire_counts = np.zeros(x_chunk.shape[1], dtype=np.float64)
        sx += x_chunk.sum(axis=0)
        sx2 += np.einsum("ij,ij->j", x_chunk, x_chunk)
        fire_counts += (x_chunk > 0).sum(axis=0)
        if activation_stats is not None:
            if "sum" not in activation_stats:
                activation_stats["sum"] = np.zeros(x_chunk.shape[1], dtype=np.float64)
                activation_stats["max"] = np.zeros(x_chunk.shape[1], dtype=np.float64)
            activation_stats["sum"] += x_chunk.sum(axis=0)
            activation_stats["max"] = np.maximum(activation_stats["max"], x_chunk.max(axis=0))
        produced += x_chunk.shape[0]
        chunks.append({"index": index, "start": start, "rows": rows_per_text})
        index += 1

    if produced == 0 or sx is None or fire_counts is None:
        raise RuntimeError(
            "QDM produced no activations — check the corpus, seq_len, and token budget"
        )
    meta = {
        "d_sae": int(sx.shape[0]),
        "produced": int(produced),
        "chunks": chunks,
        "has_residuals": decoder_directions is not None,
        "sx": sx,
        "sx2": sx2,
        "fire_counts": fire_counts,
        "signal_sum": signal_sum,
        "firing_rates": fire_counts / produced,
    }
    (cache_dir / "cache_meta.json").write_text(
        json.dumps({"produced": int(produced), "chunks": chunks,
                    "has_residuals": decoder_directions is not None, "d_sae": int(sx.shape[0])}),
        encoding="utf-8",
    )
    print(f"[SAE] FP16 cache complete: {produced} tokens across {len(chunks)} chunks, d_sae={sx.shape[0]}")
    return meta


def _qdm_statistics_from_cache(
    model,
    quant_state: dict,
    sae,
    texts: list[str],
    hook: str,
    seq_len: int,
    *,
    cache_dir: Path,
    meta: dict,
    decoder_directions: np.ndarray | None,
    exclude_bos: bool = False,
    encode_float32: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, tuple[np.ndarray, np.ndarray] | None]:
    """One encode pass per quantized condition against the cached FP16 features.

    Loads the quantized state exactly once, re-encodes each cached chunk, and
    folds the paired sufficient statistics with the FP16 side read from disk.
    Returns the same tuple shape as ``_stream_qdm_statistics``.
    """
    from .correlation import pearson_from_sufficient_statistics

    _apply_state_dict(model, quant_state)
    sae.eval()
    d_sae = meta["d_sae"]
    n_chunks = len(meta["chunks"])
    sy = np.zeros(d_sae, dtype=np.float64)
    sy2 = np.zeros(d_sae, dtype=np.float64)
    sxy = np.zeros(d_sae, dtype=np.float64)
    perturbation_sum = np.zeros(d_sae, dtype=np.float64) if decoder_directions is not None else None
    produced = 0

    for ci, chunk in enumerate(meta["chunks"]):
        if ci % 50 == 0:
            print(f"[SAE]   quant cache: chunk {ci}/{n_chunks}, {produced}/{meta['produced']} tokens")
        start = chunk["start"]
        rows_per_text = chunk["rows"]
        x = np.load(cache_dir / f"chunk_{chunk['index']:06d}_x.npy").astype(np.float64)
        h_fp16 = (
            np.load(cache_dir / f"chunk_{chunk['index']:06d}_h.npy").astype(np.float64)
            if meta["has_residuals"] else None
        )
        ys: list[np.ndarray] = []
        hqs: list[np.ndarray] = []
        for text, want in zip(texts[start : start + len(rows_per_text)], rows_per_text):
            if want == 0:
                continue
            tokens = _prepare_sequence_tokens(model, text, seq_len)
            if tokens.numel() == 0:
                raise RuntimeError("quantized pass tokenization disagrees with the FP16 cache")
            if decoder_directions is None:
                y = _encode_batch(model, sae, tokens, hook, exclude_bos=exclude_bos, encode_float32=encode_float32)
            else:
                y, hq = _encode_batch_with_residual(
                    model, sae, tokens, hook, exclude_bos=exclude_bos, encode_float32=encode_float32
                )
                hqs.append(hq[:want])
            if y.shape[0] < want:
                raise RuntimeError("quantized pass tokenization disagrees with the FP16 cache")
            ys.append(y[:want])
        y_chunk = np.concatenate(ys, axis=0)
        if y_chunk.shape != x.shape:
            raise RuntimeError(
                f"cache alignment broken: quant {y_chunk.shape} vs fp16 {x.shape}"
            )
        sy += y_chunk.sum(axis=0)
        sy2 += np.einsum("ij,ij->j", y_chunk, y_chunk)
        sxy += np.einsum("ij,ij->j", x, y_chunk)
        if decoder_directions is not None and h_fp16 is not None:
            h_quant = np.concatenate(hqs, axis=0)
            perturbation_sum += np.abs(
                (h_quant - h_fp16) @ decoder_directions.T
            ).sum(axis=0)
        produced += x.shape[0]

    if produced != meta["produced"]:
        raise RuntimeError(
            f"quantized pass produced {produced} tokens, FP16 cache has {meta['produced']}"
        )
    corr = pearson_from_sufficient_statistics(
        meta["sx"], sy, meta["sx2"], sy2, sxy, produced,
    )
    directional = (
        (meta["signal_sum"] / produced, perturbation_sum / produced)
        if decoder_directions is not None and perturbation_sum is not None else None
    )
    return corr, meta["firing_rates"], produced, directional


def _load_model(
    model_name: str,
    device: str,
    *,
    revision: str | None = None,
    dtype: str = "float16",
    no_processing: bool = False,
):
    """Load one HookedTransformer (TransformerLens).

    ``no_processing`` uses ``from_pretrained_no_processing`` (fold_ln=False,
    center_writing_weights=False, center_unembed=False, fold_value_biases=False).
    This is the reproduction default because Duan 2606.03002 quantizes the *raw*
    HF projection weights, whereas default ``from_pretrained`` folds the RMSNorm
    scales into W_Q/W_K/W_V/W_O/W_in/W_out first — so quantizing them applies a
    different (milder) perturbation than the paper's. It also keeps the residual
    stream closest to the HF model the Gemma Scope SAE was trained on
    (TransformerLens itself advises no-processing under reduced precision).
    """
    import torch  # lazy heavy dependency
    import transformer_lens  # lazy heavy dependency

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs = {"device": device, "dtype": torch_dtype}
    if revision:
        kwargs["revision"] = revision
    loader = (
        transformer_lens.HookedTransformer.from_pretrained_no_processing
        if no_processing
        else transformer_lens.HookedTransformer.from_pretrained
    )
    return loader(model_name, **kwargs)


def _configured_corpus(cfg: object, name: str, provenance: RunProvenance) -> dict[str, object]:
    """Return the actual loader specification, never a legacy corpus nickname."""
    experiment = getattr(cfg, "experiment", {}) or {}
    configured = (experiment.get("corpora") or {}).get(name)
    if not isinstance(configured, Mapping):
        raise ValueError(f"missing executable corpus config corpora.{name}")  # noqa: TRY004 - config-contract violation is deliberately ValueError
    spec = dict(configured)
    if name == "activation":
        # Reference provenance owns the activation corpus identity.  Reconcile
        # rather than silently prefer one metadata block over another.
        if spec.get("identifier") != provenance.corpus_id or spec.get("split") != provenance.corpus_split:
            raise ValueError(
                "activation corpus config disagrees with declared SAE provenance: "
                f"config={spec.get('identifier')}@{spec.get('split')}, "
                f"provenance={provenance.corpus_id}@{provenance.corpus_split}"
            )
        spec["config"] = provenance.corpus_config
        spec["revision"] = provenance.corpus_revision
    return spec


def _load_sae(*, release: str, sae_id: str, revision: str | None, device: str):
    """Load a pinned SAE, enforcing the checkpoint revision across sae_lens versions.

    sae_lens through 6.49.x exposes no ``revision`` argument, so a configured
    pin is enforced by verifying the release repo's HEAD still equals the pinned
    sha — then the unpinned download is byte-identical to the pinned checkpoint.
    A drifted upstream repo is refused loudly instead of loading silently.
    """
    import os

    from sae_lens import SAE  # lazy heavy dependency

    # Bound the params.npz download: a blocked/slow endpoint must surface as a
    # timeout error, not as a silent multi-minute hang after model load.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    kwargs: dict[str, object] = {"release": release, "sae_id": sae_id, "device": device}
    if revision:
        if "revision" in inspect.signature(SAE.from_pretrained).parameters:
            kwargs["revision"] = revision
        else:
            _assert_sae_checkpoint_not_drifted(release, revision)
    loaded = SAE.from_pretrained(**kwargs)
    # Older sae_lens returned (sae, cfg_dict); >=6.49 returns the bare SAE.
    return loaded[0] if isinstance(loaded, tuple) else loaded


def _sae_release_repo_id(release: str) -> str | None:
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    entry = get_pretrained_saes_directory().get(release)
    return getattr(entry, "repo_id", None) if entry is not None else None


def _assert_sae_checkpoint_not_drifted(release: str, revision: str) -> None:
    """Refuse to load unless the SAE repo HEAD still equals the pinned revision."""
    from huggingface_hub import HfApi  # lazy heavy dependency

    repo_id = _sae_release_repo_id(release)
    if not repo_id:
        raise RuntimeError(
            f"cannot enforce SAE revision pin: no HF repo known for release {release!r}"
        )
    try:
        head = HfApi().model_info(repo_id, timeout=30).sha
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"cannot verify SAE checkpoint pin for {repo_id} ({error.__class__.__name__}: {error}). "
            "The pinned sae_revision cannot be enforced without reaching huggingface.co; check "
            "network/proxy access (or set HF_HUB_OFFLINE=1 with a warm cache) and retry. "
            "Do NOT remove the pin to work around this."
        ) from error
    if head != revision:
        raise RuntimeError(
            "SAE checkpoint drift detected: pinned sae_revision "
            f"{revision} != current HEAD {head} of {repo_id}. This sae_lens version "
            "cannot pin download revisions, so continuing would silently load a "
            "different checkpoint. Re-pin sae_revision to the verified HEAD (after "
            "re-validation) or install a revision-aware sae_lens."
        )


def write_resolved_provenance(
    path: Path, *, cfg: object, provenance: RunProvenance, model_id: str, model_revision: str | None
) -> None:
    """Write the loader inputs used for a run; requested/used disagreement is impossible."""
    import importlib.metadata

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    record = {
        "requested": dataclasses.asdict(provenance),
        "resolved": {
            "model": {"identifier": model_id, "revision": model_revision},
            "tokenizer": {"identifier": provenance.tokenizer_id, "revision": provenance.tokenizer_revision},
            "corpus": {"identifier": provenance.corpus_id, "config": provenance.corpus_config, "split": provenance.corpus_split, "revision": provenance.corpus_revision},
            "sae": ({"release": provenance.sae_release, "sae_id": provenance.sae_id, "revision": provenance.sae_revision} if isinstance(provenance, ReferenceProvenance) else None),
        },
        "libraries": {"datasets": version("datasets"), "transformer_lens": version("transformer-lens"), "sae_lens": version("sae-lens")},
        "config_path": str(getattr(cfg, "config_path", "")),
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


_REFERENCE_FIELDS = tuple(field.name for field in dataclasses.fields(ReferenceProvenance))
_HARMONIZED_FIELDS = tuple(field.name for field in dataclasses.fields(HarmonizedProvenance))


def provenance_from_config(cfg: object, mode: ExperimentMode | str) -> RunProvenance:
    """Build the mode-appropriate provenance from the experiment config.

    Fails loudly when the ``provenance`` block is missing, contains unknown
    keys, or leaves reference fields unresolved. Unresolved reproduction
    provenance is a hard stop: a reproduction claim cannot be
    built on guessed corpus/tokenization/checkpoint details.
    """
    normalized = ExperimentMode(mode)
    experiment = getattr(cfg, "experiment", None) or {}
    block = experiment.get("provenance")
    config_path = getattr(cfg, "config_path", None)
    if block is None:
        raise ValueError(
            f"SAE mode {normalized.value!r} requires a 'provenance:' block in the experiment "
            f"config ({config_path or 'load one via Config.from_experiment'}); "
            "see docs/scientific_contract.md for the required fields"
        )
    if not isinstance(block, Mapping):
        raise TypeError(
            f"the 'provenance:' block in {config_path} must be a mapping of provenance fields, "
            f"got {type(block).__name__}"
        )
    if normalized is ExperimentMode.REPRODUCTION:
        fields, cls = _REFERENCE_FIELDS, ReferenceProvenance
    else:
        fields, cls = _HARMONIZED_FIELDS, HarmonizedProvenance
    unknown = sorted(set(block) - set(fields))
    if unknown:
        raise ValueError(
            f"unknown provenance fields {unknown} in {config_path}; expected only {sorted(fields)}"
        )
    return cls(**{name: block.get(name) for name in fields})


def _concatenated_token_ids(model, texts: list[str], max_tokens: int):
    """Tokenize behavioral documents into one flat stream for perplexity.

    Returns a 1-D token tensor covering at most ``max_tokens + 1`` positions so
    every scored target token has a predecessor. Tokenization runs once and the
    identical stream is reused by every model condition.
    """
    import torch  # lazy heavy dependency

    chunks = []
    total = 0
    for text in texts:
        ids = model.to_tokens(text, prepend_bos=False, truncate=False).reshape(-1)
        if ids.numel() == 0:
            continue
        chunks.append(ids)
        total += ids.numel()
        if total >= max_tokens + 1:
            break
    if not chunks:
        raise RuntimeError("behavioral corpus produced no tokens for perplexity")
    return torch.cat(chunks)[: max_tokens + 1]


def run_qdm(
    model_name: str,
    bits: int,
    *,
    sae_release: str,
    sae_id: str,
    hook: str,
    corpus: str,
    max_tokens: int,
    seq_len: int,
    quant_modules: dict[str, Iterable[str]],
    skip_modules: dict[str, Iterable[str]],
    active_firing_rate_min: float,
    survived_corr_min: float,
    degraded_corr_min: float,
    seed: int = 0,
    device: str = "cuda",
) -> QDMResult:
    """Run the full QDM pipeline for one bit-width.

    Bits == 16 means no quantization (FP16 reference). All other bits apply RTN
    (2606.03002 §3.2 Eq. 2) to the quantizable projections and leave the rest
    untouched. Returns survival rates + per-feature correlations.
    """
    from sae_lens import SAE  # lazy heavy dependency

    fp16_model = _load_model(model_name, device)
    loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    # Older sae_lens returned (sae, cfg_dict); >=6.49 returns the bare SAE.
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    fp16_state = {
        name: param.detach().cpu().clone() for name, param in fp16_model.named_parameters()
    }
    texts = _corpus_texts(corpus, max_tokens, seed)

    if bits == 16:
        # FP16 reference row: the model is compared against itself, so every
        # correlation is 1.0 by construction. Only the firing rates (and hence the
        # active-feature count) carry information here.
        _, firing_rates, _ = _stream_qdm_statistics(
            fp16_model, fp16_model, sae, texts, hook, max_tokens, seq_len
        )
        return QDMResult(
            bits=bits,
            survived_pct=100.0,
            degraded_pct=0.0,
            damaged_pct=0.0,
            mean_corr=1.0,
            n_active=int((firing_rates > active_firing_rate_min).sum()),
            correlations=np.ones(firing_rates.shape[0]),
        )

    # Quantized weights go on a SEPARATE model instance; quantizing the FP16
    # model in place would compare it against itself (c_j == 1.0 everywhere).
    quant_model = _load_model(model_name, device)
    _apply_state_dict(
        quant_model, quantize_state_dict(fp16_state, bits, quant_modules, skip_modules)
    )

    corr, firing_rates, _ = _stream_qdm_statistics(
        fp16_model, quant_model, sae, texts, hook, max_tokens, seq_len
    )
    return _result_from_statistics(
        bits, corr, firing_rates, active_firing_rate_min, survived_corr_min, degraded_corr_min
    )


def _result_from_statistics(
    bits: int,
    corr: np.ndarray,
    firing_rates: np.ndarray,
    active_firing_rate_min: float,
    survived_corr_min: float,
    degraded_corr_min: float,
) -> QDMResult:
    """Assemble a QDMResult from streamed correlations and FP16 firing rates."""
    rates = survival_rates(
        corr, firing_rates, active_firing_rate_min, survived_corr_min, degraded_corr_min
    )
    active = firing_rates > active_firing_rate_min
    active_corr = corr[active]
    finite = np.isfinite(active_corr)
    mean_corr = float(active_corr[finite].mean()) if finite.any() else 0.0
    return QDMResult(
        bits=bits,
        survived_pct=rates["survived"],
        degraded_pct=rates["degraded"],
        damaged_pct=rates["damaged"],
        mean_corr=mean_corr,
        n_active=int(active.sum()),
        correlations=corr,
    )


def _sae_decoder_directions(sae) -> np.ndarray:
    """Return unit decoder vectors, the preregistered SAE readout directions."""
    weights = getattr(sae, "W_dec", None)
    if weights is None:
        raise RuntimeError("frozen SAE exposes no W_dec decoder matrix; cannot build directional SAE SPR")
    array = weights.detach().float().cpu().numpy().astype(np.float64)
    if array.ndim != 2:
        raise ValueError(f"SAE W_dec must be [features, d_model], got {array.shape}")
    norms = np.linalg.norm(array, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("SAE decoder directions contain zero or non-finite rows")
    return array / norms[:, None]


def _write_sae_spr_artifact(
    *, output_dir: Path, correlations: Mapping[int, np.ndarray], firing_rates: np.ndarray,
    directional: Mapping[int, tuple[np.ndarray, np.ndarray]], active_threshold: float,
    activation_stats: Mapping[str, np.ndarray] | None = None,
    survived_corr_min: float = 0.9, degraded_corr_min: float = 0.5,
) -> Path:
    """Persist every active feature at every quantized bit width for transfer.

    Columns: feature_id, firing_rate_fp16,
    mean/peak_activation_fp16, correlation, survival_class, signal,
    directional_perturbation, spr — for every configured bit width INT8..INT4,
    never only the gate bit width.
    """
    active = firing_rates > active_threshold
    n_tokens_fp16 = activation_stats.get("n_tokens") if activation_stats else None
    act_sum = activation_stats.get("sum") if activation_stats else None
    act_max = activation_stats.get("max") if activation_stats else None
    rows: list[dict[str, object]] = []
    for bits, corr in sorted(correlations.items()):
        if bits == 16 or bits not in directional:
            continue
        signal, perturbation = directional[bits]
        for feature in np.flatnonzero(active):
            c = float(corr[feature])
            # A degenerate feature (zero variance under quantization) has an
            # undefined Pearson correlation (0/0 -> NaN). That is maximal damage,
            # so pin it to 0.0: survival_class stays "damaged" (NaN already sorted
            # there) and target = 1 - clip(c) becomes finite. np.clip does NOT
            # sanitize NaN, so an unsanitized NaN here breaks the SPR loader.
            if not np.isfinite(c):
                c = 0.0
            survival = "survived" if c > survived_corr_min else "degraded" if c >= degraded_corr_min else "damaged"
            rows.append({
                "bits": bits, "feature": int(feature), "correlation": c,
                "signal": float(signal[feature]), "perturbation": float(perturbation[feature]),
                "firing": float(firing_rates[feature]),
                "mean_act": float(act_sum[feature] / n_tokens_fp16) if act_sum is not None and n_tokens_fp16 else float("nan"),
                "peak_act": float(act_max[feature]) if act_max is not None else float("nan"),
                "survival_class": survival,
            })
    if not rows:
        raise RuntimeError("SAE SPR artifact has no active quantized feature rows")
    directory = output_dir / "harmonized" / "spr" / "instruments"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "sae.npz"
    col = lambda key: np.asarray([row[key] for row in rows])
    signal = col("signal").astype(np.float64)
    perturbation = col("perturbation").astype(np.float64)
    corr = col("correlation").astype(np.float64)
    np.savez_compressed(
        destination, spr=signal / np.maximum(perturbation, 1e-8), target=np.clip(1.0 - corr, 0.0, 1.0),
        bit_width=col("bits").astype(np.int64), groups=col("feature").astype(str),
        signal=signal, perturbation=perturbation,
        target_kind=np.asarray("continuous"), feature_id=col("feature").astype(np.int64),
        raw_target=corr, correlation=corr,
        firing_rate_fp16=col("firing").astype(np.float64),
        mean_activation_fp16=col("mean_act").astype(np.float64),
        peak_activation_fp16=col("peak_act").astype(np.float64),
        survival_class=np.asarray([row["survival_class"] for row in rows]),
        direction_source=np.asarray("sae_decoder_direction_unit_normalized"),
        instrument=np.asarray("sae"),
    )
    return destination


def run_sae_sweep(
    cfg: object,
    bits: Iterable[int],
    *,
    output_dir: Path | None = None,
    seed: int = 0,
    mode: ExperimentMode | str = "reproduction",
) -> dict[int, QDMResult]:
    """Run the QDM sweep across bit-widths and write mode-separated artifacts.

    Artifacts land under ``<output_dir>/<mode>/sae/`` so a
    harmonized run can never be mistaken for a reproduction. Besides
    ``survival_rates.csv`` and the gate correlations, the sweep persists:

    - ``quantization_audit_int<b>.json`` — per-module RTN coverage report;
    - ``perplexity.csv`` — aligned FP16/INT sliding-window PPL;
    - ``metadata.json`` — complete per-condition run metadata.

    The mode/provenance contract is validated before any model loads: a
    reproduction run with unresolved reference provenance fails loudly.
    """
    from ..config import REPO_ROOT

    provenance = provenance_from_config(cfg, mode)
    normalized_mode = validate_protocol(mode, provenance)

    m = cfg.model
    q = cfg.quant["compression"]
    swe = cfg.quant["sae_survival"]
    if output_dir is None:
        output_dir = REPO_ROOT / cfg.quant["results"]["root"]

    device = m["model"]["device"]
    hook = m["sae"]["hook"]
    max_tokens = m["perplexity"]["n_tokens"]
    # Activation block length (Duan 2606.03002 §3.8/App. K/H): Gemma L=256,
    # Pythia L=512. seq_len is the fallback for profiles predating the split.
    seq_len = int(m["perplexity"].get("activation_seq_len", m["perplexity"]["seq_len"]))
    window_size = m["perplexity"]["sliding_window_size"]
    stride = m["perplexity"]["stride"]

    experiment = getattr(cfg, "experiment", {}) or {}
    model_spec = experiment.get("model") or {}
    model_id = str(model_spec.get("identifier") or m["model"]["name"])
    model_revision = model_spec.get("revision") or m["model"].get("revision")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("SAE execution requires configured model revision")
    activation_corpus = _configured_corpus(cfg, "activation", provenance)
    behavioral_corpus = _configured_corpus(cfg, "behavioral", provenance)
    if isinstance(provenance, ReferenceProvenance):
        sae_release, sae_id, sae_revision = provenance.sae_release, provenance.sae_id, provenance.sae_revision
    else:
        sae_release, sae_id, sae_revision = m["sae"]["release"], m["sae"]["sae_id"], None

    # Exactly ONE model copy is resident on the GPU: the canonical FP16 state
    # lives on CPU, each quantized state is derived from it and loaded once per
    # condition. A second resident model would OOM an 8 GB card (~5.2 GB FP16).
    assert_memory_feasible(cfg, n_copies_on_gpu=1)
    model_dtype = str(m["model"].get("dtype", "float16"))
    no_processing = bool(m["model"].get("no_processing", False))
    print(f"[SAE] Loading model {model_id} dtype={model_dtype} no_processing={no_processing} "
          f"(no_processing quantizes RAW HF weights, matching Duan 2606.03002).")
    model = _load_model(
        model_id, device, revision=model_revision, dtype=model_dtype, no_processing=no_processing,
    )
    print(f"[SAE] Loading SAE release={sae_release} sae_id={sae_id}...")
    sae = _load_sae(release=sae_release, sae_id=sae_id, revision=sae_revision, device=device)
    print("[SAE] SAE loaded.")
    decoder_directions = _sae_decoder_directions(sae)
    fp16_state = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    print("[SAE] Loading activation corpus...")
    texts = _corpus_texts(activation_corpus, max_tokens, seed)
    print(f"[SAE] Corpus ready: {len(texts)} documents.")

    # Tokenization protocol (Duan 2606.03002 App. K): ``concatenate`` = lines
    # into a flat stream chunked into L=seq_len blocks; ``duan_exact`` = the
    # repo's build_tokens_2d; ``none`` = per-document. All feed the same block
    # list to the FP16 and quantized passes, keeping features aligned.
    packing = str(getattr(provenance, "sequence_packing", "none") or "none")
    _bos_policy = str(getattr(provenance, "bos_policy", "prepend_each_block"))
    if packing == "concatenate":
        sequences = pack_concatenated_blocks(
            model, texts, seq_len, bos_policy=_bos_policy,
        )
        print(f"[SAE] Packing=concatenate: {len(sequences)} blocks of L={seq_len} "
              f"(bos_policy={_bos_policy}).")
    elif packing == "duan_exact":
        # Duan's build_tokens_2d (repo script 03): "\n\n".join -> encode once ->
        # reshape into L blocks. bos_policy is the isolation toggle.
        sequences = pack_duan_exact_blocks(
            model, texts, seq_len, bos_policy=_bos_policy, max_tokens=max_tokens,
        )
        print(f"[SAE] Packing=duan_exact: {len(sequences)} blocks of L={seq_len} "
              f"(bos_policy={_bos_policy}, max_tokens={max_tokens}).")
    elif packing in ("none", ""):
        sequences = texts
    else:
        raise ValueError(
            f"unsupported sequence_packing {packing!r}; use 'concatenate', 'duan_exact', or 'none'"
        )

    experiment_memory = (experiment.get("memory") or {}) if isinstance(experiment, dict) else {}
    chunk_texts = int(experiment_memory.get("stream_chunk_texts", 16))
    resource_usage: dict[str, object] = {"device": device, "stages": {}}

    results: dict[int, QDMResult] = {}
    correlations: dict[int, np.ndarray] = {}
    directional_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fp16_firing_rates: np.ndarray | None = None
    activation_stats: dict[str, np.ndarray] = {}
    audits: dict[int, object] = {}

    out = artifact_directory(output_dir, normalized_mode)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "_fp16_feature_cache"

    # The FP16 encode is identical for every bit-width, so it runs ONCE and
    # caches per-chunk features on disk; each quantized condition then needs a
    # single state-dict load and one encode pass (vs per-chunk swapping).
    # Exclude the BOS position from the analysis (default on): Gemma-2's BOS is
    # an attention-sink outlier that inflates both active count and survival
    # (Duan 2606.03002). It stays in the forward pass; only its row is dropped.
    exclude_bos = bool(m["sae"].get("exclude_bos", True))
    # flat_stream has no BOS, so position 0 is real content: dropping it would
    # delete data. exclude_bos is only valid when a BOS is actually prepended.
    if exclude_bos and _bos_policy == "flat_stream":
        raise ValueError(
            "exclude_bos=true is invalid with bos_policy=flat_stream: there is no BOS "
            "row to drop, so position 0 is real content. Set exclude_bos:false for "
            "flat_stream (the Duan-exact / no-BOS protocol)."
        )
    # Match Duan's `sae.encode(acts.float())` when requested; default off keeps
    # the native-dtype encode byte-identical for existing configs.
    encode_float32 = bool(m["sae"].get("encode_float32", False))
    print(f"[SAE] exclude_bos={exclude_bos} encode_float32={encode_float32} "
          f"(BOS position dropped from firing/correlation analysis when exclude_bos).")
    fp16_meta = _fp16_feature_cache_pass(
        model, fp16_state, sae, sequences, hook, max_tokens, seq_len,
        chunk_texts=chunk_texts, cache_dir=cache_dir,
        decoder_directions=decoder_directions, activation_stats=activation_stats,
        exclude_bos=exclude_bos, encode_float32=encode_float32,
    )
    activation_stats["n_tokens"] = np.asarray(fp16_meta["produced"], dtype=np.int64)
    fp16_firing_rates = fp16_meta["firing_rates"]
    results[16] = QDMResult(
        bits=16, survived_pct=100.0, degraded_pct=0.0, damaged_pct=0.0, mean_corr=1.0,
        n_active=int((fp16_firing_rates > swe["active_firing_rate_min"]).sum()),
        correlations=np.ones(fp16_firing_rates.shape[0]),
    )
    correlations[16] = results[16].correlations
    resource_usage["stages"]["fp16"] = _gpu_memory_snapshot(device)

    for b in bits:
        if b == 16:
            continue
        _reset_gpu_peak(device)
        print(f"[SAE] INT{b}: quantizing + encoding from cache...")
        # Quantize from the CPU-resident canonical state; the quantized copy is
        # loaded into the single live model once for this condition's pass.
        # quantize_state_dict never mutates its input, so no defensive clone.
        quantized = quantize_state_dict(
            fp16_state, b, q["quant_modules"], q["skip_modules"],
        )
        audits[b] = quantized.audit
        corr, firing_rates, n_tokens, directional = _qdm_statistics_from_cache(
            model, quantized, sae, sequences, hook, seq_len,
            cache_dir=cache_dir, meta=fp16_meta, decoder_directions=decoder_directions,
            exclude_bos=exclude_bos, encode_float32=encode_float32,
        )
        if directional is None:
            raise RuntimeError("SAE directional statistics unexpectedly missing")
        directional_stats[b] = directional
        results[b] = _result_from_statistics(
            b, corr, firing_rates, swe["active_firing_rate_min"],
            swe["survived_corr_min"], swe["degraded_corr_min"],
        )
        correlations[b] = corr
        resource_usage["stages"][f"int{b}"] = _gpu_memory_snapshot(device)
        print(
            f"[SAE] INT{b}: survived={results[b].survived_pct:.2f}% "
            f"degraded={results[b].degraded_pct:.2f}% damaged={results[b].damaged_pct:.2f}% "
            f"(n_active={results[b].n_active}, n_tokens={n_tokens}, seed={seed})"
        )
    # Leave the live model in the canonical FP16 state for the perplexity pass.
    _apply_state_dict(model, fp16_state)
    if not bool(experiment_memory.get("keep_feature_cache", False)):
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    write_resolved_provenance(
        out / "provenance_resolved.json", cfg=cfg, provenance=provenance,
        model_id=model_id, model_revision=model_revision,
    )
    with open(out / "survival_rates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["bits", "survived_pct", "degraded_pct", "damaged_pct", "mean_corr", "n_active"]
        )
        writer.writeheader()
        for b, res in sorted(results.items()):
            writer.writerow(
                {
                    "bits": b,
                    "survived_pct": round(res.survived_pct, 3),
                    "degraded_pct": round(res.degraded_pct, 3),
                    "damaged_pct": round(res.damaged_pct, 3),
                    "mean_corr": round(res.mean_corr, 5),
                    "n_active": res.n_active,
                }
            )

    for b, audit in audits.items():
        audit.write_json(out / f"quantization_audit_int{b}.json")

    if normalized_mode is ExperimentMode.HARMONIZED_R1:
        if fp16_firing_rates is None:
            raise RuntimeError("SAE sweep has no FP16 firing rates for the active-feature gate")
        _write_sae_spr_artifact(
            output_dir=output_dir, correlations=correlations, firing_rates=fp16_firing_rates,
            directional=directional_stats, active_threshold=swe["active_firing_rate_min"],
            activation_stats=activation_stats,
            survived_corr_min=swe["survived_corr_min"], degraded_corr_min=swe["degraded_corr_min"],
        )

    # Per-feature correlations for the gate bit-width. The SPR arm consumes these
    # as the SAE instrument's survival outcome, and `sae_correlations` in config
    # names this artifact.
    gate_bits = cfg.quant["gate_sae_int6"]["bits"]
    if gate_bits in correlations:
        np.savez_compressed(
            out / f"correlations_int{gate_bits}.npz",
            correlations=correlations[gate_bits],
            bits=gate_bits,
            seed=seed,
        )

    _run_aligned_perplexity(
        cfg,
        model=model,
        fp16_state=fp16_state,
        quantized_bits=[b for b in results if b != 16],
        window_size=window_size,
        stride=stride,
        seed=seed,
        out=out,
        behavioral_corpus=behavioral_corpus,
        resource_usage=resource_usage,
    )

    _apply_state_dict(model, fp16_state)
    (out / "resource_usage.json").write_text(
        json.dumps(resource_usage, indent=2, sort_keys=True), encoding="utf-8"
    )

    _write_sweep_metadata(
        cfg,
        mode=normalized_mode,
        provenance=provenance,
        results=results,
        audits=audits,
        window_size=window_size,
        stride=stride,
        seq_len=seq_len,
        seed=seed,
        out=out,
    )

    if normalized_mode is ExperimentMode.REPRODUCTION:
        _write_sae_gate_report_from_sweep(cfg, results, out)
    return results


def _write_sae_gate_report_from_sweep(
    cfg: object, results: dict[int, QDMResult], out: Path
) -> None:
    """Persist the gate report (reproduction_gate.json) beside survival rates."""
    from .gates import sae_gate_report, write_sae_gate_report

    g = cfg.quant["gate_sae_int6"]
    bits = g["bits"]
    res = results.get(bits)
    if res is None:
        return
    report = sae_gate_report(
        {"survived": res.survived_pct, "degraded": res.degraded_pct, "damaged": res.damaged_pct},
        {"survived": g["survived_target"], "degraded": g["degraded_target"], "damaged": g["damaged_target"]},
        tolerance_pp=g["tolerance_pp"],
        bits=bits,
    )
    write_sae_gate_report(report, out / "reproduction_gate.json")


def _run_aligned_perplexity(
    cfg: object,
    *,
    model,
    fp16_state: dict,
    quantized_bits: list[int],
    window_size: int,
    stride: int,
    seed: int,
    out: Path,
    behavioral_corpus: Mapping[str, object],
    resource_usage: dict[str, object] | None = None,
) -> None:
    """Compute the FP16 + INT sliding-window PPL table on an identical token stream.

    Single-active-model design: the one live model is set to the
    FP16 state, scored, then set to each quantized state in turn. The behavioral
    corpus is tokenized once; every condition scores exactly the same target
    tokens, verified explicitly against the FP16 reference (the same invariant
    ``evaluate_aligned_perplexities`` enforces). Quantized weights are computed
    on CPU per bit-width, so the sweep never holds multiple full model states on
    the GPU.
    """
    from .perplexity import sliding_window_perplexity

    m = cfg.model
    q = cfg.quant["compression"]
    ppl_budget = int(m["perplexity"].get("ppl_max_tokens", m["perplexity"]["n_tokens"]))
    texts = _corpus_texts(behavioral_corpus, ppl_budget, seed)
    input_ids = _concatenated_token_ids(model, texts, ppl_budget)

    ppl_results: dict[int, object] = {}
    reference = None
    for b in [16, *quantized_bits]:
        _reset_gpu_peak(str(m["model"].get("device", "cpu")))
        if b == 16:
            _apply_state_dict(model, fp16_state)
        else:
            _apply_state_dict(
                model,
                quantize_state_dict(
                    fp16_state, b, q["quant_modules"], q["skip_modules"],
                ),
            )
        result = sliding_window_perplexity(
            model, input_ids, bits=b, window_size=window_size, stride=stride,
            token_budget=ppl_budget, seed=seed,
        )
        if b == 16:
            reference = result
            if not math.isfinite(reference.ppl) or reference.ppl <= 0:
                raise FloatingPointError(f"invalid FP16 perplexity reference: {reference.ppl}")
        else:
            if (
                result.token_digest != reference.token_digest
                or result.scored_token_indices != reference.scored_token_indices
            ):
                raise RuntimeError(f"INT{b} perplexity tokens are not aligned with FP16")
            result = dataclasses.replace(
                result,
                delta_ppl_percent=100.0 * (result.ppl - reference.ppl) / reference.ppl,
            )
            print(
                f"[SAE] INT{b}: ppl={result.ppl:.3f} "
                f"(delta={result.delta_ppl_percent:+.2f}% vs FP16, n_tokens={result.n_tokens})"
            )
        ppl_results[b] = result
        if resource_usage is not None:
            resource_usage["stages"][f"ppl_{'fp16' if b == 16 else f'int{b}'}"] = _gpu_memory_snapshot(
                str(m["model"].get("device", "cpu"))
            )
    write_perplexity_csv(out / "perplexity.csv", ppl_results)


def _write_sweep_metadata(
    cfg: object,
    *,
    mode: ExperimentMode,
    provenance: RunProvenance,
    results: dict[int, QDMResult],
    audits: dict[int, object],
    window_size: int,
    stride: int,
    seq_len: int,
    seed: int,
    out: Path,
) -> None:
    """Persist complete per-condition run metadata beside the result tables."""
    from ..config import REPO_ROOT

    m = cfg.model
    config_path = getattr(cfg, "config_path", None)
    entries = [
        build_run_metadata(
            mode=mode,
            provenance=provenance,
            model_id=m["model"]["name"],
            model_revision=m["model"].get("revision"),
            sae_release=m["sae"]["release"],
            sae_id=m["sae"]["sae_id"],
            sae_revision=m["sae"].get("revision"),
            hook_name=m["sae"]["hook"],
            token_budget=int(m["perplexity"]["n_tokens"]),
            sequence_length=seq_len,
            window_size=window_size,
            stride=stride,
            bits=b,
            seed=seed,
            quantization_audit=audits[b].to_dict() if b in audits else None,
            device=m["model"]["device"],
            dtype=str(m["model"].get("dtype", "float16")),
            config_path=str(config_path) if config_path else None,
            repository=REPO_ROOT,
        )
        for b in sorted(results)
    ]
    write_metadata(out / "metadata.json", entries)


def assert_gate_1_from_sweep(results: dict[int, QDMResult], cfg: object) -> None:
    """Run gate 1 (SAE INT6) against an in-memory sweep result.

    Fails the pipeline immediately if the reproduction target is not matched.
    The gate's bit-width and targets come from config (``gate_sae_int6``).
    """
    g = cfg.quant["gate_sae_int6"]
    bits = g["bits"]
    if bits not in results:
        raise AssertionError(f"gate 1 requires an INT{bits} SAE run; none present in sweep")
    res = results[bits]
    assert_sae_int6_gate(
        {"survived": res.survived_pct, "degraded": res.degraded_pct, "damaged": res.damaged_pct},
        {"survived": g["survived_target"], "degraded": g["degraded_target"], "damaged": g["damaged_target"]},
        tolerance_pp=g["tolerance_pp"],
    )
