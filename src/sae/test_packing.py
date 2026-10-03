"""Duan 2606.03002 App. K concatenate-and-chunk tokenization contract.

Validates ``pack_concatenated_blocks`` (flat line concatenation into fixed L
blocks) and ``_prepare_sequence_tokens`` (packed-tensor vs raw-string dispatch)
on a deterministic fake model, without any GPU or model download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.sae.qdm import (
    _prepare_sequence_tokens,
    pack_concatenated_blocks,
    pack_duan_exact_blocks,
)

BOS = 2


def _char_ids(text: str) -> list[int]:
    return [(ord(c) % 50) + 3 for c in text]


class _EncodeModel:
    """Char-level model whose tokenizer.encode matches to_tokens content ids.

    Mirrors Duan's ``build_tokens_2d``: ``tokenizer.encode(text, add_special_tokens=False)``
    is a single-call, no-special-token encode of the whole joined string (so the
    "\\n\\n" line separators appear as real tokens in the stream).
    """

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            bos_token_id=BOS,
            encode=lambda text, add_special_tokens=False: _char_ids(text),
        )

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        ids = [BOS] if prepend_bos else []
        ids += _char_ids(text)
        return torch.tensor([ids], dtype=torch.long)


class _FakeModel:
    """Char-level tokenizer: each char -> id (ord % 50 + 3); BOS=2 when requested."""

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(bos_token_id=BOS)

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        ids = [BOS] if prepend_bos else []
        ids += [(ord(c) % 50) + 3 for c in text]
        return torch.tensor([ids], dtype=torch.long)


def _content_ids(model, texts):
    rows = [model.to_tokens(t, prepend_bos=False).reshape(-1) for t in texts]
    return torch.cat([r for r in rows if r.numel()])


def test_prepend_each_block_has_bos_and_exact_length():
    model = _FakeModel()
    texts = ["hello world", "the quick brown fox", "lorem ipsum dolor"]
    block_len = 6
    blocks = pack_concatenated_blocks(model, texts, block_len, bos_policy="prepend_each_block")
    # Every full block is [1, block_len] and starts with BOS.
    for b in blocks[:-1]:
        assert b.shape == (1, block_len)
        assert int(b[0, 0]) == BOS
    # Content (everything after each block's BOS) equals the flat concatenated stream.
    recovered = torch.cat([b[0, 1:] for b in blocks])
    assert torch.equal(recovered, _content_ids(model, texts))


def test_flat_stream_is_pure_slices_no_injected_bos():
    model = _FakeModel()
    texts = ["abcdefgh", "ijklmnop"]
    block_len = 5
    blocks = pack_concatenated_blocks(model, texts, block_len, bos_policy="flat_stream")
    stream = _content_ids(model, texts)
    assert torch.equal(torch.cat([b[0] for b in blocks]), stream)
    for b in blocks[:-1]:
        assert b.shape == (1, block_len)


def test_concatenation_is_token_level_not_per_line_bos():
    """No per-line BOS is injected between concatenated lines (paper protocol)."""
    model = _FakeModel()
    texts = ["aa", "bb"]
    blocks = pack_concatenated_blocks(model, texts, 8, bos_policy="flat_stream")
    flat = torch.cat([b[0] for b in blocks])
    # 4 content tokens total, no interior BOS(2).
    assert flat.numel() == 4
    assert BOS not in flat.tolist()


def test_empty_lines_are_skipped():
    model = _FakeModel()
    blocks = pack_concatenated_blocks(model, ["", "ab", "", "cd"], 4, bos_policy="flat_stream")
    assert torch.equal(torch.cat([b[0] for b in blocks]), _content_ids(model, ["ab", "cd"]))


def test_rejects_bad_arguments():
    model = _FakeModel()
    with pytest.raises(ValueError, match="block_len"):
        pack_concatenated_blocks(model, ["ab"], 1, bos_policy="flat_stream")
    with pytest.raises(ValueError, match="bos_policy"):
        pack_concatenated_blocks(model, ["ab"], 4, bos_policy="bogus")
    with pytest.raises(RuntimeError, match="no tokens"):
        pack_concatenated_blocks(model, ["", ""], 4, bos_policy="flat_stream")


def test_duan_exact_flat_stream_is_join_encode_once_no_bos():
    """flat_stream: pure L-slices of the "\\n\\n"-joined single-encode stream, no BOS,
    partial tail dropped."""
    model = _EncodeModel()
    texts = ["abcd", "efgh"]
    joined = "\n\n".join(texts)          # "abcd\n\nefgh"  -> 10 chars
    stream = _char_ids(joined)
    blocks = pack_duan_exact_blocks(model, texts, block_len=4, bos_policy="flat_stream")
    assert len(blocks) == len(stream) // 4          # 2 blocks (last 2 tokens dropped)
    for i, b in enumerate(blocks):
        assert b.shape == (1, 4)
        assert b[0].tolist() == stream[i * 4 : (i + 1) * 4]
    flat = torch.cat([b[0] for b in blocks]).tolist()
    assert BOS not in flat                          # no BOS injected anywhere


def test_duan_exact_includes_newline_separator_tokens():
    """Unlike pack_concatenated_blocks, the '\\n\\n' join puts separator tokens
    into the stream (this is a real token-stream difference, not cosmetic)."""
    model = _EncodeModel()
    newline_id = (ord("\n") % 50) + 3
    blocks = pack_duan_exact_blocks(model, ["ab", "cd"], block_len=2, bos_policy="flat_stream")
    flat = torch.cat([b[0] for b in blocks]).tolist()
    assert newline_id in flat                        # "\n\n" separators are tokenized
    # pack_concatenated_blocks (per-line, no separator) never contains them.
    concat = pack_concatenated_blocks(model, ["ab", "cd"], 2, bos_policy="flat_stream")
    assert newline_id not in torch.cat([b[0] for b in concat]).tolist()


def test_duan_exact_prepend_each_block_has_bos_and_len():
    """prepend_each_block (isolation): [BOS] + (L-1) content per block, content is
    the identical join-encode stream."""
    model = _EncodeModel()
    texts = ["abcd", "efgh"]
    stream = _char_ids("\n\n".join(texts))           # 10 ids
    step = 3
    blocks = pack_duan_exact_blocks(model, texts, block_len=4, bos_policy="prepend_each_block")
    assert len(blocks) == len(stream) // step        # 3 blocks
    for i, b in enumerate(blocks):
        assert b.shape == (1, 4)
        assert int(b[0, 0]) == BOS
        assert b[0, 1:].tolist() == stream[i * step : (i + 1) * step]


def test_duan_exact_respects_max_tokens():
    model = _EncodeModel()
    blocks = pack_duan_exact_blocks(model, ["abcdefgh", "ijklmnop"], block_len=4,
                                    bos_policy="flat_stream", max_tokens=4)
    assert len(blocks) == 1                          # budget 4 // block_len 4


def test_duan_exact_rejects_bad_arguments():
    model = _EncodeModel()
    with pytest.raises(ValueError, match="block_len"):
        pack_duan_exact_blocks(model, ["ab"], 1, bos_policy="flat_stream")
    with pytest.raises(ValueError, match="bos_policy"):
        pack_duan_exact_blocks(model, ["ab"], 4, bos_policy="bogus")
    with pytest.raises(RuntimeError, match="no non-empty text"):
        pack_duan_exact_blocks(model, ["", "  "], 4, bos_policy="flat_stream")


class _DetModel:
    """Deterministic activations as a pure function of (tokens, scale)."""

    def __init__(self, hook: str, d_model: int = 8) -> None:
        self.hook = hook
        self.d_model = d_model
        self.scale = torch.zeros(1)
        self.tokenizer = SimpleNamespace(bos_token_id=BOS)

    def named_parameters(self):
        return [("scale", self.scale)]

    def to_tokens(self, text, prepend_bos=True, truncate=True):
        ids = [BOS] if prepend_bos else []
        ids += [(ord(c) % 50) + 3 for c in text]
        return torch.tensor([ids], dtype=torch.long)

    def run_with_cache(self, tokens, names_filter=None):
        import numpy as np
        n = tokens.shape[1]
        base = np.linspace(-1, 1, n * self.d_model).reshape(1, n, self.d_model)
        acts = base + float(self.scale.sum()) * 0.1
        return None, {self.hook: torch.tensor(acts, dtype=torch.float32)}


class _ReluSAE:
    def eval(self):
        return self

    def encode(self, flat):
        return torch.relu(flat)

    W_dec = torch.eye(8)


def test_packed_blocks_flow_through_cache_path_aligned(tmp_path):
    """End-to-end: pre-packed [1, L] blocks feed the FP16 cache + quant passes,
    and the two passes stay aligned (same block list, same token count)."""
    from src.sae.qdm import _fp16_feature_cache_pass, _qdm_statistics_from_cache

    hook = "blocks.12.hook_resid_post"
    model = _DetModel(hook)
    sae = _ReluSAE()
    blocks = [torch.arange(6, dtype=torch.long).reshape(1, 6) + 3 * i for i in range(4)]

    meta = _fp16_feature_cache_pass(
        model, {"scale": torch.zeros(1)}, sae, blocks, hook, max_tokens=1000, seq_len=64,
        chunk_texts=2, cache_dir=tmp_path, decoder_directions=None, activation_stats=None,
    )
    _corr, firing, n, _ = _qdm_statistics_from_cache(
        model, {"scale": torch.ones(1)}, sae, blocks, hook, seq_len=64,
        cache_dir=tmp_path, meta=meta, decoder_directions=None,
    )
    assert n == 4 * 6, n                 # 4 blocks x 6 tokens, no re-tokenization drift
    assert firing.shape[0] == 8          # d_model features
    assert meta["produced"] == 24


def test_encode_batch_excludes_bos_position():
    """exclude_bos drops position 0 (the BOS) from the analysed activation rows."""
    from src.sae.qdm import _encode_batch

    hook = "blocks.12.hook_resid_post"
    model = _DetModel(hook)
    sae = _ReluSAE()
    tokens = torch.arange(6, dtype=torch.long).reshape(1, 6)  # [1, 6]
    full = _encode_batch(model, sae, tokens, hook, exclude_bos=False)
    dropped = _encode_batch(model, sae, tokens, hook, exclude_bos=True)
    assert full.shape[0] == 6
    assert dropped.shape[0] == 5                 # BOS row removed
    import numpy as np
    assert np.allclose(dropped, full[1:])        # exactly the non-BOS rows


def test_encode_batch_float32_flag_matches_for_float_inputs():
    """encode_float32 casts residual acts to float32 before sae.encode (Duan's
    sae.encode(acts.float())). For already-float acts it is a no-op; the flag must
    thread through without error and give an identical result."""
    from src.sae.qdm import _encode_batch

    hook = "blocks.12.hook_resid_post"
    model = _DetModel(hook)
    sae = _ReluSAE()
    tokens = torch.arange(6, dtype=torch.long).reshape(1, 6)
    import numpy as np
    native = _encode_batch(model, sae, tokens, hook, encode_float32=False)
    forced = _encode_batch(model, sae, tokens, hook, encode_float32=True)
    assert np.allclose(native, forced)


def test_cache_path_excludes_bos_consistently(tmp_path):
    """With exclude_bos the FP16 cache and quant passes both drop BOS and stay aligned."""
    from src.sae.qdm import _fp16_feature_cache_pass, _qdm_statistics_from_cache

    hook = "blocks.12.hook_resid_post"
    model = _DetModel(hook)
    sae = _ReluSAE()
    blocks = [torch.arange(6, dtype=torch.long).reshape(1, 6) + 3 * i for i in range(4)]
    meta = _fp16_feature_cache_pass(
        model, {"scale": torch.zeros(1)}, sae, blocks, hook, max_tokens=1000, seq_len=64,
        chunk_texts=2, cache_dir=tmp_path, decoder_directions=None, activation_stats=None,
        exclude_bos=True,
    )
    _corr, _firing, n, _ = _qdm_statistics_from_cache(
        model, {"scale": torch.ones(1)}, sae, blocks, hook, seq_len=64,
        cache_dir=tmp_path, meta=meta, decoder_directions=None, exclude_bos=True,
    )
    assert n == 4 * 5, n           # 4 blocks x (6 - 1 BOS) tokens
    assert meta["produced"] == 20


def test_load_model_no_processing_routes_and_resolves_dtype(monkeypatch):
    """no_processing selects from_pretrained_no_processing; dtype string is resolved."""
    import sys
    import types

    calls = {}

    class _FakeHT:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["loader"] = "processed"; calls["kw"] = kw; return "M"

        @staticmethod
        def from_pretrained_no_processing(name, **kw):
            calls["loader"] = "no_processing"; calls["kw"] = kw; return "M"

    monkeypatch.setitem(sys.modules, "transformer_lens", types.SimpleNamespace(HookedTransformer=_FakeHT))
    from src.sae.qdm import _load_model

    _load_model("m", "cpu", no_processing=True, dtype="bfloat16")
    assert calls["loader"] == "no_processing"
    assert calls["kw"]["dtype"] is torch.bfloat16

    _load_model("m", "cpu", no_processing=False, dtype="float16")
    assert calls["loader"] == "processed"
    assert calls["kw"]["dtype"] is torch.float16


def test_prepare_sequence_tokens_dispatches_tensor_and_string():
    model = _FakeModel()
    # Pre-packed tensor: passed through and truncated to seq_len.
    packed = torch.arange(10, dtype=torch.long).reshape(1, 10)
    out = _prepare_sequence_tokens(model, packed, seq_len=6)
    assert out.shape == (1, 6)
    assert torch.equal(out, packed[:, :6])
    # Raw string: tokenized with a prepended BOS.
    out2 = _prepare_sequence_tokens(model, "abc", seq_len=32)
    assert int(out2[0, 0]) == BOS
    assert out2.shape[1] == 4  # BOS + 3 chars
