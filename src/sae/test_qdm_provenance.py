"""Regression tests for executable SAE provenance."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from src.sae.protocol import ReferenceProvenance
from src.sae.qdm import _corpus_texts, write_resolved_provenance


def test_configured_corpus_revision_reaches_dataset_loader(monkeypatch):
    """datasets 5.x forwards unknown kwargs into BuilderConfig and raises, so a
    load_dataset without an explicit timeout parameter must never receive one."""
    calls = []

    def load_dataset(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return [{"text": "provenance test"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    _corpus_texts(
        {"identifier": "org/corpus", "config": "subset", "split": "validation", "revision": "abc123", "streaming": False},
        1,
        0,
    )
    assert calls == [("org/corpus", {"split": "validation", "name": "subset", "revision": "abc123", "streaming": False})]


def test_timeout_passed_only_when_load_dataset_declares_it(monkeypatch):
    calls = []

    def load_dataset(identifier, *, timeout=None, **kwargs):
        calls.append((identifier, timeout))
        return [{"text": "provenance test"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    _corpus_texts(
        {"identifier": "org/corpus", "config": "subset", "split": "validation", "revision": "abc123", "streaming": False},
        1,
        0,
    )
    assert calls == [("org/corpus", 300)]


def test_corpus_pull_budget_covers_the_full_wikitext_train_split(monkeypatch):
    """Regression: the 500k reproduction budget must not starve short-line corpora.

    Duan 2606.03002 evaluates 499,968 tokens from ALL non-empty WikiText-2-raw
    train lines (~36,718 short lines). Pulling only n_tokens // 64 documents
    yields ~7.8k lines (~140k tokens) and silently shortens the sweep.
    """
    n_lines = 36_718

    def load_dataset(identifier, **kwargs):
        return [{"text": f"line {i}"} for i in range(n_lines)]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    texts = _corpus_texts(
        {"identifier": "wikitext", "config": "wikitext-2-raw-v1", "split": "train",
         "revision": "pinned", "streaming": False},
        500_000,
        0,
    )
    assert len(texts) == n_lines, (
        f"only {len(texts)} of {n_lines} wikitext lines pulled — the sweep would "
        "evaluate far fewer tokens than the paper's 499,968"
    )


def test_reproduction_profile_pins_duan_block_length(monkeypatch):
    """Duan 2606.03002 §3.8/App. K: Gemma QDM extraction uses block length L=256.

    Regression: the reproduction profile previously carried only seq_len=512
    (Pythia's block length), silently doubling every extracted context window
    relative to the paper's protocol.
    """
    from src.config import REPO_ROOT, Config

    cfg = Config.from_experiment(REPO_ROOT / "configs" / "reproduction" / "sae_gemma2_2b.yaml")
    perplexity = cfg.model["perplexity"]
    activation_seq_len = int(perplexity.get("activation_seq_len", perplexity["seq_len"]))
    assert activation_seq_len == 256, (
        f"SAE activation extraction must use the paper's Gemma block length L=256, "
        f"got {activation_seq_len}"
    )


def test_corpus_load_failure_is_loud(monkeypatch):
    import pytest

    def load_dataset(identifier, **kwargs):
        raise ConnectionError("blocked endpoint")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    with pytest.raises(RuntimeError, match="failed to load corpus org/corpus"):
        _corpus_texts(
            {"identifier": "org/corpus", "config": "subset", "split": "validation",
             "revision": "abc123", "streaming": False},
            1,
            0,
        )


def test_resolved_provenance_records_loader_inputs(tmp_path):
    provenance = ReferenceProvenance(
        source_reference="source@1", corpus_id="org/corpus", corpus_config="subset",
        corpus_split="train", corpus_revision="corpus@1", tokenization_protocol="packed",
        tokenizer_id="org/tokenizer", tokenizer_revision="tokenizer@1", bos_policy="always",
        sequence_packing="fixed", sae_release="release", sae_id="id", sae_revision="sae@1",
    )
    destination = tmp_path / "provenance_resolved.json"
    write_resolved_provenance(
        destination, cfg=SimpleNamespace(config_path="profile.yaml"), provenance=provenance,
        model_id="org/model", model_revision="model@1",
    )
    text = destination.read_text(encoding="utf-8")
    assert '"revision": "corpus@1"' in text
    assert '"revision": "model@1"' in text
    assert '"revision": "sae@1"' in text
