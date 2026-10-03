"""Scientific-contract tests for SAE protocol separation and PPL alignment."""

import csv

import pytest
import torch

from src.sae.perplexity import evaluate_aligned_perplexities, write_perplexity_csv
from src.sae.protocol import (
    ExperimentMode,
    HarmonizedProvenance,
    ReferenceProvenance,
    artifact_directory,
    validate_protocol,
)


def _reference(**overrides):
    values = {
        "source_reference": "reference implementation commit abc123",
        "corpus_id": "reference/corpus",
        "corpus_config": "default",
        "corpus_split": "validation",
        "corpus_revision": "corpus-commit",
        "tokenization_protocol": "reference concat-v1",
        "tokenizer_id": "model/tokenizer",
        "tokenizer_revision": "tokenizer-commit",
        "bos_policy": "single_document_bos",
        "sequence_packing": "concatenate_then_chunk",
        "sae_release": "release",
        "sae_id": "checkpoint",
        "sae_revision": "sae-commit",
    }
    values.update(overrides)
    return ReferenceProvenance(**values)


def _harmonized():
    return HarmonizedProvenance(
        corpus_id="common/r1",
        corpus_config=None,
        corpus_split="train",
        corpus_revision=None,
        tokenization_protocol="r1 common token bank",
        tokenizer_id="model/tokenizer",
        tokenizer_revision=None,
        bos_policy="prepend_each_document",
        sequence_packing="fixed_length",
    )


def test_reproduction_rejects_unknown_reference_provenance():
    with pytest.raises(ValueError, match="corpus_revision"):
        validate_protocol(
            ExperimentMode.REPRODUCTION,
            _reference(corpus_revision="unknown" if False else ""),
        )


def test_harmonized_cannot_be_labeled_reproduction():
    with pytest.raises(ValueError, match="ReferenceProvenance"):
        validate_protocol(ExperimentMode.REPRODUCTION, _harmonized())
    with pytest.raises(ValueError, match="cannot be mislabeled"):
        validate_protocol(ExperimentMode.HARMONIZED_R1, _reference())


def test_mode_artifact_directories_do_not_overlap(tmp_path):
    reproduction = artifact_directory(tmp_path, ExperimentMode.REPRODUCTION)
    harmonized = artifact_directory(tmp_path, ExperimentMode.HARMONIZED_R1)
    assert reproduction == tmp_path / "reproduction" / "sae"
    assert harmonized == tmp_path / "harmonized" / "sae"
    assert reproduction != harmonized


class _PredictNext(torch.nn.Module):
    def __init__(self, confidence: float):
        super().__init__()
        self.confidence = confidence
        self.register_buffer("anchor", torch.tensor(0.0))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, tokens, return_type="logits"):
        vocab = 32
        logits = torch.zeros((*tokens.shape, vocab), device=tokens.device)
        next_tokens = tokens[:, 1:]
        logits[:, :-1].scatter_(2, next_tokens.unsqueeze(-1), self.confidence)
        return logits


def test_sliding_window_scores_strictly_aligned_tokens_and_delta():
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = evaluate_aligned_perplexities(
        {16: _PredictNext(8.0), 6: _PredictNext(4.0)},
        tokens,
        window_size=5,
        stride=2,
        token_budget=7,
        seed=17,
    )
    assert results[16].scored_token_indices == tuple(range(1, 8))
    assert results[6].scored_token_indices == results[16].scored_token_indices
    assert results[6].token_digest == results[16].token_digest
    assert results[16].n_tokens == results[6].n_tokens == 7
    assert results[16].delta_ppl_percent == pytest.approx(0.0)
    assert results[6].delta_ppl_percent > 0


def test_perplexity_artifact_has_priority_2_schema(tmp_path):
    results = evaluate_aligned_perplexities(
        {16: _PredictNext(8.0), 4: _PredictNext(2.0)},
        torch.tensor([1, 2, 3, 4, 5, 6]),
        window_size=4,
        stride=2,
        token_budget=5,
        seed=3,
    )
    path = tmp_path / "perplexity.csv"
    write_perplexity_csv(path, results)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() == {
        "condition", "bits", "ppl", "delta_ppl_percent", "protocol",
        "n_tokens", "seq_len", "stride", "seed",
    }
    assert {row["condition"] for row in rows} == {"FP16", "INT4"}
