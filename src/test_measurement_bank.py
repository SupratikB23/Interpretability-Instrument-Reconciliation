"""Unit tests for scripts/build_measurement_bank.py.

Pure-logic tests only (no network, no HF): the record builder, ID scheme,
empty-input refusal, and manifest round-trip. Loaded via importlib because
scripts/ is not a package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_measurement_bank.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_measurement_bank", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _kwargs(**overrides):
    base = {
        "source": "wikitext",
        "tokenizer_name": "whitespace",
        "count_fn": builder.count_tokens_whitespace,
        "category": "general-web",
        "license": "TEST",
        "dataset_id": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_split": "train",
        "dataset_revision": "abc123",
    }
    base.update(overrides)
    return base


def test_record_ids_are_deterministic_and_zero_padded():
    assert builder.make_record_id("wikitext", 0) == "wikitext:000000"
    assert builder.make_record_id("wikitext", 42) == "wikitext:000042"
    with pytest.raises(ValueError, match="non-empty"):
        builder.make_record_id("  ", 0)
    with pytest.raises(ValueError, match="non-negative"):
        builder.make_record_id("wikitext", -1)


def test_build_records_skips_empty_and_counts_tokens():
    records, skipped = builder.build_records(
        ["hello world", "   ", "", "a b c"], **_kwargs()
    )
    assert skipped == 2
    assert [r["id"] for r in records] == ["wikitext:000000", "wikitext:000001"]
    assert [r["token_count"] for r in records] == [2, 3]
    assert records[0]["tokenizer"] == "whitespace"
    assert records[0]["dataset_revision"] == "abc123"


def test_build_records_refuses_empty_bank():
    with pytest.raises(ValueError, match="empty bank"):
        builder.build_records([], **_kwargs())
    with pytest.raises(ValueError, match="empty bank"):
        builder.build_records(["  ", ""], **_kwargs())


def test_manifest_round_trip(tmp_path):
    records, _ = builder.build_records(["alpha beta", "gamma"], **_kwargs())
    path = tmp_path / "bank.jsonl"
    builder.write_manifest(records, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    loaded = [json.loads(line) for line in lines]
    assert loaded[0]["text"] == "alpha beta"
    assert set(loaded[0]) == {
        "id", "source", "text", "token_count", "tokenizer", "category",
        "license", "dataset_id", "dataset_config", "dataset_split",
        "dataset_revision",
    }
