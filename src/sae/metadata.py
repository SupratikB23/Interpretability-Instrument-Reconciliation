"""Complete, JSON-safe run metadata for SAE artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import ExperimentMode, RunProvenance, provenance_dict


@dataclass(frozen=True)
class RunMetadata:
    """Caller-visible scientific identity for one SAE condition.

    Unknown optional values remain ``None`` in artifacts; they are never replaced
    with guessed revisions or environment details.
    """

    mode: str
    timestamp_utc: str
    git_commit: str | None
    instrument: str
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    sae_release: str
    sae_id: str
    sae_revision: str | None
    hook_name: str
    corpus_id: str
    corpus_config: str | None
    corpus_split: str
    corpus_revision: str | None
    token_budget: int
    sequence_length: int
    window_size: int
    stride: int
    bits: int
    seed: int
    compression_method: str
    quantization_audit: dict[str, Any] | None
    package_versions: dict[str, str | None]
    device: str
    dtype: str
    config_path: str | None
    protocol_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit(repository: Path | None) -> str | None:
    if repository is None:
        return None
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value or None


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Return installed versions, preserving unavailable packages as null."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_metadata(
    *,
    mode: ExperimentMode,
    provenance: RunProvenance,
    model_id: str,
    model_revision: str | None,
    sae_release: str,
    sae_id: str,
    sae_revision: str | None,
    hook_name: str,
    token_budget: int,
    sequence_length: int,
    window_size: int,
    stride: int,
    bits: int,
    seed: int,
    quantization_audit: dict[str, Any] | None,
    device: str,
    dtype: str,
    config_path: str | None = None,
    repository: Path | None = None,
) -> RunMetadata:
    return RunMetadata(
        mode=mode.value,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(repository),
        instrument="sae",
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=provenance.tokenizer_id,
        tokenizer_revision=provenance.tokenizer_revision,
        sae_release=sae_release,
        sae_id=sae_id,
        sae_revision=sae_revision,
        hook_name=hook_name,
        corpus_id=provenance.corpus_id,
        corpus_config=provenance.corpus_config,
        corpus_split=provenance.corpus_split,
        corpus_revision=provenance.corpus_revision,
        token_budget=token_budget,
        sequence_length=sequence_length,
        window_size=window_size,
        stride=stride,
        bits=bits,
        seed=seed,
        compression_method="none" if bits == 16 else "RTN",
        quantization_audit=quantization_audit,
        package_versions=package_versions(("numpy", "torch", "transformer-lens", "sae-lens", "datasets")),
        device=device,
        dtype=dtype,
        config_path=config_path,
        protocol_provenance=provenance_dict(provenance),
    )


def write_metadata(path: str | Path, metadata: RunMetadata | list[RunMetadata]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.to_dict() for item in metadata] if isinstance(metadata, list) else metadata.to_dict()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
