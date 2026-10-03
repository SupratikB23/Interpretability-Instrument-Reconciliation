"""Explicit experiment modes and provenance contracts for the SAE arm."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentMode(str, Enum):
    """Scientifically distinct SAE protocols."""

    REPRODUCTION = "reproduction"
    HARMONIZED_R1 = "harmonized"


@dataclass(frozen=True)
class ReferenceProvenance:
    """Resolved inputs required before an SAE run may claim reproduction."""

    source_reference: str
    corpus_id: str
    corpus_config: str | None
    corpus_split: str
    corpus_revision: str
    tokenization_protocol: str
    tokenizer_id: str
    tokenizer_revision: str
    bos_policy: str
    sequence_packing: str
    sae_release: str
    sae_id: str
    sae_revision: str

    def validate(self) -> None:
        missing = [
            name
            for name, value in asdict(self).items()
            if name != "corpus_config" and (value is None or not str(value).strip())
        ]
        if missing:
            raise ValueError(
                "SAE reproduction provenance is unresolved for "
                f"{', '.join(missing)}; supply the exact reference corpus, tokenization, "
                "tokenizer revision, and SAE checkpoint provenance before running"
            )


@dataclass(frozen=True)
class HarmonizedProvenance:
    """Configured common-corpus identity for an R1 comparison run."""

    corpus_id: str
    corpus_config: str | None
    corpus_split: str
    corpus_revision: str | None
    tokenization_protocol: str
    tokenizer_id: str
    tokenizer_revision: str | None
    bos_policy: str
    sequence_packing: str

    def validate(self) -> None:
        required = (
            "corpus_id",
            "corpus_split",
            "tokenization_protocol",
            "tokenizer_id",
            "bos_policy",
            "sequence_packing",
        )
        missing = [name for name in required if not str(getattr(self, name) or "").strip()]
        if missing:
            raise ValueError(
                "harmonized SAE provenance is missing configured fields: " + ", ".join(missing)
            )


RunProvenance = ReferenceProvenance | HarmonizedProvenance


def validate_protocol(mode: ExperimentMode | str, provenance: RunProvenance | None) -> ExperimentMode:
    """Validate mode/provenance separation and return a normalized mode."""
    try:
        normalized = ExperimentMode(mode)
    except ValueError as error:
        allowed = ", ".join(item.value for item in ExperimentMode)
        raise ValueError(f"unknown SAE experiment mode {mode!r}; expected one of: {allowed}") from error

    if normalized is ExperimentMode.REPRODUCTION:
        if not isinstance(provenance, ReferenceProvenance):
            raise ValueError(
                "SAE reproduction requires explicit ReferenceProvenance; configured common-corpus "
                "settings are harmonized inputs and cannot support a reproduction claim"
            )
    elif isinstance(provenance, ReferenceProvenance):
        raise ValueError(
            "reference reproduction provenance was supplied to harmonized mode; use "
            "HarmonizedProvenance so the artifact cannot be mislabeled as reproduction"
        )
    elif not isinstance(provenance, HarmonizedProvenance):
        raise ValueError("harmonized SAE mode requires explicit HarmonizedProvenance")

    provenance.validate()
    return normalized


def artifact_directory(root: str | Path, mode: ExperimentMode | str) -> Path:
    """Return the non-overlapping artifact directory for one SAE protocol."""
    normalized = ExperimentMode(mode)
    return Path(root) / normalized.value / "sae"


def provenance_dict(provenance: RunProvenance) -> dict[str, Any]:
    return asdict(provenance)
