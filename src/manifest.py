"""Run-manifest registration.

Every stage registers the artifacts it produces in ``<results_root>/manifest.json``
so the full-run-readiness gate and downstream analysis can verify completeness
instead of trusting that a stage silently produced what it claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def register_artifact(
    results_root: str | Path,
    *,
    artifact_type: str,
    path: str | Path,
    stage: str,
    instrument: str | None = None,
    condition: str | None = None,
    status: str = "complete",
) -> None:
    """Append one artifact entry to the run manifest."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    entries: list[dict] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError(
                f"manifest {manifest_path} is corrupt; refusing to append — inspect or remove it"
            ) from None
    entries.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_type": artifact_type,
            "path": str(path),
            "experiment_id": stage,
            "instrument": instrument,
            "condition": condition,
            "status": status,
        }
    )
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(results_root: str | Path) -> list[dict]:
    """Return the registered manifest entries (empty when none exist yet)."""
    manifest_path = Path(results_root) / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))
