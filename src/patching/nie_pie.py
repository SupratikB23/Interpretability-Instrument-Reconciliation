"""Condition-safe NIE / PIE activation patching.

Every condition owns its model state, callable, activations, and baseline metrics.
Labels are validated keys, never switches or cosmetic metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..sae.protocol import ExperimentMode

Activation = npt.NDArray[np.generic]
ModelFn = Callable[[str, Activation, str], float]


@dataclass(frozen=True)
class PatchingCondition:
    """All state required to patch one genuinely distinct model condition."""

    label: str
    state_id: str
    state: object
    model_fn: ModelFn
    clean_acts: Mapping[str, Activation]
    corrupted_acts: Mapping[str, Activation]


@dataclass(frozen=True)
class PatchingResult:
    """Complete causal decomposition for one component and condition."""

    component: str
    label: str
    clean_metric: float
    corrupted_metric: float
    noised_metric: float
    denoised_metric: float
    nie: float
    pie: float
    int_term: float
    delta_nie: float = 0.0
    delta_pie: float = 0.0
    delta_int: float = 0.0
    normalized_nie_damage: float = 0.0
    normalized_pie_damage: float = 0.0
    normalized_int_damage: float = 0.0
    normalized_baseline_damage: float = 0.0
    int_confounded: bool = False
    nie_sign_disagreement: bool = False
    pie_sign_disagreement: bool = False
    int_sign_disagreement: bool = False
    nie_rank: int = 0
    pie_rank: int = 0
    rank_disagreement: bool = False


def compute_nie(
    model_fn: ModelFn,
    component: str,
    clean_act: Activation,
    corrupted_act: Activation,
    clean_metric: float,
) -> float:
    """Return clean_metric minus the noised metric."""
    del clean_act
    return float(clean_metric - model_fn(component, corrupted_act, "clean"))


def compute_pie(
    model_fn: ModelFn,
    component: str,
    clean_act: Activation,
    corrupted_act: Activation,
    corrupted_metric: float,
) -> float:
    """Return the denoised metric minus corrupted_metric."""
    del corrupted_act
    return float(model_fn(component, clean_act, "corrupted") - corrupted_metric)


def is_int_confounded(nie: float, pie: float, eps: float = 1e-9) -> bool:
    """Whether non-negligible NIE and PIE have opposite signs."""
    return abs(nie) >= eps and abs(pie) >= eps and bool((nie < 0) != (pie < 0))


def _sign_disagrees(value: float, reference: float, eps: float = 1e-9) -> bool:
    if abs(value) < eps or abs(reference) < eps:
        return False
    return bool((value < 0) != (reference < 0))


def _activation_fingerprint(condition: PatchingCondition, components: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for component in components:
        for kind, acts in (("clean", condition.clean_acts), ("corrupted", condition.corrupted_acts)):
            array = np.asarray(acts[component])
            digest.update(component.encode())
            digest.update(kind.encode())
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_conditions(
    conditions: Mapping[str, PatchingCondition], components: Sequence[str]
) -> None:
    if "fp16" not in conditions:
        raise ValueError("patching requires an explicit 'fp16' reference condition")
    if not components:
        raise ValueError("patching requires at least one component")

    state_ids: set[str] = set()
    state_objects: set[int] = set()
    callables: set[int] = set()
    activation_fingerprints: dict[str, str] = {}
    for key, condition in conditions.items():
        if key != condition.label:
            raise ValueError(
                f"condition key {key!r} does not match state label {condition.label!r}; "
                "labels must select their own state"
            )
        if not condition.state_id:
            raise ValueError(f"condition {key!r} has an empty state_id")
        if condition.state_id in state_ids or id(condition.state) in state_objects:
            raise ValueError(f"condition {key!r} reuses another condition's model state")
        if id(condition.model_fn) in callables:
            raise ValueError(f"condition {key!r} reuses another condition's model callable")
        state_ids.add(condition.state_id)
        state_objects.add(id(condition.state))
        callables.add(id(condition.model_fn))

        for component in components:
            if component not in condition.clean_acts or component not in condition.corrupted_acts:
                raise ValueError(
                    f"condition {key!r} lacks clean/corrupted activation for {component!r}"
                )
            clean = np.asarray(condition.clean_acts[component])
            corrupted = np.asarray(condition.corrupted_acts[component])
            if clean.size == 0 or corrupted.size == 0:
                raise ValueError(f"condition {key!r}, component {component!r} has an empty activation")
            if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(corrupted)):
                raise ValueError(
                    f"condition {key!r}, component {component!r} has non-finite activations"
                )

        fingerprint = _activation_fingerprint(condition, components)
        if fingerprint in activation_fingerprints:
            other = activation_fingerprints[fingerprint]
            raise ValueError(
                f"conditions {other!r} and {key!r} have identical clean/corrupted "
                "activations; refusing cosmetic condition labels"
            )
        activation_fingerprints[fingerprint] = key


def decompose_component(
    condition: PatchingCondition,
    component: str,
) -> PatchingResult:
    """Compute baselines, patched metrics, NIE, PIE, and INT for one condition."""
    clean_act = np.asarray(condition.clean_acts[component])
    corrupted_act = np.asarray(condition.corrupted_acts[component])
    clean_metric = float(condition.model_fn(component, clean_act, "clean"))
    corrupted_metric = float(condition.model_fn(component, corrupted_act, "corrupted"))
    noised_metric = float(condition.model_fn(component, corrupted_act, "clean"))
    denoised_metric = float(condition.model_fn(component, clean_act, "corrupted"))
    metrics = (clean_metric, corrupted_metric, noised_metric, denoised_metric)
    if not all(np.isfinite(metric) for metric in metrics):
        raise ValueError(
            f"condition {condition.label!r}, component {component!r} produced non-finite metrics"
        )
    nie = clean_metric - noised_metric
    pie = denoised_metric - corrupted_metric
    return PatchingResult(
        component=component,
        label=condition.label,
        clean_metric=clean_metric,
        corrupted_metric=corrupted_metric,
        noised_metric=noised_metric,
        denoised_metric=denoised_metric,
        nie=nie,
        pie=pie,
        int_term=nie - pie,
        int_confounded=is_int_confounded(nie, pie),
    )


def _safe_damage(reference: float, value: float, eps: float = 1e-12) -> float:
    """Fraction of reference magnitude lost; zero-reference changes remain loud."""
    if abs(reference) < eps:
        return 0.0 if abs(value) < eps else float(np.copysign(np.inf, reference - value))
    return float((reference - value) / abs(reference))


def _add_reference_comparisons(raw: list[PatchingResult]) -> list[PatchingResult]:
    reference = {r.component: r for r in raw if r.label == "fp16"}
    enriched: list[PatchingResult] = []
    for result in raw:
        fp = reference[result.component]
        baseline_gap = result.clean_metric - result.corrupted_metric
        fp_baseline_gap = fp.clean_metric - fp.corrupted_metric
        enriched.append(
            replace(
                result,
                delta_nie=result.nie - fp.nie,
                delta_pie=result.pie - fp.pie,
                delta_int=result.int_term - fp.int_term,
                normalized_nie_damage=_safe_damage(fp.nie, result.nie),
                normalized_pie_damage=_safe_damage(fp.pie, result.pie),
                normalized_int_damage=_safe_damage(fp.int_term, result.int_term),
                normalized_baseline_damage=_safe_damage(fp_baseline_gap, baseline_gap),
                nie_sign_disagreement=_sign_disagrees(result.nie, fp.nie),
                pie_sign_disagreement=_sign_disagrees(result.pie, fp.pie),
                int_sign_disagreement=_sign_disagrees(result.int_term, fp.int_term),
            )
        )

    ranked: list[PatchingResult] = []
    for label in dict.fromkeys(result.label for result in enriched):
        group = [result for result in enriched if result.label == label]
        nie_order = {
            result.component: rank
            for rank, result in enumerate(sorted(group, key=lambda item: (-abs(item.nie), item.component)), 1)
        }
        pie_order = {
            result.component: rank
            for rank, result in enumerate(sorted(group, key=lambda item: (-abs(item.pie), item.component)), 1)
        }
        ranked.extend(
            replace(
                result,
                nie_rank=nie_order[result.component],
                pie_rank=pie_order[result.component],
                rank_disagreement=nie_order[result.component] != pie_order[result.component],
            )
            for result in group
        )
    return ranked


def run_patching_arm(
    components: Sequence[str],
    conditions: Mapping[str, PatchingCondition],
    *,
    output_dir: Path,
    mode: ExperimentMode | str = "harmonized",
) -> list[PatchingResult]:
    """Run condition-specific patching and persist complete metrics.

    Artifacts are written under ``<output_dir>/<mode>/patching/`` so the experiment mode is part of the artifact path, not a label.
    """
    normalized_mode = ExperimentMode(mode)
    _validate_conditions(conditions, components)
    raw = [
        decompose_component(condition, component)
        for condition in conditions.values()
        for component in components
    ]
    results = _add_reference_comparisons(raw)

    out = output_dir / normalized_mode.value / "patching"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "nie_pie.csv"
    fields = list(PatchingResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": normalized_mode.value,
        "instrument": "activation_patching",
        "components": list(components),
        "conditions": {
            key: condition.state_id for key, condition in conditions.items()
        },
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    return results


def list_int_confounded(results: Sequence[PatchingResult]) -> list[PatchingResult]:
    """Return cells where NIE and PIE disagree in sign."""
    return [result for result in results if result.int_confounded]
