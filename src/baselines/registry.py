"""Registry for implemented real-data noisy-label baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.baselines.confident_learning import ConfidentLearningBaseline
from src.baselines.coteaching import CoTeachingLiteBaseline
from src.baselines.decoupling import DecouplingBaseline
from src.baselines.fine_style import FINEStyleBaseline
from src.baselines.noisy_supervised import NoisySupervisedBaseline


@dataclass(frozen=True)
class BaselineRegistryEntry:
    method: str
    method_family: str
    implementation_status: str
    faithfulness_level: str
    uses_noisy_y_train: bool
    uses_clean_y_test_for_eval_only: bool
    smoke_passed: bool
    include_in_formal_results: bool
    notes: str
    factory: Callable[..., object]


REGISTRY: dict[str, BaselineRegistryEntry] = {
    "Noisy-Supervised": BaselineRegistryEntry(
        method="Noisy-Supervised",
        method_family="noisy_supervised",
        implementation_status="implemented_smoke_passed",
        faithfulness_level="plain tabular classifier trained on noisy labels",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        smoke_passed=True,
        include_in_formal_results=True,
        notes="D5.5 baseline expansion method.",
        factory=NoisySupervisedBaseline,
    ),
    "Confident-Learning": BaselineRegistryEntry(
        method="Confident-Learning",
        method_family="confident_learning",
        implementation_status="implemented_smoke_passed",
        faithfulness_level="cleanlab when installed, otherwise documented confidence filtering",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        smoke_passed=True,
        include_in_formal_results=True,
        notes="D5.5 baseline expansion method.",
        factory=ConfidentLearningBaseline,
    ),
    "Co-Teaching-lite": BaselineRegistryEntry(
        method="Co-Teaching-lite",
        method_family="co_teaching_lite",
        implementation_status="implemented_smoke_passed",
        faithfulness_level="lightweight tabular small-loss exchange; not full deep Co-Teaching",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        smoke_passed=True,
        include_in_formal_results=True,
        notes="D5.5 baseline expansion method.",
        factory=CoTeachingLiteBaseline,
    ),
    "Decoupling": BaselineRegistryEntry(
        method="Decoupling",
        method_family="decoupling",
        implementation_status="implemented_smoke_pending",
        faithfulness_level="standard tabular implementation of disagreement-update Decoupling",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        smoke_passed=False,
        include_in_formal_results=False,
        notes="D9.5 candidate; enters reinforced results only after smoke gate passes.",
        factory=DecouplingBaseline,
    ),
    "FINE-style": BaselineRegistryEntry(
        method="FINE-style",
        method_family="fine_style",
        implementation_status="implemented_smoke_pending",
        faithfulness_level="representation-eigenvector filtering inspired by FINE; not full original implementation",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        smoke_passed=False,
        include_in_formal_results=False,
        notes="D9.5 candidate; method name must not be shortened to FINE.",
        factory=FINEStyleBaseline,
    ),
}


def get_baseline_entry(method: str) -> BaselineRegistryEntry:
    try:
        return REGISTRY[method]
    except KeyError as exc:
        raise KeyError(f"Unknown registered baseline: {method}") from exc


def make_baseline(method: str, **kwargs):
    entry = get_baseline_entry(method)
    return entry.factory(**kwargs)


def registry_metadata(smoke_passed: set[str] | None = None) -> dict[str, dict[str, object]]:
    passed = smoke_passed or set()
    out: dict[str, dict[str, object]] = {}
    for method, entry in REGISTRY.items():
        out[method] = {
            "method": entry.method,
            "method_family": entry.method_family,
            "implementation_status": "implemented_smoke_passed"
            if (entry.smoke_passed or method in passed)
            else entry.implementation_status,
            "faithfulness_level": entry.faithfulness_level,
            "uses_noisy_y_train": entry.uses_noisy_y_train,
            "uses_clean_y_test_for_eval_only": entry.uses_clean_y_test_for_eval_only,
            "smoke_passed": bool(entry.smoke_passed or method in passed),
            "include_in_formal_results": bool(entry.include_in_formal_results or method in passed),
            "notes": entry.notes,
        }
    return out
