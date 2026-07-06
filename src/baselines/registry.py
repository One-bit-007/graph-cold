"""Registry for implemented real-data noisy-label baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.baselines.confident_learning import ConfidentLearningBaseline
from src.baselines.coteaching import CoTeachingBaseline
from src.baselines.decoupling import DecouplingBaseline
from src.baselines.fine_style import FINEBaseline
from src.baselines.mcre import MCReBaseline
from src.baselines.morse import MORSEBaseline
from src.baselines.noisy_supervised import NoisySupervisedBaseline


@dataclass(frozen=True)
class BaselineRegistryEntry:
    method: str
    method_family: str
    implementation_status: str
    faithfulness_level: str
    uses_noisy_y_train: bool
    uses_clean_y_test_for_eval_only: bool
    verified: bool
    include_in_formal_results: bool
    notes: str
    factory: Callable[..., object]


REGISTRY: dict[str, BaselineRegistryEntry] = {
    "Noisy-Supervised": BaselineRegistryEntry(
        method="Noisy-Supervised",
        method_family="noisy_supervised",
        implementation_status="verified_implementation",
        faithfulness_level="plain tabular classifier trained on noisy labels",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method.",
        factory=NoisySupervisedBaseline,
    ),
    "Confident-Learning": BaselineRegistryEntry(
        method="Confident-Learning",
        method_family="confident_learning",
        implementation_status="verified_implementation",
        faithfulness_level="cleanlab when installed, otherwise documented confidence filtering",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method.",
        factory=ConfidentLearningBaseline,
    ),
    "Co-Teaching": BaselineRegistryEntry(
        method="Co-Teaching",
        method_family="co_teaching",
        implementation_status="verified_implementation",
        faithfulness_level="standard small-loss exchange Co-Teaching with two tabular classifiers",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method.",
        factory=CoTeachingBaseline,
    ),
    "Decoupling": BaselineRegistryEntry(
        method="Decoupling",
        method_family="decoupling",
        implementation_status="verified_implementation",
        faithfulness_level="standard tabular implementation of disagreement-update Decoupling",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method.",
        factory=DecouplingBaseline,
    ),
    "FINE": BaselineRegistryEntry(
        method="FINE",
        method_family="fine",
        implementation_status="verified_implementation",
        faithfulness_level="FINE eigenvector filtering adapter with per-setting stability caveat",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method; unstable settings are retained with a caveat.",
        factory=FINEBaseline,
    ),
    "MCRe": BaselineRegistryEntry(
        method="MCRe",
        method_family="mcre",
        implementation_status="verified_implementation",
        faithfulness_level="MCRe-style class-wise representation purification for tabular IDS features",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method.",
        factory=MCReBaseline,
    ),
    "MORSE": BaselineRegistryEntry(
        method="MORSE",
        method_family="morse",
        implementation_status="verified_implementation",
        faithfulness_level="MORSE-style noisy-as-unlabeled semi-supervised purification",
        uses_noisy_y_train=True,
        uses_clean_y_test_for_eval_only=True,
        verified=True,
        include_in_formal_results=True,
        notes="Formal baseline expansion method.",
        factory=MORSEBaseline,
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


def registry_metadata(verified_methods: set[str] | None = None) -> dict[str, dict[str, object]]:
    passed = verified_methods or set()
    out: dict[str, dict[str, object]] = {}
    for method, entry in REGISTRY.items():
        out[method] = {
            "method": entry.method,
            "method_family": entry.method_family,
            "implementation_status": "verified_implementation"
            if (entry.verified or method in passed)
            else entry.implementation_status,
            "faithfulness_level": entry.faithfulness_level,
            "uses_noisy_y_train": entry.uses_noisy_y_train,
            "uses_clean_y_test_for_eval_only": entry.uses_clean_y_test_for_eval_only,
            "verified": bool(entry.verified or method in passed),
            "include_in_formal_results": bool(entry.include_in_formal_results or method in passed),
            "notes": entry.notes,
        }
    return out
