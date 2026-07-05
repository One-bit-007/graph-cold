"""Shared helpers for the D9.5 baseline reinforcement runners."""
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.base import BaselineResult, array_hash
from src.baselines.decoupling import DecouplingBaseline
from src.baselines.fine_style import FINEStyleBaseline
from src.experiments import d5, d5_baseline_expansion


METHODS = ("Decoupling", "FINE-style")
EXTRA_FIELDNAMES = (
    "faithfulness_level",
    "baseline_source",
    "smoke_passed",
    "implementation_notes",
)
REINFORCED_FIELDNAMES = d5_baseline_expansion.EXPANDED_FIELDNAMES + EXTRA_FIELDNAMES

FAITHFULNESS = {
    "Decoupling": "standard tabular implementation of disagreement-update Decoupling",
    "FINE-style": "representation-eigenvector filtering inspired by FINE; not full original implementation",
}
BASELINE_SOURCE = {
    "Decoupling": "local_sklearn_sgd_disagreement_update",
    "FINE-style": "local_standardized_features_pca_eigenvector_filter",
}
IMPLEMENTATION_NOTES = {
    "Decoupling": "two SGD classifiers update on post-warmup disagreements using noisy_y_train",
    "FINE-style": "class-wise top-eigenvector alignment filter; explicitly not full FINE",
}


def make_baseline(method: str, seed: int, noise_rate: float):
    if method == "Decoupling":
        return DecouplingBaseline(seed=seed, epochs=4, warmup_epochs=1, batch_size=16384, alpha=1e-3)
    if method == "FINE-style":
        return FINEStyleBaseline(seed=seed, noise_rate=noise_rate, n_components=32, classifier_epochs=5)
    raise ValueError(f"Unsupported D9.5 baseline: {method}")


def timed_baseline(
    baseline,
    bundle: d5.FormalBundle,
    noisy: np.ndarray,
    spec: dict[str, Any],
    representation=None,
) -> tuple[BaselineResult, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    result = baseline.fit_predict(
        bundle.dataset.X_train,
        noisy,
        bundle.dataset.X_test,
        bundle.dataset.num_classes,
        y_clean_train=bundle.dataset.y_train,
        y_clean_test=bundle.dataset.y_test,
        noise_rate=float(spec["noise_rate"]),
        representation=representation,
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, time.perf_counter() - start, peak / (1024 * 1024)


def row_from_result(
    bundle: d5.FormalBundle,
    spec: dict[str, Any],
    seed: int,
    result: BaselineResult,
    runtime_sec: float,
    memory_mb: float,
    evidence: np.ndarray,
    flip: np.ndarray,
    smoke_passed: bool,
) -> dict[str, Any]:
    row = d5_baseline_expansion._row_from_result(bundle, spec, seed, result, runtime_sec, memory_mb, evidence, flip)
    row["faithfulness_level"] = FAITHFULNESS[result.method]
    row["baseline_source"] = BASELINE_SOURCE[result.method]
    row["smoke_passed"] = bool(smoke_passed)
    row["implementation_notes"] = IMPLEMENTATION_NOTES[result.method]
    return {name: row.get(name, "") for name in REINFORCED_FIELDNAMES}


def finite_metrics(row: dict[str, Any]) -> bool:
    keys = ("macro_f1", "fpr", "fnr", "err", "err_tail", "err_final", "compression_ratio", "runtime_sec", "memory_mb")
    return bool(np.isfinite([float(row[key]) for key in keys]).all())


def perfect_metric_anomaly(row: dict[str, Any]) -> bool:
    return bool(float(row["macro_f1"]) >= 0.999 and float(row["fpr"]) <= 0.001 and float(row["fnr"]) <= 0.001)


def pass_smoke_row(row: dict[str, Any], result: BaselineResult, noisy: np.ndarray) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not finite_metrics(row):
        reasons.append("non_finite_metric")
    if perfect_metric_anomaly(row):
        reasons.append("perfect_metric_anomaly")
    if result.details.get("training_label_hash") != array_hash(noisy):
        reasons.append("not_trained_on_noisy_y_train")
    if float(row["macro_f1"]) < 0.50:
        reasons.append("macro_f1_below_0_50")
    if result.method == "FINE-style" and float(row["retained_fraction"]) <= 0.0:
        reasons.append("fine_style_zero_retention")
    if result.method == "Decoupling":
        if "disagreement_fraction" not in result.details or "update_fraction" not in result.details:
            reasons.append("decoupling_missing_disagreement_metadata")
    return not reasons, reasons


def original_expanded_with_extra(path: str | Path = "results/table_main_expanded.csv") -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    for col in EXTRA_FIELDNAMES:
        if col not in frame.columns:
            frame[col] = ""
    return frame.reindex(columns=REINFORCED_FIELDNAMES)


def assert_original_rows_unchanged(original_path: str | Path, reinforced: pd.DataFrame) -> bool:
    original = pd.read_csv(original_path, keep_default_na=False)
    original_cols = list(original.columns)
    prefix = reinforced.iloc[: len(original)][original_cols].reset_index(drop=True)
    return bool(prefix.equals(original.reset_index(drop=True)))
