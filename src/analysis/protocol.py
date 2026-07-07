"""Canonical aggregation protocol for paper-facing Graph-CoLD numbers.

P2 fixes reviewer-visible number drift by using one headline protocol across
status reports, paper tables, and manuscript text. The protocol is deliberately
simple and traceable:

1. Use the frozen expanded real-data matrix, normally
   ``results/table_main_expanded.csv``.
2. Group by ``dataset, reported_as, noise_type, noise_rate, graph_beta, method``.
3. Average repeated seeds inside each scenario.
4. Average the scenario means with equal scenario weight for each method.

This matches the scenario-level paired tests in :mod:`src.analysis.stat_tests`
and avoids treating seed repeats from the same scenario as independent evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_ID = "p2_canonical_scenario_mean_v1"
SOURCE_CSV = Path("results/table_main_expanded.csv")
SCENARIO_KEYS = ("dataset", "reported_as", "noise_type", "noise_rate", "graph_beta")
METRICS = ("macro_f1", "fpr", "fnr", "err_final", "compression_ratio", "runtime_sec", "memory_mb")


def source_hash(path: str | Path = SOURCE_CSV) -> str:
    """Return the SHA-256 hash for a frozen source table."""

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scenario_means(frame: pd.DataFrame, metrics: tuple[str, ...] = METRICS) -> pd.DataFrame:
    """Average seeds within each canonical scenario and method."""

    required = {"method", *SCENARIO_KEYS, *metrics}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical protocol input is missing columns: {missing}")
    grouped = (
        frame.groupby([*SCENARIO_KEYS, "method"], dropna=False)[list(metrics)]
        .mean()
        .reset_index()
    )
    grouped["protocol_id"] = PROTOCOL_ID
    return grouped


def headline_table(frame: pd.DataFrame, metrics: tuple[str, ...] = METRICS) -> pd.DataFrame:
    """Return one canonical headline row per method."""

    scenarios = scenario_means(frame, metrics=metrics)
    grouped = scenarios.groupby("method", dropna=False)[list(metrics)].agg(["mean", _std]).reset_index()
    grouped.columns = [
        col[0] if not col[1] else f"{col[0]}_{'std' if col[1] == '_std' else col[1]}"
        for col in grouped.columns
    ]
    scenario_counts = scenarios.groupby("method", dropna=False).size().rename("scenario_count").reset_index()
    out = grouped.merge(scenario_counts, on="method", how="left")
    out.insert(0, "protocol_id", PROTOCOL_ID)
    return out.sort_values("method").reset_index(drop=True)


def method_headline_map(frame: pd.DataFrame, metric: str = "macro_f1") -> dict[str, float]:
    """Map each method to its canonical headline value for one metric."""

    table = headline_table(frame, metrics=(metric,))
    return {str(row["method"]): float(row[f"{metric}_mean"]) for _, row in table.iterrows()}


def write_protocol_artifacts(
    source_csv: str | Path = SOURCE_CSV,
    out_csv: str | Path = "tables/table_p2_canonical_headline.csv",
    out_json: str | Path = "reports/p2_number_consistency.json",
) -> dict[str, Any]:
    """Write canonical headline table and traceability metadata."""

    source = Path(source_csv)
    frame = pd.read_csv(source)
    headline = headline_table(frame)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headline.to_csv(out_csv, index=False)
    report = {
        "protocol_id": PROTOCOL_ID,
        "source_csv": str(source).replace("\\", "/"),
        "source_sha256": source_hash(source),
        "scenario_keys": list(SCENARIO_KEYS),
        "seed_aggregation": "mean_over_seeds_per_scenario",
        "headline_aggregation": "equal_weight_mean_over_scenarios",
        "metrics": list(METRICS),
        "headline_csv": str(out_csv).replace("\\", "/"),
        "method_headlines": {
            str(row["method"]): {
                metric: float(row[f"{metric}_mean"])
                for metric in METRICS
                if f"{metric}_mean" in headline.columns and np.isfinite(float(row[f"{metric}_mean"]))
            }
            for _, row in headline.iterrows()
        },
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(_json_dumps(report), encoding="utf-8")
    return report


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, allow_nan=False)
