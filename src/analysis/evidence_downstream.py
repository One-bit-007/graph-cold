"""Downstream checks for evidence-preserving retention claims."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAIR_KEYS = ("dataset", "noise_type", "noise_rate", "graph_beta", "seed")


def tail_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    benign_class: int = 0,
    tail_quantile: float = 0.5,
) -> float:
    """Recall over low-frequency non-benign classes."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    labels, counts = np.unique(y_true[y_true != benign_class], return_counts=True)
    if labels.size == 0:
        return float("nan")
    threshold = float(np.quantile(counts, tail_quantile))
    tail_labels = labels[counts <= threshold]
    mask = np.isin(y_true, tail_labels)
    if not mask.any():
        return float("nan")
    return float(np.mean(y_pred[mask] == y_true[mask]))


def high_noise_fnr(row: pd.Series | dict[str, Any], threshold: float = 0.4) -> float:
    """Return FNR only for high-noise settings, otherwise NaN."""

    noise_rate = float(row.get("noise_rate", 0.0))
    if noise_rate < threshold:
        return float("nan")
    return float(row.get("fnr", np.nan))


def pair_graphcold_vs_hard(frame: pd.DataFrame) -> pd.DataFrame:
    """Create narrative-ready Graph-CoLD versus hard-ablation deltas."""

    required = {"method", "fnr", "macro_f1", *PAIR_KEYS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for downstream comparison: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(list(PAIR_KEYS), dropna=False):
        by = part.set_index("method")
        if "Graph-CoLD" not in by.index or "ablation_hard" not in by.index:
            continue
        graph = by.loc["Graph-CoLD"]
        hard = by.loc["ablation_hard"]
        record = {key: value for key, value in zip(PAIR_KEYS, keys)}
        graph_tail = float(graph.get("tail_recall", np.nan))
        hard_tail = float(hard.get("tail_recall", np.nan))
        record.update(
            {
                "graphcold_macro_f1": float(graph["macro_f1"]),
                "hard_macro_f1": float(hard["macro_f1"]),
                "macro_f1_delta": float(graph["macro_f1"] - hard["macro_f1"]),
                "graphcold_fnr": float(graph["fnr"]),
                "hard_fnr": float(hard["fnr"]),
                "fnr_delta_graphcold_minus_hard": float(graph["fnr"] - hard["fnr"]),
                "graphcold_high_noise_fnr": high_noise_fnr(graph),
                "hard_high_noise_fnr": high_noise_fnr(hard),
                "tail_recall_graphcold": graph_tail,
                "tail_recall_hard": hard_tail,
                "tail_recall_delta": graph_tail - hard_tail if np.isfinite([graph_tail, hard_tail]).all() else np.nan,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def counterfactual_soft_retention(
    soft_weights: np.ndarray,
    hard_weights: np.ndarray,
    flip_mask: np.ndarray,
    y_true: np.ndarray,
    *,
    y_pred_soft: np.ndarray | None = None,
    retention_threshold: float = 0.1,
) -> dict[str, float]:
    """Inspect samples kept by soft evidence weighting and removed by hard deletion."""

    soft = np.asarray(soft_weights, dtype=float) >= float(retention_threshold)
    hard = np.asarray(hard_weights, dtype=float) >= float(retention_threshold)
    flip = np.asarray(flip_mask, dtype=bool)
    y_true = np.asarray(y_true, dtype=np.int64)
    if soft.shape != hard.shape or soft.shape != flip.shape or soft.shape != y_true.shape:
        raise ValueError("weights, flip_mask, and y_true must be aligned.")
    counterfactual = soft & ~hard
    total = int(counterfactual.sum())
    if total == 0:
        return {
            "soft_retained_hard_deleted_n": 0.0,
            "soft_retained_hard_deleted_clean_fraction": float("nan"),
            "soft_retained_hard_deleted_correct_fraction": float("nan"),
        }
    clean_fraction = float(np.mean(~flip[counterfactual]))
    correct_fraction = float("nan")
    if y_pred_soft is not None:
        pred = np.asarray(y_pred_soft, dtype=np.int64)
        if pred.shape != y_true.shape:
            raise ValueError("y_pred_soft must align with y_true when provided.")
        correct_fraction = float(np.mean(pred[counterfactual] == y_true[counterfactual]))
    return {
        "soft_retained_hard_deleted_n": float(total),
        "soft_retained_hard_deleted_clean_fraction": clean_fraction,
        "soft_retained_hard_deleted_correct_fraction": correct_fraction,
    }


def write_downstream_benefit_csv(frame: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    """Write Graph-CoLD versus hard-ablation downstream deltas."""

    comparison = pair_graphcold_vs_hard(frame)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False)
    return comparison
