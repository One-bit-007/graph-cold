"""Paired statistical tests for Graph-CoLD result tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PAIR_KEYS = ["dataset", "noise_type", "noise_rate", "graph_beta", "beta", "seed"]


def paired_summary(
    frame: pd.DataFrame,
    metric: str = "macro_f1",
    method_a: str = "Graph-CoLD",
    method_b: str = "CoLD",
) -> dict[str, Any]:
    required = {"method", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"Result frame must contain columns: {sorted(required)}")
    keys = [key for key in PAIR_KEYS if key in frame.columns]
    if "seed" not in keys:
        raise ValueError("Paired tests require a seed column; naive pooled tests are forbidden.")
    working = frame.copy()
    for key in keys:
        if working[key].isna().any():
            working[key] = working[key].fillna("__NA__")
    a = working[working["method"] == method_a][keys + [metric]].rename(columns={metric: "a"})
    b = working[working["method"] == method_b][keys + [metric]].rename(columns={metric: "b"})
    pairs = a.merge(b, on=keys, how="inner")
    if pairs.empty:
        raise ValueError(f"No paired rows found for {method_a} vs {method_b}.")
    diff = pairs["a"].to_numpy(dtype=float) - pairs["b"].to_numpy(dtype=float)
    if len(pairs) < 2:
        t_stat, p_val = np.nan, np.nan
    else:
        t_stat, p_val = stats.ttest_rel(pairs["a"], pairs["b"], alternative="greater")
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    effect = float(np.mean(diff) / sd) if sd > 0 else float("inf")
    report = {
        "test": "paired_t_test_greater",
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "pairing_keys": keys,
        "n_pairs": int(len(pairs)),
        "method_a_mean": float(pairs["a"].mean()),
        "method_a_std": float(pairs["a"].std(ddof=1)) if len(pairs) > 1 else 0.0,
        "method_b_mean": float(pairs["b"].mean()),
        "method_b_std": float(pairs["b"].std(ddof=1)) if len(pairs) > 1 else 0.0,
        "mean_diff": float(np.mean(diff)),
        "effect_size_cohen_dz": effect,
        "t_stat": None if np.isnan(t_stat) else float(t_stat),
        "p_value": None if np.isnan(p_val) else float(p_val),
        "significant_p_lt_0_05": bool(False if np.isnan(p_val) else p_val < 0.05),
        "extreme_p_value_warning": bool(False if np.isnan(p_val) else p_val < 1e-20),
        "naive_pooled_test_used": False,
    }
    return report


def grouped_paired_summary(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...] = ("dataset", "noise_type", "noise_rate"),
    metric: str = "macro_f1",
) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": paired_summary(frame, metric=metric), "groups": {}, "comparisons": {}}
    for method in sorted(set(frame.get("method", [])) - { "Graph-CoLD" }):
        try:
            out["comparisons"][f"Graph-CoLD_vs_{method}"] = paired_summary(
                frame,
                metric=metric,
                method_a="Graph-CoLD",
                method_b=method,
            )
        except ValueError as exc:
            out["comparisons"][f"Graph-CoLD_vs_{method}"] = {"skipped": True, "reason": str(exc)}
    cols = [col for col in group_cols if col in frame.columns]
    for key, part in frame.groupby(cols, dropna=False):
        label = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
        try:
            out["groups"][label] = paired_summary(part, metric=metric)
        except ValueError as exc:
            out["groups"][label] = {"skipped": True, "reason": str(exc)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--input", dest="input_csv")
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--out")
    args = parser.parse_args()
    csv_path = args.input_csv or args.csv
    if not csv_path:
        raise SystemExit("Provide a result CSV path as positional csv or --input.")
    report = grouped_paired_summary(pd.read_csv(csv_path), metric=args.metric)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
