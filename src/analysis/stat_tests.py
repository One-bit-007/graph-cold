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
SCENARIO_KEYS = ["dataset", "noise_type", "noise_rate", "graph_beta", "beta"]


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


def scenario_level_paired_summary(
    frame: pd.DataFrame,
    metric: str = "macro_f1",
    method_a: str = "Graph-CoLD",
    method_b: str = "CoLD",
    bootstrap_iters: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Independence-aware paired comparison with one observation per scenario.

    Rows are first averaged over repeated seeds for each
    (dataset, noise_type, noise_rate, graph_beta/beta, method) cell. This avoids
    treating seed-level repeats from the same scenario as independent samples.
    The returned effective sample size is therefore the number of paired
    scenarios, not the number of seed-level rows.
    """

    required = {"method", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"Result frame must contain columns: {sorted(required)}")
    keys = [key for key in SCENARIO_KEYS if key in frame.columns]
    if not keys:
        raise ValueError("Scenario-level tests require at least one scenario key.")
    working = frame.copy()
    for key in keys:
        if working[key].isna().any():
            working[key] = working[key].fillna("__NA__")
    grouped = working.groupby(keys + ["method"], dropna=False)[metric].mean().reset_index()
    a = grouped[grouped["method"] == method_a][keys + [metric]].rename(columns={metric: "a"})
    b = grouped[grouped["method"] == method_b][keys + [metric]].rename(columns={metric: "b"})
    pairs = a.merge(b, on=keys, how="inner")
    if pairs.empty:
        raise ValueError(f"No scenario-level paired rows found for {method_a} vs {method_b}.")
    diff = pairs["a"].to_numpy(dtype=float) - pairs["b"].to_numpy(dtype=float)
    if len(pairs) < 2:
        t_stat, p_val = np.nan, np.nan
    else:
        t_stat, p_val = stats.ttest_rel(pairs["a"], pairs["b"], alternative="greater")
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    effect = float(np.mean(diff) / sd) if sd > 0 else float("inf")
    ci_low, ci_high = _bootstrap_ci(diff, bootstrap_iters=bootstrap_iters, seed=seed)
    return {
        "test": "scenario_level_paired_t_test_greater",
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "pairing_keys": keys,
        "aggregation": "mean_over_seeds_per_scenario",
        "effective_n": int(len(pairs)),
        "n_pairs": int(len(pairs)),
        "method_a_mean": float(pairs["a"].mean()),
        "method_a_std": float(pairs["a"].std(ddof=1)) if len(pairs) > 1 else 0.0,
        "method_b_mean": float(pairs["b"].mean()),
        "method_b_std": float(pairs["b"].std(ddof=1)) if len(pairs) > 1 else 0.0,
        "mean_diff": float(np.mean(diff)),
        "mean_diff_ci95": [ci_low, ci_high],
        "effect_size_cohen_dz": effect,
        "t_stat": None if np.isnan(t_stat) else float(t_stat),
        "p_value": None if np.isnan(p_val) else float(p_val),
        "significant_p_lt_0_05": bool(False if np.isnan(p_val) else p_val < 0.05),
        "naive_pooled_test_used": False,
    }


def grouped_paired_summary(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...] = ("dataset", "noise_type", "noise_rate"),
    metric: str = "macro_f1",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "overall": paired_summary(frame, metric=metric),
        "independence_aware": {},
        "groups": {},
        "comparisons": {},
        "multiple_comparison_correction": {"method": "holm_bonferroni", "family": "Graph-CoLD_vs_each_method"},
    }
    try:
        out["independence_aware"]["overall"] = scenario_level_paired_summary(frame, metric=metric)
    except ValueError as exc:
        out["independence_aware"]["overall"] = {"skipped": True, "reason": str(exc)}
    for method in sorted(set(frame.get("method", [])) - { "Graph-CoLD" }):
        try:
            comparison = paired_summary(
                frame,
                metric=metric,
                method_a="Graph-CoLD",
                method_b=method,
            )
            try:
                comparison["scenario_level"] = scenario_level_paired_summary(
                    frame,
                    metric=metric,
                    method_a="Graph-CoLD",
                    method_b=method,
                )
            except ValueError as exc:
                comparison["scenario_level"] = {"skipped": True, "reason": str(exc)}
            out["comparisons"][f"Graph-CoLD_vs_{method}"] = comparison
        except ValueError as exc:
            out["comparisons"][f"Graph-CoLD_vs_{method}"] = {"skipped": True, "reason": str(exc)}
    _apply_holm_bonferroni(out["comparisons"])
    cols = [col for col in group_cols if col in frame.columns]
    for key, part in frame.groupby(cols, dropna=False):
        label = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
        try:
            out["groups"][label] = paired_summary(part, metric=metric)
        except ValueError as exc:
            out["groups"][label] = {"skipped": True, "reason": str(exc)}
    return out


def _apply_holm_bonferroni(comparisons: dict[str, Any]) -> None:
    valid: list[tuple[str, float]] = []
    for name, info in comparisons.items():
        if info.get("skipped"):
            continue
        scenario = info.get("scenario_level", {})
        p_value = scenario.get("p_value", info.get("p_value"))
        if p_value is None or not np.isfinite(float(p_value)):
            continue
        valid.append((name, float(p_value)))
    m = len(valid)
    if m == 0:
        return
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, p_value) in enumerate(ordered, start=1):
        holm = min(1.0, (m - rank + 1) * p_value)
        running = max(running, holm)
        adjusted[name] = running
    for name, p_adj in adjusted.items():
        comparisons[name]["p_value_holm"] = float(p_adj)
        comparisons[name]["significant_holm_0_05"] = bool(p_adj < 0.05)
        if isinstance(comparisons[name].get("scenario_level"), dict):
            comparisons[name]["scenario_level"]["p_value_holm"] = float(p_adj)
            comparisons[name]["scenario_level"]["significant_holm_0_05"] = bool(p_adj < 0.05)


def _bootstrap_ci(diff: np.ndarray, bootstrap_iters: int = 2000, seed: int = 42) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64)
    if diff.size == 0:
        return 0.0, 0.0
    if diff.size == 1 or bootstrap_iters <= 0:
        value = float(np.mean(diff))
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(int(bootstrap_iters), dtype=np.float64)
    for idx in range(int(bootstrap_iters)):
        sample = rng.choice(diff, size=diff.size, replace=True)
        means[idx] = float(np.mean(sample))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


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
