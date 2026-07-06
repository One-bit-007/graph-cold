"""Sanity checks for formal result tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HONEST_DATASETS = {"cicids2017", "cesnet_tls_year22", "unsw_nb15", "ustc_tfc2016"}
FORMAL_D5_DATASETS = {"cicids2017", "cesnet_tls_year22"}
FORMAL_D5_METHODS = {"Graph-CoLD", "CoLD", "ablation_hard"}
EXPANDED_D5_METHODS = {
    "Noisy-Supervised",
    "Confident-Learning",
    "CL-filtering",
    "Co-Teaching",
    "FINE",
    "Decoupling",
    "MCRe",
    "MORSE",
}
VALID_IMPLEMENTATION_STATUSES = {"reused_verified_d5", "verified_implementation"}
FORBIDDEN_RESULT_TERMS = ("synthetic", "fallback", "emulation", "dummy", "placeholder")
EXPECTED_ACTIVE_VIEWS = {
    "cicids2017": "host|ip|temporal",
    "cesnet_tls_year22": "ip|temporal",
}
VALID_VIEWS = {"host", "ip", "process", "temporal", "threat_intel"}


def check_results(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    checks = {
        "no_nan_inf": bool(not numeric.isna().any().any() and np.isfinite(numeric.to_numpy(dtype=float)).all()),
        "no_perfect_anomaly": _no_perfect_anomaly(frame),
        "ablation_hard_distinct_from_cold": _ablation_distinct(frame),
        "err_graphcold_gt_hard": _err_direction(frame),
        "beta0_matches_symmetric": _beta0_matches_symmetric(frame),
        "active_views_valid": _active_views_valid(frame),
        "active_views_match_dataset_contract": _active_views_match_contract(frame),
        "dataset_names_honest": _dataset_names_honest(frame),
        "maltls22_absent_unless_verified": _maltls_absent_or_verified(frame),
        "formal_d5_scope_only": _formal_scope_only(frame),
        "no_optc_rows": _no_optc_rows(frame),
        "no_fake_baseline_rows": _no_fake_baseline_rows(frame),
        "no_forbidden_result_strings": _no_forbidden_result_strings(frame),
        "implementation_status_valid": _implementation_status_valid(frame),
        "sample_policy_present": _column_present_and_nonempty(frame, "sample_policy"),
        "dataset_hash_present": _column_present_and_nonempty(frame, "dataset_hash"),
        "source_verified_true": _source_verified_true(frame),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "blocking_reasons": [name for name, ok in checks.items() if not ok],
    }


def _no_perfect_anomaly(frame: pd.DataFrame) -> bool:
    needed = {"macro_f1", "fpr", "fnr"}
    if not needed.issubset(frame.columns):
        return True
    perfect = (frame["macro_f1"] >= 0.999) & (frame["fpr"] <= 0.001) & (frame["fnr"] <= 0.001)
    return bool(not perfect.any())


def _ablation_distinct(frame: pd.DataFrame) -> bool:
    if "variant" in frame.columns:
        hard = frame[frame["variant"] == "ablation_hard"]
        cold = frame[frame.get("method", pd.Series(dtype=str)) == "CoLD"]
    else:
        hard = frame[frame.get("method", pd.Series(dtype=str)) == "ablation_hard"]
        cold = frame[frame.get("method", pd.Series(dtype=str)) == "CoLD"]
    if hard.empty or cold.empty or "macro_f1" not in frame.columns:
        return True
    return bool(abs(float(hard["macro_f1"].mean()) - float(cold["macro_f1"].mean())) >= 1e-6)


def _err_direction(frame: pd.DataFrame) -> bool:
    err_col = "err_final" if "err_final" in frame.columns else "err"
    if err_col not in frame.columns:
        return True
    method_col = "method" if "method" in frame.columns else "variant" if "variant" in frame.columns else None
    if method_col is None:
        return True
    noisy = frame[frame.get("noise_type", pd.Series(["noisy"] * len(frame))) != "clean"]
    graph = noisy[noisy[method_col].isin(["Graph-CoLD", "Graph-CoLD-full"])][err_col]
    hard = noisy[noisy[method_col].isin(["ablation_hard", "CoLD"])][err_col]
    if graph.empty or hard.empty:
        return True
    return bool(float(graph.mean()) > float(hard.mean()))


def _beta0_matches_symmetric(frame: pd.DataFrame) -> bool:
    beta_col = "graph_beta" if "graph_beta" in frame.columns else "beta"
    needed = {"noise_type", "noise_rate", beta_col, "method", "macro_f1"}
    if not needed.issubset(frame.columns):
        return True
    beta_values = pd.to_numeric(frame[beta_col], errors="coerce")
    graph0 = frame[(frame["noise_type"] == "graph_consistency") & (beta_values.fillna(-1) == 0)]
    sym = frame[frame["noise_type"] == "symmetric"]
    if graph0.empty or sym.empty:
        return True
    keys = ["noise_rate", "method"]
    if "dataset" in frame.columns:
        keys.insert(0, "dataset")
    if "seed" in frame.columns:
        keys.append("seed")
    merged = graph0.merge(sym, on=keys, suffixes=("_g0", "_sym"))
    if merged.empty:
        return True
    return bool((merged["macro_f1_g0"] - merged["macro_f1_sym"]).abs().mean() <= 0.05)


def _active_views_valid(frame: pd.DataFrame) -> bool:
    if "active_views" not in frame.columns:
        return True
    for value in frame["active_views"].dropna().astype(str):
        views = [part.strip() for part in value.replace("|", ",").split(",") if part.strip()]
        if any(view not in VALID_VIEWS for view in views):
            return False
    return True


def _active_views_match_contract(frame: pd.DataFrame) -> bool:
    if not {"dataset", "active_views"}.issubset(frame.columns):
        return True
    for dataset, expected in EXPECTED_ACTIVE_VIEWS.items():
        part = frame[frame["dataset"].astype(str) == dataset]
        if part.empty:
            continue
        values = {str(value) for value in part["active_views"].dropna().unique()}
        if values != {expected}:
            return False
    return True


def _dataset_names_honest(frame: pd.DataFrame) -> bool:
    if "dataset" not in frame.columns:
        return True
    datasets = {str(value) for value in frame["dataset"].dropna().unique()}
    forbidden = {"synthetic", "fallback", "emulation"}
    return bool(datasets.issubset(HONEST_DATASETS) and not datasets.intersection(forbidden))


def _formal_scope_only(frame: pd.DataFrame) -> bool:
    if "dataset" not in frame.columns:
        return True
    datasets = {str(value).lower() for value in frame["dataset"].dropna().unique()}
    return bool(datasets.issubset(FORMAL_D5_DATASETS))


def _no_optc_rows(frame: pd.DataFrame) -> bool:
    if "dataset" not in frame.columns:
        return True
    return "optc" not in {str(value).lower() for value in frame["dataset"].dropna().unique()}


def _no_fake_baseline_rows(frame: pd.DataFrame) -> bool:
    if "method" not in frame.columns:
        return True
    methods = {str(value) for value in frame["method"].dropna().unique()}
    if "implementation_status" not in frame.columns:
        return bool(methods.issubset(FORMAL_D5_METHODS))
    allowed = FORMAL_D5_METHODS | EXPANDED_D5_METHODS
    if not methods.issubset(allowed):
        return False
    expanded = frame[~frame["method"].isin(FORMAL_D5_METHODS)]
    if expanded.empty:
        return True
    statuses = {str(value) for value in expanded["implementation_status"].dropna().unique()}
    if not statuses.issubset({"verified_implementation"}):
        return False
    if "verified" in expanded.columns:
        verified_text = expanded["verified"].astype(str).str.strip().str.lower()
        requires_verified = verified_text.isin({"true", "false", "1", "0"})
        verified = expanded.loc[requires_verified, "verified"]
        if verified.empty:
            return True
        if verified.dtype == bool:
            if not bool(verified.all()):
                return False
        elif not bool(verified.astype(str).str.lower().isin({"true", "1"}).all()):
            return False
    if "faithfulness_level" in expanded.columns:
        fine = expanded[expanded["method"] == "FINE"]
        if not fine.empty and not fine["faithfulness_level"].astype(str).str.contains("eigenvector", case=False, regex=False).all():
            return False
    return True


def _no_forbidden_result_strings(frame: pd.DataFrame) -> bool:
    object_cols = frame.select_dtypes(include=["object", "string"])
    for col in object_cols.columns:
        values = object_cols[col].dropna().astype(str).str.lower()
        if values.empty:
            continue
        if values.str.contains("|".join(FORBIDDEN_RESULT_TERMS), regex=True).any():
            return False
    return True


def _implementation_status_valid(frame: pd.DataFrame) -> bool:
    if "implementation_status" not in frame.columns:
        return True
    statuses = {str(value) for value in frame["implementation_status"].dropna().unique()}
    return bool(statuses.issubset(VALID_IMPLEMENTATION_STATUSES))


def _column_present_and_nonempty(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column]
    return bool(values.notna().all() and (values.astype(str).str.len() > 0).all())


def _source_verified_true(frame: pd.DataFrame) -> bool:
    if "source_verified" not in frame.columns:
        return False
    values = frame["source_verified"]
    if values.dtype == bool:
        return bool(values.all())
    return bool(values.astype(str).str.lower().isin({"true", "1"}).all())


def _maltls_absent_or_verified(frame: pd.DataFrame) -> bool:
    if "dataset" not in frame.columns:
        return True
    if "maltls22" not in {str(value).lower() for value in frame["dataset"].dropna().unique()}:
        return True
    if "source_verified" not in frame.columns:
        return False
    return bool(frame[frame["dataset"].astype(str).str.lower() == "maltls22"]["source_verified"].all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--input", dest="input_csv")
    parser.add_argument("--out")
    args = parser.parse_args()
    csv_path = args.input_csv or args.csv
    if not csv_path:
        raise SystemExit("Provide a result CSV path as positional csv or --input.")
    report = check_results(pd.read_csv(csv_path))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
