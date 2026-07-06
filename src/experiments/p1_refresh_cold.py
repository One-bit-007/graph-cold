"""Refresh CoLD rows after the independent hard-purifier correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.result_sanity import check_results
from src.analysis.stat_tests import grouped_paired_summary
from src.experiments import cicids_mini_matrix, d5
from src.models.evidence import compute as compute_evidence


def refresh_cold_rows(out_dir: str | Path = "results", configs_dir: str | Path = "configs", reports_dir: str | Path = "reports") -> dict:
    out = Path(out_dir)
    configs = Path(configs_dir)
    reports = Path(reports_dir)
    main_path = out / "table_main.csv"
    expanded_path = out / "table_main_expanded.csv"
    if not main_path.exists():
        raise FileNotFoundError("CoLD refresh requires results/table_main.csv.")

    main = pd.read_csv(main_path, keep_default_na=False)
    before = _method_summary(main)
    scale_policy = d5.write_scale_policy_report(reports)
    replacement_rows = []
    for dataset_name in d5.FORMAL_DATASETS:
        for seed in d5.SEEDS:
            bundle = d5._load_formal_dataset(dataset_name, seed, configs, scale_policy)
            anomaly = cicids_mini_matrix.smoke_realdata._feature_anomaly(bundle.dataset.X_train, bundle.dataset.y_train)
            evidence = compute_evidence(
                bundle.dataset.y_train,
                {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
                anomaly=anomaly,
            )
            graph_cache: dict[float, object] = {}
            for spec in d5._noise_specs():
                noisy, flip = d5._inject_noise(bundle.dataset, spec, seed, graph_cache)
                context = d5._graphcold_context(bundle, spec, seed, flip, evidence, graph_cache)
                row, _pred = d5._evaluate_method(bundle, spec, seed, "CoLD", noisy, flip, context, {})
                replacement_rows.append(row)

    refreshed = pd.DataFrame(replacement_rows, columns=d5.FIELDNAMES)
    main = _replace_method_rows(main, refreshed, method="CoLD", columns=d5.FIELDNAMES)
    main.to_csv(main_path, index=False)

    stat_tests = grouped_paired_summary(main, metric="macro_f1")
    (out / "stat_tests.json").write_text(json.dumps(stat_tests, indent=2), encoding="utf-8")
    (reports / "d5_statistical_validity_report.json").write_text(json.dumps(stat_tests, indent=2), encoding="utf-8")
    sanity = check_results(main)
    (reports / "d5_result_sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    if expanded_path.exists():
        expanded = pd.read_csv(expanded_path, keep_default_na=False)
        annotated = refreshed.copy()
        annotated["method_family"] = "cold"
        annotated["implementation_status"] = "reused_verified_d5"
        expanded = _replace_method_rows(expanded, annotated, method="CoLD", columns=expanded.columns)
        expanded.to_csv(expanded_path, index=False)
        expanded_stats = grouped_paired_summary(expanded, metric="macro_f1")
        (out / "stat_tests_baseline_expansion.json").write_text(json.dumps(expanded_stats, indent=2), encoding="utf-8")
        (reports / "d5_expanded_statistical_validity_report.json").write_text(json.dumps(expanded_stats, indent=2), encoding="utf-8")
        expanded_sanity = check_results(expanded)
        (reports / "d5_expanded_sanity_report.json").write_text(json.dumps(expanded_sanity, indent=2), encoding="utf-8")

    after = _method_summary(main)
    report = {
        "completed": True,
        "refreshed_method": "CoLD",
        "rows_refreshed": int(len(refreshed)),
        "before": before,
        "after": after,
        "sanity": sanity,
    }
    (reports / "p1_cold_refresh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "p1_cold_refresh_report.md").write_text(_report_md(report), encoding="utf-8")
    return report


def _replace_method_rows(frame: pd.DataFrame, replacement: pd.DataFrame, method: str, columns) -> pd.DataFrame:
    keep = frame[frame["method"] != method].copy()
    replacement = replacement.reindex(columns=columns)
    out = pd.concat([keep, replacement], ignore_index=True)
    sort_cols = [col for col in ["dataset", "noise_type", "noise_rate", "graph_beta", "seed", "method"] if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _method_summary(frame: pd.DataFrame) -> dict:
    noisy = frame[frame["noise_type"] != "clean"]
    summary = noisy.groupby("method")[["macro_f1", "err_final", "fnr"]].mean()
    return {method: {key: float(value) for key, value in row.items()} for method, row in summary.to_dict(orient="index").items()}


def _report_md(report: dict) -> str:
    lines = ["# P1 CoLD Refresh Report", "", f"- Completed: {report['completed']}", f"- Rows refreshed: {report['rows_refreshed']}", ""]
    for label in ("before", "after"):
        lines.append(f"## {label.title()}")
        for method, metrics in report[label].items():
            lines.append(f"- {method}: Macro-F1={metrics['macro_f1']:.6f}, ERR={metrics['err_final']:.6f}, FNR={metrics['fnr']:.6f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    print(json.dumps(refresh_cold_rows(args.out, args.configs, args.reports), indent=2))


if __name__ == "__main__":
    main()
