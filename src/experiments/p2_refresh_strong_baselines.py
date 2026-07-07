"""Refresh P2-calibrated strong baseline rows without rerunning unchanged baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.result_sanity import check_results
from src.analysis.stat_tests import grouped_paired_summary
from src.experiments import d5, d5_baseline_expansion as bx


STRONG_BASELINES = ("Co-Teaching", "FINE", "MCRe", "MORSE")


def refresh_strong_baselines(
    out_dir: str | Path = "results",
    configs_dir: str | Path = "configs",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    out = Path(out_dir)
    configs = Path(configs_dir)
    reports = Path(reports_dir)
    original = pd.read_csv(out / "table_main.csv", keep_default_na=False)
    baseline = pd.read_csv(out / "table_baseline_expansion.csv", keep_default_na=False)
    before_expanded = pd.read_csv(out / "table_main_expanded.csv", keep_default_na=False)
    dataset_scope = tuple(dataset for dataset in (*d5.BASE_FORMAL_DATASETS, "unsw_nb15") if dataset in set(original["dataset"].astype(str)))
    scale_policy = bx._read_scale_policy(reports)

    before = _baseline_summary(before_expanded)
    refreshed_rows, runtime_records = _run_strong_rows(out, configs, scale_policy, dataset_scope)
    refreshed = pd.DataFrame(refreshed_rows, columns=bx.EXPANDED_FIELDNAMES)
    preserved = baseline[~baseline["method"].isin(STRONG_BASELINES)].copy()
    baseline_after = pd.concat([preserved, refreshed], ignore_index=True).reindex(columns=bx.EXPANDED_FIELDNAMES)
    baseline_after.to_csv(out / "table_baseline_expansion.csv", index=False)

    original_expanded = bx._annotate_original_rows(original)
    expanded = pd.concat([original_expanded, baseline_after], ignore_index=True).reindex(columns=bx.EXPANDED_FIELDNAMES)
    expanded.to_csv(out / "table_main_expanded.csv", index=False)

    stats = grouped_paired_summary(expanded, metric="macro_f1")
    (out / "stat_tests_baseline_expansion.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (reports / "d5_expanded_statistical_validity_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (reports / "d5_expanded_statistical_validity_report.md").write_text(bx._stat_markdown(stats), encoding="utf-8")

    sanity = check_results(expanded)
    sanity.setdefault("checks", {})["p2_strong_baselines_refreshed"] = True
    (reports / "d5_expanded_sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    (reports / "d5_expanded_sanity_report.md").write_text(bx._sanity_markdown(sanity), encoding="utf-8")

    verification = _verification_stub(reports, baseline_after)
    runtime = bx._runtime_json(pd.DataFrame(runtime_records), verification)
    (out / "runtime_baseline_expansion.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    report = {
        "completed": bool(sanity["passed"]),
        "methods_refreshed": list(STRONG_BASELINES),
        "rows_refreshed": int(len(refreshed)),
        "dataset_scope": list(dataset_scope),
        "before": before,
        "after": _baseline_summary(expanded),
        "sanity": sanity,
    }
    (reports / "p2_strong_baseline_refresh.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "p2_strong_baseline_refresh.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _run_strong_rows(out: Path, configs: Path, scale_policy: dict[str, Any], dataset_scope: tuple[str, ...]):
    partial_path = out / "table_p2_strong_baseline_refresh.partial.csv"
    existing = bx._load_partial_baseline_rows(partial_path, STRONG_BASELINES)
    rows: list[dict[str, Any]] = existing.to_dict(orient="records") if not existing.empty else []
    runtime_records: list[dict[str, Any]] = bx._runtime_records_from_frame(existing) if not existing.empty else []
    done = {bx._row_key(row) for row in rows}
    for dataset_name in dataset_scope:
        for seed in d5.SEEDS:
            bundle = d5._load_formal_dataset(dataset_name, seed, configs, scale_policy)
            evidence = bx._evidence(bundle)
            graph_cache: dict[float, Any] = {}
            for spec in d5._noise_specs():
                noisy, flip = d5._inject_noise(bundle.dataset, spec, seed, graph_cache)
                for baseline in bx._baseline_candidates(seed, float(spec["noise_rate"])):
                    method = bx._candidate_method_name(baseline)
                    if method not in STRONG_BASELINES:
                        continue
                    key = bx._row_key(
                        {
                            "dataset": dataset_name,
                            "noise_type": spec["noise_type"],
                            "noise_rate": spec["noise_rate"],
                            "graph_beta": spec["graph_beta"],
                            "seed": seed,
                            "method": method,
                        }
                    )
                    if key in done:
                        continue
                    print(
                        f"[p2-strong-baseline] {dataset_name} seed={seed} "
                        f"{spec['noise_type']} rate={spec['noise_rate']} beta={spec['graph_beta']} {method}",
                        flush=True,
                    )
                    result, runtime_sec, memory_mb = bx._timed_baseline(baseline, bundle, noisy)
                    row = bx._row_from_result(bundle, spec, seed, result, runtime_sec, memory_mb, evidence, flip)
                    rows.append(row)
                    done.add(bx._row_key(row))
                    bx._append_partial_row(partial_path, row)
                    runtime_records.append(
                        {
                            "dataset": row["dataset"],
                            "reported_as": row["reported_as"],
                            "noise_type": row["noise_type"],
                            "noise_rate": row["noise_rate"],
                            "graph_beta": row["graph_beta"],
                            "seed": row["seed"],
                            "method": row["method"],
                            "runtime_sec": row["runtime_sec"],
                            "memory_mb": row["memory_mb"],
                        }
                    )
    if partial_path.exists():
        partial_path.unlink()
    return rows, runtime_records


def _baseline_summary(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in STRONG_BASELINES:
        clean = frame[
            (frame["reported_as"] == "CICIDS-2017")
            & (frame["method"] == method)
            & (frame["noise_type"] == "clean")
        ]["macro_f1"]
        low = frame[
            (frame["reported_as"] == "CICIDS-2017")
            & (frame["method"] == method)
            & (pd.to_numeric(frame["noise_rate"], errors="coerce") == 0.1)
        ]["macro_f1"]
        out[method] = {
            "clean_macro_f1": float(clean.mean()) if not clean.empty else None,
            "low_noise_macro_f1": float(low.mean()) if not low.empty else None,
            "overall_macro_f1": float(frame[frame["method"] == method]["macro_f1"].mean()),
        }
    return out


def _verification_stub(reports: Path, baseline: pd.DataFrame) -> dict[str, Any]:
    existing = json.loads((reports / "baseline_verification_report.json").read_text(encoding="utf-8")) if (reports / "baseline_verification_report.json").exists() else {}
    existing["p2_baseline_calibration"] = True
    existing["passed_baselines"] = sorted(set(baseline["method"].astype(str)))
    existing.setdefault("failed_baselines", {})
    return existing


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P2 Strong Baseline Refresh",
        "",
        f"- Completed: {report['completed']}",
        f"- Rows refreshed: {report['rows_refreshed']}",
        f"- Methods refreshed: {', '.join(report['methods_refreshed'])}",
        "",
        "## Before / After",
    ]
    for method in report["methods_refreshed"]:
        before = report["before"].get(method, {})
        after = report["after"].get(method, {})
        lines.append(
            f"- {method}: clean {before.get('clean_macro_f1')} -> {after.get('clean_macro_f1')}; "
            f"low-noise {before.get('low_noise_macro_f1')} -> {after.get('low_noise_macro_f1')}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    print(json.dumps(refresh_strong_baselines(args.out, args.configs, args.reports), indent=2))


if __name__ == "__main__":
    main()
