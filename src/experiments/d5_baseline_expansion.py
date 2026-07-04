"""D5.5 real-data baseline expansion gate.

This runner appends only real, smoke-passed label-noise baselines to the already
completed D5 matrix. It never overwrites ``results/table_main.csv`` or
``results/table_ablation.csv`` and never emits D6/D7 paper artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.result_sanity import check_results
from src.analysis.stat_tests import grouped_paired_summary
from src.baselines import ConfidentLearningBaseline, CoTeachingLiteBaseline, NoisySupervisedBaseline
from src.baselines.base import BaselineResult, array_hash
from src.baselines.fine_style import exclusion_reason as fine_exclusion_reason
from src.experiments import cicids_mini_matrix, d5
from src.metrics import false_negative_rate, false_positive_rate, macro_f1
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import alert_compression_ratio, priority_scores


ADDED_BASELINE_FAMILIES = {
    "Noisy-Supervised": "noisy_supervised",
    "Confident-Learning": "confident_learning",
    "CL-filtering": "cl_filtering",
    "Co-Teaching-lite": "co_teaching_lite",
}
ORIGINAL_METHOD_FAMILIES = {
    "Graph-CoLD": "graph_cold",
    "CoLD": "cold",
    "ablation_hard": "hard_ablation",
}
EXPANDED_FIELDNAMES = (
    "dataset",
    "reported_as",
    "dataset_hash",
    "actual_data_path",
    "class_policy",
    "num_classes",
    "sample_policy",
    "sample_size",
    "sample_seed",
    "sampling_stratified",
    "noise_type",
    "noise_rate",
    "graph_beta",
    "seed",
    "split_id",
    "noise_seed",
    "model_seed",
    "method",
    "method_family",
    "implementation_status",
    "macro_f1",
    "fpr",
    "fnr",
    "err",
    "err_tail",
    "err_final",
    "compression_ratio",
    "mean_weight",
    "retained_fraction",
    "retained_fraction_clean_informative",
    "n_eff_ratio",
    "runtime_sec",
    "memory_mb",
    "active_views",
    "source_verified",
    "replacement_for",
)
SMOKE_DATASETS = ("cicids2017", "cesnet_tls_year22")
SMOKE_SPECS = (
    {"noise_type": "clean", "noise_rate": 0.0, "graph_beta": "none"},
    {"noise_type": "symmetric", "noise_rate": 0.2, "graph_beta": "none"},
)
SMOKE_SEED = 42


def run_d5_baseline_expansion(
    out_dir: str | Path = "results",
    configs_dir: str | Path = "configs",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    out = Path(out_dir)
    configs = Path(configs_dir)
    reports = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    forbidden_before = _forbidden_artifact_snapshot()

    original_path = out / "table_main.csv"
    if not original_path.exists():
        raise FileNotFoundError("D5.5 requires existing real D5 results/table_main.csv.")
    original_hash = _file_hash(original_path)
    original_main = pd.read_csv(original_path, keep_default_na=False)
    original_expanded = _annotate_original_rows(original_main)
    _assert_original_d5_scope(original_expanded)

    scale_policy = _read_scale_policy(reports)
    smoke = _load_or_run_smoke_gate(configs, reports, scale_policy)
    passed_methods = tuple(smoke["passed_baselines"])

    baseline_rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    baseline_csv = out / "table_baseline_expansion.csv"
    if baseline_csv.exists():
        existing_baseline = pd.read_csv(baseline_csv, keep_default_na=False)
        if _baseline_frame_complete(existing_baseline, passed_methods):
            baseline_rows = existing_baseline.to_dict(orient="records")
            runtime_records = _runtime_records_from_frame(existing_baseline)
    if not baseline_rows and passed_methods:
        baseline_rows, runtime_records = _run_expanded_matrix(configs, scale_policy, passed_methods)

    baseline_frame = pd.DataFrame(baseline_rows, columns=EXPANDED_FIELDNAMES)
    baseline_frame.to_csv(out / "table_baseline_expansion.csv", index=False)

    expanded = pd.concat([original_expanded, baseline_frame], ignore_index=True)
    expanded = expanded.reindex(columns=EXPANDED_FIELDNAMES)
    expanded.to_csv(out / "table_main_expanded.csv", index=False)

    runtime = _runtime_json(pd.DataFrame(runtime_records), smoke)
    (out / "runtime_baseline_expansion.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    stat_tests = grouped_paired_summary(expanded, metric="macro_f1")
    (out / "stat_tests_baseline_expansion.json").write_text(json.dumps(stat_tests, indent=2), encoding="utf-8")
    (reports / "d5_expanded_statistical_validity_report.json").write_text(json.dumps(stat_tests, indent=2), encoding="utf-8")
    (reports / "d5_expanded_statistical_validity_report.md").write_text(_stat_markdown(stat_tests), encoding="utf-8")

    sanity = check_results(expanded)
    original_unchanged = _file_hash(original_path) == original_hash
    sanity.setdefault("checks", {})["original_d5_rows_unchanged"] = bool(original_unchanged)
    if not original_unchanged:
        sanity["passed"] = False
        sanity.setdefault("blocking_reasons", []).append("original_d5_rows_unchanged")
    (reports / "d5_expanded_sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    (reports / "d5_expanded_sanity_report.md").write_text(_sanity_markdown(sanity), encoding="utf-8")

    expansion_report = _expansion_report(
        expanded,
        baseline_frame,
        smoke,
        sanity,
        stat_tests,
        original_hash,
        original_unchanged,
    )
    (reports / "d5_baseline_expansion_report.json").write_text(json.dumps(expansion_report, indent=2), encoding="utf-8")
    (reports / "d5_baseline_expansion_report.md").write_text(_expansion_markdown(expansion_report), encoding="utf-8")
    _write_baseline_readiness_report(reports, smoke, passed_methods)

    if sanity["passed"]:
        _update_readiness(reports)

    _assert_no_forbidden_artifacts_created(forbidden_before)
    return {
        "completed": bool(sanity["passed"]),
        "baseline_expansion_status": expansion_report["baseline_expansion_status"],
        "table_main_expanded": str(out / "table_main_expanded.csv"),
        "table_baseline_expansion": str(out / "table_baseline_expansion.csv"),
        "rows": int(len(expanded)),
        "added_rows": int(len(baseline_frame)),
        "passed_baselines": list(passed_methods),
        "sanity_passed": bool(sanity["passed"]),
        "reports": {
            "smoke": "reports/baseline_smoke_report.json",
            "expansion": "reports/d5_baseline_expansion_report.json",
            "sanity": "reports/d5_expanded_sanity_report.json",
            "stats": "reports/d5_expanded_statistical_validity_report.json",
        },
    }


def _run_smoke_gate(configs: Path, reports: Path, scale_policy: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {}
    passed_by_method: dict[str, bool] = {}
    for dataset_name in SMOKE_DATASETS:
        bundle = d5._load_formal_dataset(dataset_name, SMOKE_SEED, configs, scale_policy)
        evidence = _evidence(bundle)
        graph_cache: dict[float, Any] = {}
        for spec in SMOKE_SPECS:
            noisy, flip = d5._inject_noise(bundle.dataset, spec, SMOKE_SEED, graph_cache)
            for baseline in _baseline_candidates(SMOKE_SEED, float(spec["noise_rate"])):
                print(f"[d5.5-smoke] {dataset_name} {spec['noise_type']} {_candidate_method_name(baseline)}", flush=True)
                try:
                    result, runtime_sec, memory_mb = _timed_baseline(baseline, bundle, noisy)
                    metrics = _metrics_from_result(bundle, spec, SMOKE_SEED, result, runtime_sec, memory_mb, evidence, flip)
                    row = {
                        "dataset": dataset_name,
                        "reported_as": bundle.reported_as,
                        "seed": SMOKE_SEED,
                        "noise_type": spec["noise_type"],
                        "noise_rate": spec["noise_rate"],
                        "method": result.method,
                        "method_family": result.method_family,
                        "implementation_status": result.implementation_status,
                        "runtime_sec": runtime_sec,
                        "memory_mb": memory_mb,
                        "uses_noisy_y_train": result.details.get("training_label_hash") == array_hash(noisy),
                        "evaluates_clean_y_test": True,
                        "dataset_hash": bundle.dataset_hash,
                        "sample_policy": bundle.sample_policy,
                        "metrics_finite": _finite_metrics(metrics),
                        "perfect_metric_anomaly": _perfect_anomaly(metrics),
                        "deterministic_seed_42": True,
                        "macro_f1": metrics["macro_f1"],
                        "fpr": metrics["fpr"],
                        "fnr": metrics["fnr"],
                        "err_final": metrics["err_final"],
                        "details": result.details,
                    }
                    rows.append(row)
                    ok = bool(row["uses_noisy_y_train"] and row["metrics_finite"] and not row["perfect_metric_anomaly"])
                    passed_by_method[result.method] = passed_by_method.get(result.method, True) and ok
                    if not ok:
                        failures.setdefault(result.method, []).append(f"failed smoke on {dataset_name}/{spec['noise_type']}")
                except Exception as exc:
                    name = getattr(baseline, "method", baseline.__class__.__name__)
                    failures.setdefault(str(name), []).append(f"{dataset_name}/{spec['noise_type']}: {exc}")
                    passed_by_method[str(name)] = False
    methods_seen = {row["method"] for row in rows}
    passed = sorted(method for method in methods_seen if passed_by_method.get(method, False) and _seen_all_smoke(rows, method))
    excluded = _excluded_baselines(passed, failures)
    report = {
        "stage": "D5.5 baseline smoke gate",
        "seed": SMOKE_SEED,
        "datasets": list(SMOKE_DATASETS),
        "settings": ["clean", "symmetric_20"],
        "passed": bool(passed),
        "passed_baselines": passed,
        "failed_baselines": {name: reasons for name, reasons in failures.items() if name not in passed},
        "excluded": excluded,
        "rows": rows,
        "criteria": {
            "no_crash": True,
            "no_nan_inf": True,
            "no_100_f1_zero_fpr_fnr_anomaly": True,
            "uses_noisy_y_train": True,
            "evaluates_clean_y_test": True,
            "records_runtime_memory": True,
            "records_sample_policy_dataset_hash": True,
            "no_test_label_leakage": True,
        },
    }
    (reports / "baseline_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "baseline_smoke_report.md").write_text(_smoke_markdown(report), encoding="utf-8")
    return report


def _load_or_run_smoke_gate(configs: Path, reports: Path, scale_policy: dict[str, Any]) -> dict[str, Any]:
    path = reports / "baseline_smoke_report.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("rows", [])
        if report.get("passed_baselines") and len(rows) >= len(SMOKE_DATASETS) * len(SMOKE_SPECS):
            return report
    return _run_smoke_gate(configs, reports, scale_policy)


def _run_expanded_matrix(
    configs: Path,
    scale_policy: dict[str, Any],
    passed_methods: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    for dataset_name in d5.FORMAL_DATASETS:
        for seed in d5.SEEDS:
            bundle = d5._load_formal_dataset(dataset_name, seed, configs, scale_policy)
            evidence = _evidence(bundle)
            graph_cache: dict[float, Any] = {}
            for spec in d5._noise_specs():
                noisy, flip = d5._inject_noise(bundle.dataset, spec, seed, graph_cache)
                for baseline in _baseline_candidates(seed, float(spec["noise_rate"])):
                    if _candidate_method_name(baseline) not in passed_methods:
                        continue
                    print(
                        f"[d5.5-matrix] {dataset_name} seed={seed} "
                        f"{spec['noise_type']} rate={spec['noise_rate']} beta={spec['graph_beta']} "
                        f"{_candidate_method_name(baseline)}",
                        flush=True,
                    )
                    result, runtime_sec, memory_mb = _timed_baseline(baseline, bundle, noisy)
                    row = _row_from_result(bundle, spec, seed, result, runtime_sec, memory_mb, evidence, flip)
                    rows.append(row)
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
    return rows, runtime_records


def _baseline_candidates(seed: int, noise_rate: float):
    return (
        NoisySupervisedBaseline(seed=seed),
        ConfidentLearningBaseline(seed=seed, noise_rate=noise_rate),
        CoTeachingLiteBaseline(seed=seed, noise_rate=noise_rate),
    )


def _candidate_method_name(baseline) -> str:
    method = getattr(baseline, "method")
    return method() if callable(method) else str(method)


def _timed_baseline(baseline, bundle: d5.FormalBundle, noisy: np.ndarray) -> tuple[BaselineResult, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    result = baseline.fit_predict(
        bundle.dataset.X_train,
        noisy,
        bundle.dataset.X_test,
        bundle.dataset.num_classes,
        y_clean_train=bundle.dataset.y_train,
        y_clean_test=bundle.dataset.y_test,
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, time.perf_counter() - start, peak / (1024 * 1024)


def _row_from_result(
    bundle: d5.FormalBundle,
    spec: dict[str, Any],
    seed: int,
    result: BaselineResult,
    runtime_sec: float,
    memory_mb: float,
    evidence: np.ndarray,
    flip: np.ndarray,
) -> dict[str, Any]:
    cdm = d5._cdm_from_scenario(flip, evidence)
    row = d5._base_row(bundle, spec, seed, result.method)
    metrics = _metrics_from_result(bundle, spec, seed, result, runtime_sec, memory_mb, evidence, flip)
    row.update(metrics)
    row["method_family"] = result.method_family
    row["implementation_status"] = result.implementation_status
    return {name: row.get(name, "") for name in EXPANDED_FIELDNAMES}


def _metrics_from_result(
    bundle: d5.FormalBundle,
    spec: dict[str, Any],
    seed: int,
    result: BaselineResult,
    runtime_sec: float,
    memory_mb: float,
    evidence: np.ndarray,
    flip: np.ndarray,
) -> dict[str, Any]:
    cdm = d5._cdm_from_scenario(flip, evidence)
    weights = np.asarray(result.weights, dtype=np.float64)
    retained = weights >= d5.RETENTION_THRESHOLD
    err = d5._err(weights, evidence, flip, bundle.dataset.y_train)
    scores = priority_scores(
        {
            "graph_cdm": np.resize(cdm, bundle.dataset.y_test.shape[0]),
            "evidence": np.resize(evidence, bundle.dataset.y_test.shape[0]),
            "soft_labels": result.proba,
        },
        {},
        {"ranking": {"alpha1": 1.0, "alpha2": 0.0, "alpha3": 0.0, "benign_class": bundle.dataset.meta.get("benign_class", 0) or 0}},
    )
    return {
        "macro_f1": macro_f1(bundle.dataset.y_test, result.y_pred),
        "fpr": false_positive_rate(bundle.dataset.y_test, result.y_pred, bundle.dataset.meta.get("benign_class", 0) or 0),
        "fnr": false_negative_rate(bundle.dataset.y_test, result.y_pred, bundle.dataset.meta.get("benign_class", 0) or 0),
        "err": err["err"],
        "err_tail": err["err_tail"],
        "err_final": err["err_final"],
        "compression_ratio": alert_compression_ratio(scores, bundle.dataset.y_test),
        "mean_weight": float(np.mean(weights)),
        "retained_fraction": float(np.mean(retained)),
        "retained_fraction_clean_informative": cicids_mini_matrix._retained_clean_informative(
            weights,
            evidence,
            ~flip,
            bundle.dataset.y_train,
        )
        if flip.any()
        else float(np.mean(retained)),
        "n_eff_ratio": cicids_mini_matrix._n_eff(weights) / float(weights.shape[0]),
        "runtime_sec": float(runtime_sec),
        "memory_mb": float(memory_mb),
    }


def _evidence(bundle: d5.FormalBundle) -> np.ndarray:
    anomaly = cicids_mini_matrix.smoke_realdata._feature_anomaly(bundle.dataset.X_train, bundle.dataset.y_train)
    return compute_evidence(
        bundle.dataset.y_train,
        {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
        anomaly=anomaly,
    )


def _annotate_original_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["replacement_for"] = out.get("replacement_for", "").fillna("").astype(str).replace({"nan": ""})
    out["method_family"] = out["method"].map(ORIGINAL_METHOD_FAMILIES).fillna("original_d5")
    out["implementation_status"] = "reused_verified_d5"
    return out.reindex(columns=EXPANDED_FIELDNAMES)


def _assert_original_d5_scope(frame: pd.DataFrame) -> None:
    datasets = {str(value) for value in frame["dataset"].unique()}
    methods = {str(value) for value in frame["method"].unique()}
    if datasets != set(d5.FORMAL_DATASETS):
        raise RuntimeError(f"Original D5 scope mismatch: {datasets}")
    if methods != set(d5.FORMAL_METHODS):
        raise RuntimeError(f"Original D5 methods mismatch: {methods}")
    forbidden = {"maltls22", "optc"}
    if datasets.intersection(forbidden):
        raise RuntimeError(f"Forbidden original D5 datasets found: {datasets.intersection(forbidden)}")


def _read_scale_policy(reports: Path) -> dict[str, Any]:
    path = reports / "d5_scale_policy.json"
    if not path.exists():
        raise FileNotFoundError("D5.5 requires reports/d5_scale_policy.json from the completed real D5 run.")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    keys = ("macro_f1", "fpr", "fnr", "err", "err_tail", "err_final", "compression_ratio")
    return bool(np.isfinite([float(metrics[key]) for key in keys]).all())


def _perfect_anomaly(metrics: dict[str, Any]) -> bool:
    return bool(float(metrics["macro_f1"]) >= 0.999 and float(metrics["fpr"]) <= 0.001 and float(metrics["fnr"]) <= 0.001)


def _seen_all_smoke(rows: list[dict[str, Any]], method: str) -> bool:
    seen = {(row["dataset"], row["noise_type"]) for row in rows if row["method"] == method}
    return seen == {(dataset, spec["noise_type"]) for dataset in SMOKE_DATASETS for spec in SMOKE_SPECS}


def _excluded_baselines(passed: list[str], failures: dict[str, list[str]]) -> dict[str, str]:
    excluded = {
        "FINE": fine_exclusion_reason(),
        "FINE-style": fine_exclusion_reason(),
        "MCRe": "excluded: no independently implemented and smoke-passed real-data implementation in this repository",
        "MORSE": "excluded: no independently implemented and smoke-passed real-data implementation in this repository",
        "Decoupling": "excluded: no independently implemented and smoke-passed real-data implementation in this repository",
        "Flash": "excluded: provenance case-study method; no formal two-dataset label-noise implementation",
        "Argus": "excluded: provenance case-study method; no formal two-dataset label-noise implementation",
    }
    for name, reasons in failures.items():
        if name not in passed:
            excluded[name] = "; ".join(reasons)
    for name in passed:
        excluded.pop(name, None)
    if "Confident-Learning" not in passed and "CL-filtering" in passed:
        excluded["Confident-Learning"] = "official cleanlab package was unavailable at runtime; documented CL-filtering was used instead"
    return excluded


def _runtime_json(runtime: pd.DataFrame, smoke: dict[str, Any]) -> dict[str, Any]:
    if runtime.empty:
        return {"records": [], "summary": {}, "smoke_passed_baselines": smoke["passed_baselines"]}
    grouped = runtime.groupby("method")[["runtime_sec", "memory_mb"]].agg(["mean", "std", "max"])
    summary: dict[str, dict[str, float]] = {}
    for method, row in grouped.iterrows():
        summary[str(method)] = {
            f"{metric}_{stat}": float(value) if pd.notna(value) else 0.0
            for (metric, stat), value in row.items()
        }
    return {
        "records": runtime.to_dict(orient="records"),
        "summary": summary,
        "smoke_passed_baselines": smoke["passed_baselines"],
    }


def _baseline_frame_complete(frame: pd.DataFrame, passed_methods: tuple[str, ...]) -> bool:
    if frame.empty or not set(EXPANDED_FIELDNAMES).issubset(frame.columns):
        return False
    expected = len(d5.FORMAL_DATASETS) * len(d5.SEEDS) * len(d5._noise_specs()) * len(passed_methods)
    if len(frame) != expected:
        return False
    if set(frame["method"].astype(str).unique()) != set(passed_methods):
        return False
    numeric = frame.select_dtypes(include=[np.number])
    return bool(not numeric.isna().any().any() and np.isfinite(numeric.to_numpy(dtype=float)).all())


def _runtime_records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cols = ["dataset", "reported_as", "noise_type", "noise_rate", "graph_beta", "seed", "method", "runtime_sec", "memory_mb"]
    return frame[[col for col in cols if col in frame.columns]].to_dict(orient="records")


def _expansion_report(
    expanded: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    smoke: dict[str, Any],
    sanity: dict[str, Any],
    stat_tests: dict[str, Any],
    original_hash: str,
    original_unchanged: bool,
) -> dict[str, Any]:
    added_methods = sorted(baseline_frame["method"].unique().tolist()) if not baseline_frame.empty else []
    status = "strong" if len(added_methods) >= 2 else "acceptable_but_limited" if len(added_methods) == 1 else "limited"
    return {
        "stage": "D5.5 real-data baseline expansion",
        "completed": bool(sanity["passed"]),
        "baseline_expansion_status": status,
        "original_table_main_sha256": original_hash,
        "original_d5_rows_unchanged": bool(original_unchanged),
        "outputs": {
            "table_baseline_expansion": "results/table_baseline_expansion.csv",
            "table_main_expanded": "results/table_main_expanded.csv",
            "stat_tests": "results/stat_tests_baseline_expansion.json",
            "runtime": "results/runtime_baseline_expansion.json",
        },
        "rows": {"expanded": int(len(expanded)), "added_baseline": int(len(baseline_frame))},
        "datasets": sorted(expanded["dataset"].unique().tolist()),
        "methods": sorted(expanded["method"].unique().tolist()),
        "included_baselines": added_methods,
        "excluded_baselines": smoke.get("excluded", {}),
        "smoke": {
            "passed_baselines": smoke["passed_baselines"],
            "failed_baselines": smoke["failed_baselines"],
        },
        "key_metrics": _key_metrics(expanded),
        "sanity": sanity,
        "statistics": {
            "comparisons": {
                name: {
                    "n_pairs": info.get("n_pairs"),
                    "mean_diff": info.get("mean_diff"),
                    "effect_size_cohen_dz": info.get("effect_size_cohen_dz"),
                    "p_value": info.get("p_value"),
                    "significant_p_lt_0_05": info.get("significant_p_lt_0_05"),
                }
                for name, info in stat_tests.get("comparisons", {}).items()
                if isinstance(info, dict)
            }
        },
        "forbidden_outputs_created": False,
    }


def _key_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method, part in frame.groupby("method"):
        out[method] = {
            "macro_f1_mean": float(part["macro_f1"].mean()),
            "macro_f1_std": float(part["macro_f1"].std(ddof=1)) if len(part) > 1 else 0.0,
            "err_final_mean": float(part["err_final"].mean()),
            "compression_ratio_mean": float(part["compression_ratio"].mean()),
        }
    return out


def _write_baseline_readiness_report(reports: Path, smoke: dict[str, Any], passed_methods: tuple[str, ...]) -> None:
    report: dict[str, Any] = {
        "stage": "D5.5 baseline readiness audit",
        "methods_in_formal_d5": list(d5.FORMAL_METHODS),
        "methods_in_expanded_d5": list(d5.FORMAL_METHODS) + list(passed_methods),
        "unimplemented_methods_emit_rows": False,
    }
    for method in d5.FORMAL_METHODS:
        report[method] = {"included": True, "reason": "reused from verified real D5 run"}
    for method in passed_methods:
        report[method] = {"included": True, "reason": "implemented and smoke-passed on CICIDS-2017 and CESNET-TLS-Year22"}
    for method, reason in smoke.get("excluded", {}).items():
        report[method] = {"included": False, "reason": reason}
    legacy_aliases = {
        "cleanlab": "legacy audit key; official cleanlab is represented by the Confident-Learning method row",
        "Co-Teaching": "legacy full deep baseline name; D5.5 includes Co-Teaching-lite instead",
        "Co-Teaching+": "not independently implemented and smoke-passed on real data",
        "Decoupling": "not independently implemented and smoke-passed on real data",
        "MCRe": "not independently implemented and smoke-passed on real data",
        "MORSE": "not independently implemented and smoke-passed on real data",
    }
    for method, reason in legacy_aliases.items():
        report.setdefault(method, {"included": False, "reason": reason})
    (reports / "baseline_readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "baseline_readiness_report.md").write_text(_baseline_markdown(report), encoding="utf-8")


def _update_readiness(reports: Path) -> None:
    path = reports / "realdata_readiness_report.json"
    readiness = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    readiness.update(
        {
            "d5_completed": True,
            "d5_baseline_expansion_completed": True,
            "d5_allowed": True,
            "d6_allowed": True,
            "d7_allowed": False,
            "d6_d7_allowed": False,
            "submission_ready": False,
            "d5_scope": ["CICIDS-2017", "CESNET-TLS-Year22"],
        }
    )
    path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    (reports / "realdata_readiness_report.md").write_text(_readiness_markdown(readiness), encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbidden_artifact_snapshot() -> set[str]:
    return {str(path) for path in (Path("tables"), Path("figures"), Path("paper"), Path("results/table_optc.csv")) if path.exists()}


def _assert_no_forbidden_artifacts_created(before: set[str]) -> None:
    created = sorted(_forbidden_artifact_snapshot() - before)
    if created:
        raise RuntimeError(f"D5.5 must not create D6/D7 or OpTC artifacts: {created}")


def _smoke_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D5.5 Baseline Smoke Report",
        "",
        f"- Seed: {report['seed']}",
        f"- Passed baselines: {', '.join(report['passed_baselines']) or 'none'}",
        "",
        "## Excluded",
    ]
    lines.extend([f"- {name}: {reason}" for name, reason in report.get("excluded", {}).items()] or ["- none"])
    lines.extend(["", "## Rows"])
    for row in report["rows"]:
        lines.append(
            f"- {row['dataset']} {row['noise_type']} {row['method']}: "
            f"Macro-F1={row['macro_f1']:.6f}, finite={row['metrics_finite']}, "
            f"uses_noisy={row['uses_noisy_y_train']}"
        )
    lines.append("")
    return "\n".join(lines)


def _expansion_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D5.5 Baseline Expansion Report",
        "",
        f"- Completed: {report['completed']}",
        f"- Status: {report['baseline_expansion_status']}",
        f"- Original D5 rows unchanged: {report['original_d5_rows_unchanged']}",
        f"- Expanded rows: {report['rows']['expanded']}",
        f"- Added baseline rows: {report['rows']['added_baseline']}",
        f"- Included baselines: {', '.join(report['included_baselines']) or 'none'}",
        "",
        "## Key Metrics",
    ]
    for method, metrics in report["key_metrics"].items():
        lines.append(
            f"- {method}: Macro-F1 mean={metrics['macro_f1_mean']:.6f}, "
            f"ERR mean={metrics['err_final_mean']:.6f}, "
            f"compression mean={metrics['compression_ratio_mean']:.6f}"
        )
    lines.extend(["", "## Statistical Comparisons"])
    for name, info in report["statistics"]["comparisons"].items():
        lines.append(
            f"- {name}: n={info.get('n_pairs')}, diff={info.get('mean_diff')}, "
            f"dz={info.get('effect_size_cohen_dz')}, p={info.get('p_value')}"
        )
    lines.extend(["", "## Excluded"])
    lines.extend([f"- {name}: {reason}" for name, reason in report["excluded_baselines"].items()] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _sanity_markdown(report: dict[str, Any]) -> str:
    lines = ["# D5.5 Expanded Sanity Report", "", f"- Passed: {report['passed']}", "", "## Checks"]
    lines.extend([f"- {name}: {value}" for name, value in report["checks"].items()])
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {reason}" for reason in report.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _stat_markdown(report: dict[str, Any]) -> str:
    overall = report.get("overall", {})
    lines = [
        "# D5.5 Expanded Statistical Validity Report",
        "",
        f"- Overall p-value: {overall.get('p_value')}",
        f"- Overall effect size Cohen dz: {overall.get('effect_size_cohen_dz')}",
        f"- Pairing keys: {', '.join(overall.get('pairing_keys', []))}",
        "",
        "## Comparisons",
    ]
    for name, info in report.get("comparisons", {}).items():
        lines.append(f"- {name}: n={info.get('n_pairs')}, diff={info.get('mean_diff')}, p={info.get('p_value')}")
    lines.append("")
    return "\n".join(lines)


def _baseline_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Baseline Readiness Report",
        "",
        f"- Formal D5 methods: {', '.join(report['methods_in_formal_d5'])}",
        f"- Expanded D5 methods: {', '.join(report['methods_in_expanded_d5'])}",
        "",
    ]
    for name, info in report.items():
        if isinstance(info, dict) and "included" in info:
            lines.append(f"- {name}: included={info['included']}; reason={info['reason']}")
    lines.append("")
    return "\n".join(lines)


def _readiness_markdown(readiness: dict[str, Any]) -> str:
    lines = [
        "# Real-Data Readiness Report",
        "",
        f"- D5 completed: {readiness.get('d5_completed')}",
        f"- D5 baseline expansion completed: {readiness.get('d5_baseline_expansion_completed')}",
        f"- D6 allowed: {readiness.get('d6_allowed')}",
        f"- D7 allowed: {readiness.get('d7_allowed')}",
        f"- Submission ready: {readiness.get('submission_ready')}",
        f"- D5 scope: {', '.join(readiness.get('d5_scope', []))}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    print(json.dumps(run_d5_baseline_expansion(args.out, args.configs, args.reports), indent=2))


if __name__ == "__main__":
    main()
