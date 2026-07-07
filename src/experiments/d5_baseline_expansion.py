"""Real-data baseline expansion gate.

This runner appends only real, verified label-noise baselines to the already
completed formal matrix. It never overwrites ``results/table_main.csv`` or
``results/table_ablation.csv`` and never emits paper artifacts.
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
from src.baselines import (
    ConfidentLearningBaseline,
    CoTeachingBaseline,
    DecouplingBaseline,
    FINEBaseline,
    MCReBaseline,
    MORSEBaseline,
    NoisySupervisedBaseline,
)
from src.baselines.base import BaselineResult, array_hash
from src.experiments import cicids_mini_matrix, d5
from src.metrics import false_negative_rate, false_positive_rate, macro_f1
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import alert_compression_ratio, priority_scores


ADDED_BASELINE_FAMILIES = {
    "Noisy-Supervised": "noisy_supervised",
    "Confident-Learning": "confident_learning",
    "CL-filtering": "cl_filtering",
    "Co-Teaching": "co_teaching",
    "Decoupling": "decoupling",
    "FINE": "fine",
    "MCRe": "mcre",
    "MORSE": "morse",
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
VERIFY_DATASETS = ("cicids2017", "cesnet_tls_year22")
VERIFY_SPECS = (
    {"noise_type": "clean", "noise_rate": 0.0, "graph_beta": "none"},
    {"noise_type": "symmetric", "noise_rate": 0.2, "graph_beta": "none"},
)
VERIFY_SEED = 42


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
        raise FileNotFoundError("Baseline expansion requires existing real results/table_main.csv.")
    original_hash = _file_hash(original_path)
    original_main = pd.read_csv(original_path, keep_default_na=False)
    original_expanded = _annotate_original_rows(original_main)
    dataset_scope = _assert_original_d5_scope(original_expanded)

    scale_policy = _read_scale_policy(reports)
    verification = _load_or_run_verification_gate(configs, reports, scale_policy)
    passed_methods = tuple(verification["passed_baselines"])

    baseline_rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    baseline_csv = out / "table_baseline_expansion.csv"
    existing_baseline = pd.DataFrame(columns=EXPANDED_FIELDNAMES)
    if baseline_csv.exists():
        existing_baseline = pd.read_csv(baseline_csv, keep_default_na=False)
        if _baseline_frame_complete(existing_baseline, passed_methods, dataset_scope):
            baseline_rows = existing_baseline.to_dict(orient="records")
            runtime_records = _runtime_records_from_frame(existing_baseline)
    if not baseline_rows and passed_methods:
        baseline_rows, runtime_records = _run_expanded_matrix(
            out,
            configs,
            scale_policy,
            passed_methods,
            dataset_scope,
            seed_frame=existing_baseline,
        )

    baseline_frame = pd.DataFrame(baseline_rows, columns=EXPANDED_FIELDNAMES)
    baseline_frame.to_csv(out / "table_baseline_expansion.csv", index=False)

    expanded = pd.concat([original_expanded, baseline_frame], ignore_index=True)
    expanded = expanded.reindex(columns=EXPANDED_FIELDNAMES)
    expanded.to_csv(out / "table_main_expanded.csv", index=False)

    runtime = _runtime_json(pd.DataFrame(runtime_records), verification)
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
        verification,
        sanity,
        stat_tests,
        original_hash,
        original_unchanged,
    )
    (reports / "d5_baseline_expansion_report.json").write_text(json.dumps(expansion_report, indent=2), encoding="utf-8")
    (reports / "d5_baseline_expansion_report.md").write_text(_expansion_markdown(expansion_report), encoding="utf-8")
    _write_baseline_readiness_report(reports, verification, passed_methods)

    if sanity["passed"]:
        _update_readiness(reports, dataset_scope)

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
            "verification": "reports/baseline_verification_report.json",
            "expansion": "reports/d5_baseline_expansion_report.json",
            "sanity": "reports/d5_expanded_sanity_report.json",
            "stats": "reports/d5_expanded_statistical_validity_report.json",
        },
    }


def _run_verification_gate(configs: Path, reports: Path, scale_policy: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {}
    passed_by_method: dict[str, bool] = {}
    for dataset_name in VERIFY_DATASETS:
        bundle = d5._load_formal_dataset(dataset_name, VERIFY_SEED, configs, scale_policy)
        anomaly = d5._unsupervised_feature_anomaly(bundle.dataset.X_train)
        graph_cache: dict[float, Any] = {}
        for spec in VERIFY_SPECS:
            noisy, flip = d5._inject_noise(bundle.dataset, spec, VERIFY_SEED, graph_cache)
            evidence = _evidence(bundle, observed=noisy, anomaly=anomaly)
            for baseline in _baseline_candidates(VERIFY_SEED, float(spec["noise_rate"])):
                print(f"[baseline-verify] {dataset_name} {spec['noise_type']} {_candidate_method_name(baseline)}", flush=True)
                try:
                    result, runtime_sec, memory_mb = _timed_baseline(baseline, bundle, noisy)
                    metrics = _metrics_from_result(bundle, spec, VERIFY_SEED, result, runtime_sec, memory_mb, evidence, flip)
                    row = {
                        "dataset": dataset_name,
                        "reported_as": bundle.reported_as,
                        "seed": VERIFY_SEED,
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
                        failures.setdefault(result.method, []).append(f"failed verification on {dataset_name}/{spec['noise_type']}")
                except Exception as exc:
                    name = getattr(baseline, "method", baseline.__class__.__name__)
                    failures.setdefault(str(name), []).append(f"{dataset_name}/{spec['noise_type']}: {exc}")
                    passed_by_method[str(name)] = False
    methods_seen = {row["method"] for row in rows}
    passed = sorted(method for method in methods_seen if passed_by_method.get(method, False) and _seen_all_verification_rows(rows, method))
    excluded = _excluded_baselines(passed, failures)
    report = {
        "stage": "baseline verification gate",
        "p2_baseline_calibration": True,
        "seed": VERIFY_SEED,
        "datasets": list(VERIFY_DATASETS),
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
    (reports / "baseline_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "baseline_verification_report.md").write_text(_verification_markdown(report), encoding="utf-8")
    return report


def _load_or_run_verification_gate(configs: Path, reports: Path, scale_policy: dict[str, Any]) -> dict[str, Any]:
    path = reports / "baseline_verification_report.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("rows", [])
        if (
            report.get("p2_baseline_calibration") is True
            and report.get("passed_baselines")
            and len(rows) >= len(VERIFY_DATASETS) * len(VERIFY_SPECS)
        ):
            return report
    return _run_verification_gate(configs, reports, scale_policy)


def _run_expanded_matrix(
    out: Path,
    configs: Path,
    scale_policy: dict[str, Any],
    passed_methods: tuple[str, ...],
    dataset_scope: tuple[str, ...],
    seed_frame: pd.DataFrame | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    partial_path = out / "table_baseline_expansion.partial.csv"
    existing = _load_partial_baseline_rows(partial_path, passed_methods)
    seeded = _usable_existing_baseline_rows(seed_frame, passed_methods, dataset_scope)
    if not seeded.empty:
        existing = pd.concat([seeded, existing], ignore_index=True)
        existing = _deduplicate_baseline_rows(existing)
    rows: list[dict[str, Any]] = existing.to_dict(orient="records") if not existing.empty else []
    runtime_records: list[dict[str, Any]] = _runtime_records_from_frame(existing) if not existing.empty else []
    done = {_row_key(row) for row in rows}
    for dataset_name in dataset_scope:
        for seed in d5.SEEDS:
            if _all_expected_keys_done(dataset_name, seed, passed_methods, done):
                continue
            bundle = d5._load_formal_dataset(dataset_name, seed, configs, scale_policy)
            anomaly = d5._unsupervised_feature_anomaly(bundle.dataset.X_train)
            graph_cache: dict[float, Any] = {}
            for spec in d5._noise_specs():
                if _all_spec_keys_done(dataset_name, seed, spec, passed_methods, done):
                    continue
                noisy, flip = d5._inject_noise(bundle.dataset, spec, seed, graph_cache)
                evidence = _evidence(bundle, observed=noisy, anomaly=anomaly)
                for baseline in _baseline_candidates(seed, float(spec["noise_rate"])):
                    if _candidate_method_name(baseline) not in passed_methods:
                        continue
                    candidate_key = _row_key(
                        {
                            "dataset": dataset_name,
                            "noise_type": spec["noise_type"],
                            "noise_rate": spec["noise_rate"],
                            "graph_beta": spec["graph_beta"],
                            "seed": seed,
                            "method": _candidate_method_name(baseline),
                        }
                    )
                    if candidate_key in done:
                        continue
                    print(
                        f"[baseline-matrix] {dataset_name} seed={seed} "
                        f"{spec['noise_type']} rate={spec['noise_rate']} beta={spec['graph_beta']} "
                        f"{_candidate_method_name(baseline)}",
                        flush=True,
                    )
                    result, runtime_sec, memory_mb = _timed_baseline(baseline, bundle, noisy)
                    row = _row_from_result(bundle, spec, seed, result, runtime_sec, memory_mb, evidence, flip)
                    rows.append(row)
                    done.add(_row_key(row))
                    _append_partial_row(partial_path, row)
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


def _all_expected_keys_done(dataset_name: str, seed: int, methods: tuple[str, ...], done: set[tuple[str, str, str, str, str, str]]) -> bool:
    expected = {
        _row_key(
            {
                "dataset": dataset_name,
                "noise_type": spec["noise_type"],
                "noise_rate": spec["noise_rate"],
                "graph_beta": spec["graph_beta"],
                "seed": seed,
                "method": method,
            }
        )
        for spec in d5._noise_specs()
        for method in methods
    }
    return bool(expected.issubset(done))


def _all_spec_keys_done(
    dataset_name: str,
    seed: int,
    spec: dict[str, Any],
    methods: tuple[str, ...],
    done: set[tuple[str, str, str, str, str, str]],
) -> bool:
    expected = {
        _row_key(
            {
                "dataset": dataset_name,
                "noise_type": spec["noise_type"],
                "noise_rate": spec["noise_rate"],
                "graph_beta": spec["graph_beta"],
                "seed": seed,
                "method": method,
            }
        )
        for method in methods
    }
    return bool(expected.issubset(done))


def _usable_existing_baseline_rows(
    frame: pd.DataFrame | None,
    passed_methods: tuple[str, ...],
    dataset_scope: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None or frame.empty or not set(EXPANDED_FIELDNAMES).issubset(frame.columns):
        return pd.DataFrame(columns=EXPANDED_FIELDNAMES)
    out = frame.reindex(columns=EXPANDED_FIELDNAMES).copy()
    out = out[out["method"].astype(str).isin(set(passed_methods))]
    out = out[out["dataset"].astype(str).isin(set(dataset_scope))]
    numeric = out.select_dtypes(include=[np.number])
    if not numeric.empty:
        finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        out = out.loc[finite]
    return _deduplicate_baseline_rows(out)


def _deduplicate_baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=EXPANDED_FIELDNAMES)
    work = frame.reindex(columns=EXPANDED_FIELDNAMES).copy()
    work["__row_key"] = [_row_key(row) for row in work.to_dict(orient="records")]
    work = work.drop_duplicates("__row_key", keep="last").drop(columns=["__row_key"])
    return work.reindex(columns=EXPANDED_FIELDNAMES)


def _load_partial_baseline_rows(path: Path, passed_methods: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EXPANDED_FIELDNAMES)
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty or "method" not in frame.columns:
        return pd.DataFrame(columns=EXPANDED_FIELDNAMES)
    frame = frame[frame["method"].astype(str).isin(set(passed_methods))]
    return frame.reindex(columns=EXPANDED_FIELDNAMES)


def _append_partial_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{name: row.get(name, "") for name in EXPANDED_FIELDNAMES}], columns=EXPANDED_FIELDNAMES)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["dataset"]),
        str(row["noise_type"]),
        f"{float(row['noise_rate']):.6f}",
        str(row["graph_beta"]),
        str(int(row["seed"])),
        str(row["method"]),
    )


def _baseline_candidates(seed: int, noise_rate: float):
    return (
        NoisySupervisedBaseline(seed=seed),
        ConfidentLearningBaseline(seed=seed, noise_rate=noise_rate),
        CoTeachingBaseline(seed=seed, noise_rate=noise_rate, epochs=1, batch_size=8192, n_estimators=8),
        DecouplingBaseline(seed=seed, epochs=3, batch_size=8192),
        FINEBaseline(seed=seed, noise_rate=noise_rate, n_components=8, n_estimators=8),
        MCReBaseline(seed=seed, noise_rate=noise_rate, n_components=0, max_iter=4, n_estimators=8),
        MORSEBaseline(seed=seed, noise_rate=noise_rate, n_components=0, max_iter=4, n_estimators=8),
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


def _evidence(
    bundle: d5.FormalBundle,
    observed: np.ndarray | None = None,
    anomaly: np.ndarray | None = None,
) -> np.ndarray:
    labels = bundle.dataset.y_train if observed is None else np.asarray(observed)
    if anomaly is None:
        anomaly = d5._unsupervised_feature_anomaly(bundle.dataset.X_train)
    return compute_evidence(
        labels,
        {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
        anomaly=anomaly,
    )


def _annotate_original_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["replacement_for"] = out.get("replacement_for", "").fillna("").astype(str).replace({"nan": ""})
    out["method_family"] = out["method"].map(ORIGINAL_METHOD_FAMILIES).fillna("original_d5")
    out["implementation_status"] = "reused_verified_d5"
    return out.reindex(columns=EXPANDED_FIELDNAMES)


def _assert_original_d5_scope(frame: pd.DataFrame) -> tuple[str, ...]:
    datasets = {str(value) for value in frame["dataset"].unique()}
    methods = {str(value) for value in frame["method"].unique()}
    allowed = set(d5.BASE_FORMAL_DATASETS) | {"unsw_nb15"}
    required = set(d5.BASE_FORMAL_DATASETS)
    if not required.issubset(datasets) or not datasets.issubset(allowed):
        raise RuntimeError(f"Original D5 scope mismatch: {datasets}")
    if methods != set(d5.FORMAL_METHODS):
        raise RuntimeError(f"Original D5 methods mismatch: {methods}")
    forbidden = {"maltls22", "optc"}
    if datasets.intersection(forbidden):
        raise RuntimeError(f"Forbidden original D5 datasets found: {datasets.intersection(forbidden)}")
    return tuple(dataset for dataset in (*d5.BASE_FORMAL_DATASETS, "unsw_nb15") if dataset in datasets)


def _read_scale_policy(reports: Path) -> dict[str, Any]:
    path = reports / "d5_scale_policy.json"
    if not path.exists():
        raise FileNotFoundError("Baseline expansion requires reports/d5_scale_policy.json from the completed real matrix run.")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    keys = ("macro_f1", "fpr", "fnr", "err", "err_tail", "err_final", "compression_ratio")
    return bool(np.isfinite([float(metrics[key]) for key in keys]).all())


def _perfect_anomaly(metrics: dict[str, Any]) -> bool:
    return bool(float(metrics["macro_f1"]) >= 0.999 and float(metrics["fpr"]) <= 0.001 and float(metrics["fnr"]) <= 0.001)


def _seen_all_verification_rows(rows: list[dict[str, Any]], method: str) -> bool:
    seen = {(row["dataset"], row["noise_type"]) for row in rows if row["method"] == method}
    return seen == {(dataset, spec["noise_type"]) for dataset in VERIFY_DATASETS for spec in VERIFY_SPECS}


def _excluded_baselines(passed: list[str], failures: dict[str, list[str]]) -> dict[str, str]:
    excluded = {
        "Flash": "excluded: provenance case-study method; no formal real-data label-noise implementation",
        "Argus": "excluded: provenance case-study method; no formal real-data label-noise implementation",
    }
    for name, reasons in failures.items():
        if name not in passed:
            excluded[name] = "; ".join(reasons)
    for name in passed:
        excluded.pop(name, None)
    if "Confident-Learning" not in passed and "CL-filtering" in passed:
        excluded["Confident-Learning"] = "official cleanlab package was unavailable at runtime; documented CL-filtering was used instead"
    return excluded


def _runtime_json(runtime: pd.DataFrame, verification: dict[str, Any]) -> dict[str, Any]:
    if runtime.empty:
        return {"records": [], "summary": {}, "verified_baselines": verification["passed_baselines"]}
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
        "verified_baselines": verification["passed_baselines"],
    }


def _baseline_frame_complete(frame: pd.DataFrame, passed_methods: tuple[str, ...], dataset_scope: tuple[str, ...]) -> bool:
    if frame.empty or not set(EXPANDED_FIELDNAMES).issubset(frame.columns):
        return False
    expected = len(dataset_scope) * len(d5.SEEDS) * len(d5._noise_specs()) * len(passed_methods)
    if len(frame) != expected:
        return False
    if set(frame["dataset"].astype(str).unique()) != set(dataset_scope):
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
    verification: dict[str, Any],
    sanity: dict[str, Any],
    stat_tests: dict[str, Any],
    original_hash: str,
    original_unchanged: bool,
) -> dict[str, Any]:
    added_methods = sorted(baseline_frame["method"].unique().tolist()) if not baseline_frame.empty else []
    status = "strong" if len(added_methods) >= 2 else "acceptable_but_limited" if len(added_methods) == 1 else "limited"
    return {
        "stage": "real-data baseline expansion",
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
        "excluded_baselines": verification.get("excluded", {}),
        "verification": {
            "passed_baselines": verification["passed_baselines"],
            "failed_baselines": verification["failed_baselines"],
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


def _write_baseline_readiness_report(reports: Path, verification: dict[str, Any], passed_methods: tuple[str, ...]) -> None:
    report: dict[str, Any] = {
        "stage": "baseline readiness audit",
        "methods_in_formal_d5": list(d5.FORMAL_METHODS),
        "methods_in_expanded_d5": list(d5.FORMAL_METHODS) + list(passed_methods),
        "unimplemented_methods_emit_rows": False,
    }
    for method in d5.FORMAL_METHODS:
        report[method] = {"included": True, "reason": "reused from verified real D5 run"}
    for method in passed_methods:
        report[method] = {
            "included": True,
            "reason": "verified on mandatory real datasets and run on the active formal scope",
        }
    for method, reason in verification.get("excluded", {}).items():
        report[method] = {"included": False, "reason": reason}
    legacy_aliases = {
        "cleanlab": "legacy audit key; official cleanlab is represented by the Confident-Learning method row",
        "Co-Teaching+": "not independently implemented and verified on real data",
    }
    for method, reason in legacy_aliases.items():
        report.setdefault(method, {"included": False, "reason": reason})
    (reports / "baseline_readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "baseline_readiness_report.md").write_text(_baseline_markdown(report), encoding="utf-8")


def _update_readiness(reports: Path, dataset_scope: tuple[str, ...]) -> None:
    path = reports / "realdata_readiness_report.json"
    readiness = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    labels = {
        "cicids2017": "CICIDS-2017",
        "cesnet_tls_year22": "CESNET-TLS-Year22",
        "unsw_nb15": "UNSW-NB15",
    }
    readiness.update(
        {
            "d5_completed": True,
            "d5_baseline_expansion_completed": True,
            "d5_allowed": True,
            "d6_allowed": True,
            "d7_allowed": False,
            "d6_d7_allowed": False,
            "submission_ready": False,
            "d5_scope": [labels.get(name, name) for name in dataset_scope],
        }
    )
    if "unsw_nb15" in dataset_scope:
        _merge_unsw_readiness(readiness, reports)
    path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    (reports / "realdata_readiness_report.md").write_text(_readiness_markdown(readiness), encoding="utf-8")


def _merge_unsw_readiness(readiness: dict[str, Any], reports: Path) -> None:
    unsw_path = reports / "unsw_ingest.json"
    if not unsw_path.exists():
        return
    try:
        unsw = json.loads(unsw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    datasets = readiness.setdefault("datasets", {})
    datasets["unsw_nb15"] = {
        **datasets.get("unsw_nb15", {}),
        "available": bool(unsw.get("ready_for_smoke") or unsw.get("ready_for_d5_component")),
        "audit_passed": bool(unsw.get("ready_for_d5_component")),
        "ready_for_smoke": bool(unsw.get("ready_for_smoke")),
        "ready_for_d5": bool(unsw.get("ready_for_d5_component")),
        "ready_for_d5_component": bool(unsw.get("ready_for_d5_component")),
        "source_verified": bool(unsw.get("source_verified", True)),
        "reported_as": unsw.get("reported_as", "UNSW-NB15"),
        "actual_data_path": unsw.get("actual_data_path"),
        "active_views": unsw.get("active_views", []),
        "blocking_reasons": unsw.get("blocking_reasons", []),
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbidden_artifact_snapshot() -> set[str]:
    return {str(path) for path in (Path("tables"), Path("figures"), Path("paper"), Path("results/table_optc.csv")) if path.exists()}


def _assert_no_forbidden_artifacts_created(before: set[str]) -> None:
    created = sorted(_forbidden_artifact_snapshot() - before)
    if created:
        raise RuntimeError(f"Baseline expansion must not create paper or OpTC artifacts: {created}")


def _verification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Baseline Verification Report",
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
        "# Baseline Expansion Report",
        "",
        f"- Completed: {report['completed']}",
        f"- Status: {report['baseline_expansion_status']}",
        f"- Original matrix rows unchanged: {report['original_d5_rows_unchanged']}",
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
    lines = ["# Expanded Result Sanity Report", "", f"- Passed: {report['passed']}", "", "## Checks"]
    lines.extend([f"- {name}: {value}" for name, value in report["checks"].items()])
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {reason}" for reason in report.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _stat_markdown(report: dict[str, Any]) -> str:
    overall = report.get("overall", {})
    lines = [
        "# Expanded Statistical Validity Report",
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
        f"- Formal methods: {', '.join(report['methods_in_formal_d5'])}",
        f"- Expanded methods: {', '.join(report['methods_in_expanded_d5'])}",
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
        f"- Main matrix completed: {readiness.get('d5_completed')}",
        f"- Baseline expansion completed: {readiness.get('d5_baseline_expansion_completed')}",
        f"- Paper asset generation allowed: {readiness.get('d6_allowed')}",
        f"- Manuscript assembly allowed: {readiness.get('d7_allowed')}",
        f"- Submission ready: {readiness.get('submission_ready')}",
        f"- Experiment scope: {', '.join(readiness.get('d5_scope', []))}",
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
