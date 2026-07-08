"""P2f tightened evidence-retention audit.

P2f supersedes the P2e rare-recovery headline. It keeps the P2e rescue weight
intact, removes the retention tautology from rare recovery, and re-runs a
better-powered tail comparison on real audit windows only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.experiments import d5
from src.metrics import (
    degree_evidence_retention_components,
    false_negative_rate,
    false_positive_rate,
    macro_f1,
    rare_evidence_recovery_rate,
)
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e


DATASETS = ("cicids2017", "cesnet_tls_year22", "unsw_nb15")
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD-soft", "Graph-CoLD-semisup")
CANDIDATES = ("Graph-CoLD-soft", "Graph-CoLD-semisup")
RATES = (0.4, 0.6, 0.8)
SEEDS = tuple(range(10))
DEFAULT_TRAIN_SIZE = 10_000
DEFAULT_TEST_SIZE = 5_000
RETENTION_THRESHOLD = p2e.RETENTION_THRESHOLD
SOFT_CFG = p2e.SOFT_CFG


def run_p2f_tighten(
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    figures_dir: str | Path = "figures",
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
) -> dict[str, Any]:
    configs = Path(configs_dir)
    out = Path(out_dir)
    reports = Path(reports_dir)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    for directory in (out, reports, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    prereg = reports / "p2f_preregistration.md"
    if not prereg.exists():
        raise FileNotFoundError("P2f preregistration must exist before running final P2f numbers.")

    gate = _p2e_gate_note(configs, reports, figures, train_size, min(test_size, 1000))
    frame, tail_frame = _run_powered_matrix(configs, reports, train_size, test_size)
    results_csv = out / "p2f_tail_powered.csv"
    tail_csv = tables / "table_p2f_tail_breakdown.csv"
    frame.to_csv(results_csv, index=False)
    tail_frame.to_csv(tail_csv, index=False)

    tests = _powered_tests(frame)
    tests_csv = tables / "table_p2f_powered_tests.csv"
    pd.DataFrame(tests["tests"]).to_csv(tests_csv, index=False)
    summary = _summary(frame)
    summary_csv = tables / "table_p2f_summary.csv"
    summary.to_csv(summary_csv, index=False)
    _plot_powered_tail(frame, figures / "fig_p2f_tail_powered.pdf")
    _plot_recovery(frame, figures / "fig_p2f_corrected_recovery.pdf")

    report = {
        "stage": "P2f",
        "completed": True,
        "real_data_only": True,
        "pre_registration": str(prereg).replace("\\", "/"),
        "scope": {
            "datasets": [p2d.DATASET_LABELS[name] for name in DATASETS],
            "methods": list(METHODS),
            "candidate_methods": list(CANDIDATES),
            "rates": [float(rate) for rate in RATES],
            "seeds": list(SEEDS),
            "train_size": int(train_size),
            "test_size": int(test_size),
            "paired_rows_per_dataset_metric": len(RATES) * len(SEEDS),
        },
        "p2e_gate_note": gate,
        "corrected_metric": {
            "rare_recovery_rate": (
                "eligible(clean rare/tail and GraphCDM>theta) samples predicted as their clean true class; "
                "retention is diagnostic only and not in the numerator"
            ),
            "evaluation": "final_trained_classifier_predictions_on_training_nodes",
            "oof_audit": "out_of_fold predictions are retained as a memorization-sensitivity audit column",
            "non_constant_examples": _nonconstant_examples(frame),
            "p2e_before": _p2e_before_numbers(),
        },
        "outputs": {
            "powered_results": str(results_csv).replace("\\", "/"),
            "tail_breakdown": str(tail_csv).replace("\\", "/"),
            "powered_tests": str(tests_csv).replace("\\", "/"),
            "summary": str(summary_csv).replace("\\", "/"),
            "tail_figure": "figures/fig_p2f_tail_powered.pdf",
            "recovery_figure": "figures/fig_p2f_corrected_recovery.pdf",
        },
        "row_count": int(len(frame)),
        "tail_breakdown_rows": int(len(tail_frame)),
        "summary": summary.to_dict(orient="records"),
        "tests": tests,
        "claims_input": _claims_input(tests, frame),
        "reproduction_commands": [
            "python -m pytest tests/test_soft_not_hard.py tests/test_rare_recovery_nontautological.py -q",
            (
                "python -m src.paper.p2f_tighten --configs configs --out results --reports reports "
                "--tables tables --figures figures --train-size 10000 --test-size 5000"
            ),
        ],
    }
    (reports / "p2f_tighten.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8")
    (reports / "p2f_tighten.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _p2e_gate_note(configs: Path, reports: Path, figures: Path, train_size: int, test_size: int) -> dict[str, Any]:
    test_result = _soft_not_hard_probe()
    tension = p2e._run_tension_gate(configs, reports, figures, train_size, test_size, (0.4, 0.6), 0.5)
    return {
        "soft_not_hard_probe_passed": test_result["passed"],
        "soft_not_hard_probe": test_result,
        "tension_regenerated": bool(tension.get("completed") and tension.get("gate_passed")),
        "max_tension_rate": float(tension.get("max_tension_rate", 0.0)),
        "expected_max_tension_approx": 0.375,
        "passed": bool(test_result["passed"] and tension.get("gate_passed") and float(tension.get("max_tension_rate", 0.0)) >= 0.35),
    }


def _soft_not_hard_probe() -> dict[str, Any]:
    cdm = np.array([0.9, 0.9, 0.2], dtype=float)
    evidence = np.array([1.0, 0.0, 0.5], dtype=float)
    hard = graph_cdm.soft_weights(cdm, evidence, {"evidence_preserving": {"theta": 0.5, "rho": 0.0}})
    soft = graph_cdm.soft_weights(cdm, evidence, {"evidence_preserving": {"theta": 0.5, "kappa": 8.0, "rho": 0.2, "lambda_rescue": 0.8}})
    return {
        "passed": bool(hard[0] < RETENTION_THRESHOLD <= soft[0] and soft[0] > soft[1] and hard[2] == 1.0),
        "hard_weights": hard.tolist(),
        "soft_weights": soft.tolist(),
    }


def _run_powered_matrix(configs: Path, reports: Path, train_size: int, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale = d5.write_scale_policy_report(reports)
    rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for dataset_name in DATASETS:
        for seed in SEEDS:
            sample = p2d._sample_bundle(dataset_name, seed, configs, reports, scale, train_size, test_size)
            tail_labels = p2e._tail_malicious_labels(sample.y_train, sample.benign_class)
            for rate in RATES:
                print(f"[p2f] {dataset_name} seed={seed} tail_asymmetric={rate:.0%}", flush=True)
                noisy, flip = p2e._inject_tail_asymmetric(sample.y_train, tail_labels, sample.benign_class, rate, seed=seed)
                evidence = compute_evidence(
                    noisy,
                    {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
                    anomaly=sample.anomaly,
                )
                cdm = d5._cdm_from_observed_labels(noisy, evidence, sample.graph, sample.num_classes)
                suspicious = cdm > float(SOFT_CFG["evidence_preserving"]["theta"])
                hard_weights = d5._weights_for_hard(cdm, evidence)
                soft_rescue = graph_cdm.soft_weights(cdm, evidence, SOFT_CFG)
                for method in METHODS:
                    start = time.perf_counter()
                    tracemalloc.start()
                    weights, train_labels, training_info = p2e._method_training_plan(
                        method,
                        sample,
                        noisy,
                        cdm,
                        evidence,
                        hard_weights,
                        soft_rescue,
                        seed,
                    )
                    y_train_final, y_test_pred = p2e._fit_predict_train_test(
                        sample.X_train,
                        train_labels,
                        sample.X_test,
                        weights,
                        method,
                        seed,
                    )
                    y_train_eval = p2e._crossfit_train_predictions(sample.X_train, train_labels, weights, method, seed)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    row = _row(
                        sample,
                        method,
                        rate,
                        seed,
                        noisy,
                        flip,
                        cdm,
                        evidence,
                        suspicious,
                        tail_labels,
                        weights,
                        y_train_eval,
                        y_train_final,
                        y_test_pred,
                        training_info,
                        time.perf_counter() - start,
                        peak / (1024 * 1024),
                    )
                    rows.append(row)
                    tail_rows.extend(
                        _tail_rows(sample, method, rate, seed, tail_labels, flip, cdm, evidence, weights, y_train_eval, y_test_pred)
                    )
    return pd.DataFrame(rows), pd.DataFrame(tail_rows)


def _row(
    sample: Any,
    method: str,
    rate: float,
    seed: int,
    noisy: np.ndarray,
    flip: np.ndarray,
    cdm: np.ndarray,
    evidence: np.ndarray,
    suspicious: np.ndarray,
    tail_labels: np.ndarray,
    weights: np.ndarray,
    y_train_eval: np.ndarray,
    y_train_final: np.ndarray,
    y_test_pred: np.ndarray,
    training_info: dict[str, Any],
    runtime_sec: float,
    memory_mb: float,
) -> dict[str, Any]:
    del noisy
    clean = ~np.asarray(flip, dtype=bool)
    binary_err = p2e._binary_err(weights, evidence, clean, sample.y_train, flip)
    degree_err = degree_evidence_retention_components(weights, evidence, clean, sample.y_train)
    recovery = rare_evidence_recovery_rate(
        weights,
        sample.y_train,
        y_train_final,
        clean,
        suspicious,
        tail_labels,
        retention_threshold=RETENTION_THRESHOLD,
    )
    oof_recovery = rare_evidence_recovery_rate(
        weights,
        sample.y_train,
        y_train_eval,
        clean,
        suspicious,
        tail_labels,
        retention_threshold=RETENTION_THRESHOLD,
    )
    retained = np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD
    return {
        "dataset": sample.dataset_key,
        "reported_as": sample.reported_as,
        "dataset_hash": sample.dataset_hash,
        "actual_data_path": sample.actual_data_path,
        "class_policy": sample.class_policy,
        "sample_policy": sample.sample_policy,
        "train_rows": int(sample.X_train.shape[0]),
        "test_rows": int(sample.X_test.shape[0]),
        "num_classes": int(sample.num_classes),
        "noise_type": "tail_asymmetric",
        "tail_flip_rate": float(rate),
        "effective_flip_rate_train": float(np.mean(flip)),
        "seed": int(seed),
        "method": method,
        "training_mode": training_info.get("training_mode", ""),
        "macro_f1": macro_f1(sample.y_test, y_test_pred),
        "fpr": false_positive_rate(sample.y_test, y_test_pred, sample.benign_class),
        "fnr": false_negative_rate(sample.y_test, y_test_pred, sample.benign_class),
        "tail_macro_f1": p2e._tail_macro_f1(sample.y_test, y_test_pred, tail_labels),
        "tail_recall": p2e._tail_recall(sample.y_test, y_test_pred, tail_labels),
        "rare_recovery_rate": recovery["rare_recovery_rate"],
        "rare_recovery_rate_oof_audit": oof_recovery["rare_recovery_rate"],
        "rare_retained_rate": recovery["rare_retained_rate"],
        "rare_clean_suspicious_count": recovery["rare_clean_suspicious_count"],
        "rare_recovered_count": recovery["rare_recovered_count"],
        "rare_recovery_eval": "final_trained_classifier_predictions_on_training_nodes",
        "err": binary_err["err"],
        "err_tail": binary_err["err_tail"],
        "err_final": binary_err["err_final"],
        "err_final_degree": degree_err["err_final_degree"],
        "mean_weight": float(np.mean(weights)),
        "retained_fraction": float(np.mean(retained)),
        "semisup_candidate_count": int(training_info.get("semisup_candidate_count", 0)),
        "semisup_selected_count": int(training_info.get("semisup_selected_count", 0)),
        "runtime_sec": float(runtime_sec),
        "memory_mb": float(memory_mb),
        "tail_labels": json.dumps([int(label) for label in tail_labels]),
        "tail_label_names": json.dumps(p2e._label_names(sample, tail_labels)),
        "active_views": sample.active_views,
        "source_verified": sample.source_verified,
        "replacement_for": sample.replacement_for,
    }


def _tail_rows(
    sample: Any,
    method: str,
    rate: float,
    seed: int,
    tail_labels: np.ndarray,
    flip: np.ndarray,
    cdm: np.ndarray,
    evidence: np.ndarray,
    weights: np.ndarray,
    y_train_eval: np.ndarray,
    y_test_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean = ~np.asarray(flip, dtype=bool)
    suspicious = cdm > float(SOFT_CFG["evidence_preserving"]["theta"])
    retained = np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD
    names = p2e._label_names(sample, tail_labels)
    for label, name in zip(tail_labels, names):
        train_mask = sample.y_train == int(label)
        test_mask = sample.y_test == int(label)
        eligible = train_mask & clean & suspicious
        recovered = eligible & (y_train_eval == sample.y_train)
        rows.append(
            {
                "dataset": sample.dataset_key,
                "reported_as": sample.reported_as,
                "tail_flip_rate": float(rate),
                "seed": int(seed),
                "method": method,
                "tail_label": int(label),
                "tail_label_name": name,
                "train_tail_count": int(train_mask.sum()),
                "test_tail_count": int(test_mask.sum()),
                "train_tail_flipped_count": int(np.sum(train_mask & flip)),
                "clean_suspicious_tail_count": int(eligible.sum()),
                "clean_suspicious_tail_recovery_rate": float(np.mean(recovered[eligible])) if eligible.any() else 0.0,
                "clean_suspicious_tail_retained_rate": float(np.mean(retained[eligible])) if eligible.any() else 0.0,
                "test_tail_recall": float(np.mean(y_test_pred[test_mask] == sample.y_test[test_mask])) if test_mask.any() else np.nan,
                "mean_tail_evidence": p2e._mean(evidence[train_mask]),
                "mean_tail_cdm": p2e._mean(cdm[train_mask]),
                "mean_tail_weight": p2e._mean(weights[train_mask]),
            }
        )
    return rows


def _powered_tests(frame: pd.DataFrame) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for dataset, part in frame.groupby("reported_as", dropna=False):
        dataset_tests: list[dict[str, Any]] = []
        for method in CANDIDATES:
            regression = _macro_regression(part, method)
            for metric in ("tail_macro_f1", "rare_recovery_rate"):
                stat = _paired_stat(part, method, "ablation_hard", metric)
                stat.update(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "aggregate_macro_f1_delta_vs_hard": regression["mean_delta"],
                        "aggregate_macro_f1_regression_p_less": regression["p_less"],
                        "no_significant_aggregate_macro_f1_regression": regression["passed"],
                    }
                )
                dataset_tests.append(stat)
        adjusted = _holm([item["p_value_greater"] for item in dataset_tests])
        for item, p_holm in zip(dataset_tests, adjusted):
            item["p_holm"] = p_holm
            item["meets_success"] = bool(
                item["mean_delta"] > 0.0
                and item["p_holm"] < 0.05
                and item["cohens_dz"] >= 0.3
                and item["no_significant_aggregate_macro_f1_regression"]
            )
            tests.append(item)
    dataset_pass = {
        dataset: bool(any(item["dataset"] == dataset and item["meets_success"] for item in tests))
        for dataset in sorted(frame["reported_as"].unique())
    }
    pass_count = sum(dataset_pass.values())
    verdict = "robust" if pass_count >= 2 else "narrow" if pass_count == 1 else "null"
    return {
        "verdict": verdict,
        "dataset_pass": dataset_pass,
        "pass_count": int(pass_count),
        "tests": tests,
    }


def _paired_stat(frame: pd.DataFrame, method: str, baseline: str, metric: str) -> dict[str, Any]:
    part = frame[frame["method"].isin([method, baseline])].copy()
    if metric == "rare_recovery_rate":
        part = part[part["rare_clean_suspicious_count"] > 0]
    pivot = part.pivot_table(index=["tail_flip_rate", "seed"], columns="method", values=metric)
    if method not in pivot.columns or baseline not in pivot.columns:
        return _empty_stat()
    pivot = pivot[[method, baseline]].dropna()
    if pivot.empty:
        return _empty_stat()
    a = pivot[method].to_numpy(dtype=float)
    b = pivot[baseline].to_numpy(dtype=float)
    diff = a - b
    ci_low, ci_high = _bootstrap_ci(diff)
    return {
        "paired_rows": int(len(pivot)),
        "method_mean": float(np.mean(a)),
        "baseline_mean": float(np.mean(b)),
        "mean_delta": float(np.mean(diff)),
        "p_value_greater": _paired_pvalue(a, b, alternative="greater"),
        "cohens_dz": _cohens_dz(diff),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
    }


def _macro_regression(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    pivot = frame[frame["method"].isin([method, "ablation_hard"])].pivot_table(
        index=["tail_flip_rate", "seed"],
        columns="method",
        values="macro_f1",
    )
    if method not in pivot.columns or "ablation_hard" not in pivot.columns:
        return {"passed": False, "p_less": 1.0, "mean_delta": 0.0}
    pivot = pivot[[method, "ablation_hard"]].dropna()
    a = pivot[method].to_numpy(dtype=float)
    b = pivot["ablation_hard"].to_numpy(dtype=float)
    diff = a - b
    p_less = _paired_pvalue(a, b, alternative="less")
    return {"passed": bool(not (np.mean(diff) < 0.0 and p_less < 0.05)), "p_less": p_less, "mean_delta": float(np.mean(diff))}


def _holm(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = p.shape[0]
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return [float(value) for value in adjusted]


def _paired_pvalue(a: np.ndarray, b: np.ndarray, alternative: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[0] < 2:
        return 1.0
    diff = a - b
    std = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    if std <= 1e-12:
        if alternative == "greater":
            return 1e-12 if mean > 0.0 else 1.0
        if alternative == "less":
            return 1e-12 if mean < 0.0 else 1.0
        return 1e-12 if not np.isclose(mean, 0.0) else 1.0
    return float(max(stats.ttest_rel(a, b, alternative=alternative).pvalue, 1e-12))


def _cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if diff.shape[0] < 2:
        return 0.0
    std = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    if std <= 1e-12:
        return 99.0 if mean > 0.0 else -99.0 if mean < 0.0 else 0.0
    return float(mean / std)


def _bootstrap_ci(diff: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    if diff.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    samples = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _empty_stat() -> dict[str, Any]:
    return {
        "paired_rows": 0,
        "method_mean": 0.0,
        "baseline_mean": 0.0,
        "mean_delta": 0.0,
        "p_value_greater": 1.0,
        "cohens_dz": 0.0,
        "bootstrap_ci95_low": 0.0,
        "bootstrap_ci95_high": 0.0,
    }


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["reported_as", "method"], dropna=False)[
            ["macro_f1", "tail_macro_f1", "tail_recall", "rare_recovery_rate", "rare_recovery_rate_oof_audit", "rare_retained_rate"]
        ]
        .mean()
        .reset_index()
    )


def _nonconstant_examples(frame: pd.DataFrame) -> dict[str, Any]:
    hard = frame[frame["method"] == "ablation_hard"]["rare_recovery_rate"].to_numpy(dtype=float)
    soft = frame[frame["method"] == "Graph-CoLD-soft"]["rare_recovery_rate"].to_numpy(dtype=float)
    return {
        "hard_max_gt_zero": bool(hard.size and np.max(hard) > 0.0),
        "hard_max": float(np.max(hard)) if hard.size else 0.0,
        "soft_min_lt_one": bool(soft.size and np.min(soft) < 1.0),
        "soft_min": float(np.min(soft)) if soft.size else 0.0,
        "hard_mean": float(np.mean(hard)) if hard.size else 0.0,
        "soft_mean": float(np.mean(soft)) if soft.size else 0.0,
    }


def _p2e_before_numbers() -> list[dict[str, Any]]:
    path = Path("tables/table_p2e_success_tests.csv")
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    part = frame[frame["metric"] == "rare_recovery_rate"].copy()
    return part[["dataset", "method", "method_mean", "baseline_mean", "mean_delta", "p_value_greater", "cohens_dz"]].to_dict(orient="records")


def _claims_input(tests: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    verdict = tests["verdict"]
    dataset_pass = tests["dataset_pass"]
    trend = _trend_text(frame)
    if verdict == "robust":
        claim = (
            "Evidence-aware soft retention significantly improves rare-class tail metrics over hard deletion "
            f"under high asymmetric noise on {tests['pass_count']} datasets; remaining dataset-level effects are reported without overclaim."
        )
    elif verdict == "narrow":
        passed = [dataset for dataset, ok in dataset_pass.items() if ok]
        claim = (
            "Evidence-aware soft retention significantly improves rare-class tail metrics over hard deletion "
            f"on {passed[0] if passed else 'one dataset'}; other datasets are trends or nulls and should not be described as significant."
        )
    else:
        claim = (
            "P2f does not support a significant evidence-retention advantage over hard deletion; use fallback A and frame Graph-CoLD as an audit/boundary result."
        )
    return {
        "verdict": verdict,
        "dataset_pass": dataset_pass,
        "bounded_statement": claim,
        "dataset_trends": trend,
    }


def _trend_text(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for dataset, part in frame.groupby("reported_as", dropna=False):
        pivot = part.pivot_table(index=["tail_flip_rate", "seed"], columns="method", values=["tail_macro_f1", "rare_recovery_rate"])
        rows.append(
            {
                "dataset": dataset,
                "soft_tail_macro_f1_delta": _pivot_delta(pivot, "tail_macro_f1", "Graph-CoLD-soft"),
                "semisup_tail_macro_f1_delta": _pivot_delta(pivot, "tail_macro_f1", "Graph-CoLD-semisup"),
                "soft_corrected_recovery_delta": _pivot_delta(pivot, "rare_recovery_rate", "Graph-CoLD-soft"),
                "semisup_corrected_recovery_delta": _pivot_delta(pivot, "rare_recovery_rate", "Graph-CoLD-semisup"),
            }
        )
    return rows


def _pivot_delta(pivot: pd.DataFrame, metric: str, method: str) -> float:
    if (metric, method) not in pivot.columns or (metric, "ablation_hard") not in pivot.columns:
        return 0.0
    diff = pivot[(metric, method)] - pivot[(metric, "ablation_hard")]
    return float(diff.dropna().mean()) if not diff.dropna().empty else 0.0


def _plot_powered_tail(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby(["reported_as", "tail_flip_rate", "method"], dropna=False)["tail_macro_f1"].mean().reset_index()
    datasets = list(summary["reported_as"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 3.6), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        part = summary[summary["reported_as"] == dataset]
        for method, sub in part.groupby("method", sort=False):
            ax.plot(sub["tail_flip_rate"], sub["tail_macro_f1"], marker="o", label=method)
        ax.set_title(dataset)
        ax.set_xlabel("Tail asymmetric rate")
        ax.set_ylabel("Tail Macro-F1")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_recovery(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby(["reported_as", "method"], dropna=False)["rare_recovery_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    labels = list(summary["reported_as"].drop_duplicates())
    methods = list(METHODS)
    x = np.arange(len(labels))
    width = 0.18
    for idx, method in enumerate(methods):
        vals = [float(summary[(summary["reported_as"] == dataset) & (summary["method"] == method)]["rare_recovery_rate"].mean()) for dataset in labels]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Corrected rare-recovery")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    tests = pd.DataFrame(report["tests"]["tests"])
    summary = pd.DataFrame(report["summary"])
    lines = [
        "# P2f Tighten Report",
        "",
        "## 1. P2e Gate Note",
        "",
        f"- Soft-not-hard probe passed: {report['p2e_gate_note']['soft_not_hard_probe_passed']}",
        f"- Tension regenerated: {report['p2e_gate_note']['tension_regenerated']}",
        f"- Max tension rate: {report['p2e_gate_note']['max_tension_rate']:.6f}",
        "",
        "## 2. Corrected Rare-recovery",
        "",
        "- Definition: recovered samples are clean rare/tail suspicious samples whose final trained classifier predicts the clean true class.",
        "- Retention is reported separately and does not enter the recovery numerator.",
        "- An out-of-fold recovery column is included only as a memorization-sensitivity audit.",
        f"- Non-constant proof: hard max={report['corrected_metric']['non_constant_examples']['hard_max']:.6f}; soft min={report['corrected_metric']['non_constant_examples']['soft_min']:.6f}.",
        "",
        "## 3. Powered Per-dataset Results",
        "",
        f"- Paired rows per dataset/metric: {report['scope']['paired_rows_per_dataset_metric']}",
        f"- Seeds: {report['scope']['seeds']}",
        f"- Test size target: {report['scope']['test_size']}",
        "",
        _frame_to_md(tests) if not tests.empty else "_No tests._",
        "",
        "## 4. Pre-registered Verdict",
        "",
        f"- Global verdict: `{report['tests']['verdict']}`",
        f"- Dataset pass map: `{json.dumps(report['tests']['dataset_pass'], sort_keys=True)}`",
        "",
        "## 5. Claims Input",
        "",
        report["claims_input"]["bounded_statement"],
        "",
        "## 6. Summary Table",
        "",
        _frame_to_md(summary) if not summary.empty else "_No summary._",
        "",
        "## 7. Honest Reject-risk Re-estimate",
        "",
        _risk_text(report["tests"]["verdict"]),
        "",
        "## 8. Reproduction Commands",
        "",
    ]
    lines.extend(f"- `{cmd}`" for cmd in report["reproduction_commands"])
    lines.append("")
    return "\n".join(lines)


def _risk_text(verdict: str) -> str:
    if verdict == "robust":
        return "Reject risk is reduced for the evidence-retention claim, but the manuscript must still avoid presenting corrected rare-recovery as an ERR substitute."
    if verdict == "narrow":
        return "Reject risk remains material: the evidence-retention benefit is dataset-bounded and should be written as a narrow empirical result."
    return "Reject risk remains high for any positive evidence-retention claim. The safest manuscript path is fallback A."


def _frame_to_md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{float(value):.6f}")
        else:
            display[col] = display[col].map(lambda value: json.dumps(value) if isinstance(value, (dict, list)) else str(value))
    cols = list(display.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()
    print(
        json.dumps(
            _jsonable(
                run_p2f_tighten(
                    configs_dir=args.configs,
                    out_dir=args.out,
                    reports_dir=args.reports,
                    tables_dir=args.tables,
                    figures_dir=args.figures,
                    train_size=args.train_size,
                    test_size=args.test_size,
                )
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
