"""P2e salvage gate for evidence-preserving retention.

This module starts with the integrity gate required by P2e. It deliberately
separates the cheap tension check from any redesign or final-result evaluation:
the offline measurement may use ``flip_mask`` to label diagnostic groups, but
Graph-CDM/evidence/graph construction still receive only observed noisy labels
and unsupervised feature anomaly.
"""
from __future__ import annotations

import argparse
import inspect
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import yaml

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


DATASETS = ("cicids2017", "cesnet_tls_year22", "unsw_nb15")
RATES = (0.4, 0.6)
PART_B_RATES = (0.4, 0.6, 0.8)
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD-soft", "Graph-CoLD-semisup", "Graph-CoLD")
DEFAULT_TRAIN_SIZE = 10_000
DEFAULT_TEST_SIZE = 1_000
TENSION_THRESHOLD = 0.05
RETENTION_THRESHOLD = 0.1
SOFT_CFG = {"evidence_preserving": {"theta": 0.5, "kappa": 4.0, "rho": 0.2, "lambda_rescue": 0.8}}


def run_p2e_salvage(
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    figures_dir: str | Path = "figures",
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    rates: tuple[float, ...] = RATES,
    run_part_b: bool = True,
) -> dict[str, Any]:
    """Run A0/A1 and stop unless the pre-redesign tension gate passes."""

    configs = Path(configs_dir)
    out = Path(out_dir)
    reports = Path(reports_dir)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    for directory in (out, reports, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    cfg = _load_model_cfg(configs)
    theta = float(cfg.get("evidence_preserving", {}).get("theta", 0.5))
    a0 = _a0_gate()
    tension = _run_tension_gate(configs, reports, figures, train_size, test_size, rates, theta)
    proceed = bool(a0["passed"] and tension["gate_passed"])
    part_b = None
    if proceed and run_part_b:
        _write_preregistration(reports)
        part_b = _run_part_b(configs, out, reports, tables, figures, train_size, test_size)
    report = {
        "stage": "P2e",
        "part": "A0_A1_tension_gate_and_part_b" if part_b is not None else "A0_A1_tension_gate",
        "completed": True,
        "a0_gate": a0,
        "a1_tension_gate": tension,
        "decision": _decision(proceed, part_b),
        "part_b_run": part_b is not None,
        "part_b": part_b,
        "reason": (
            "Part B ran on pre-registered tail-asymmetric real-data comparisons."
            if part_b is not None
            else "Clean rare/tail samples are flagged suspicious often enough to test a rescue mechanism."
            if proceed
            else "Tension gate failed or A0 failed; do not tune a rescue mechanism without measurable tension."
        ),
        "reproduction_commands": [
            "python -m pytest tests/test_no_oracle_leakage.py -q",
            (
                "python -m src.paper.p2e_salvage --configs configs --out results --reports reports "
                "--tables tables --figures figures --train-size 10000 --test-size 1000"
            ),
        ],
    }
    (reports / "p2e_tension_gate.json").write_text(json.dumps(_jsonable(tension), indent=2, allow_nan=False), encoding="utf-8")
    (reports / "p2e_tension_gate.md").write_text(_tension_md(tension), encoding="utf-8")
    (reports / "p2e_salvage.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8")
    (reports / "p2e_salvage.md").write_text(_salvage_md(report), encoding="utf-8")
    return report


def _a0_gate() -> dict[str, Any]:
    p2d_report = Path("reports/p2d_clean_rerun.json")
    p2d_data = json.loads(p2d_report.read_text(encoding="utf-8")) if p2d_report.exists() else {}
    cdm_source = inspect.getsource(d5._cdm_from_observed_labels)
    context_source = inspect.getsource(d5._graphcold_context)
    graph_source = inspect.getsource(d5._lightweight_graph)
    checks = {
        "p2d_report_exists": p2d_report.exists(),
        "p2d_completed": bool(p2d_data.get("completed", False)),
        "p2d_gate_passed": bool(p2d_data.get("p2c_gate", {}).get("passed", False)),
        "cdm_observed_only_no_flip_mask": "flip" not in cdm_source and "clean" not in cdm_source,
        "context_deletes_flip_arg": "del seed, flip" in context_source,
        "context_uses_noisy_observed_labels": "observed = bundle.dataset.y_train if noisy is None else np.asarray(noisy)" in context_source,
        "graph_ignores_y_train": "y_train" not in graph_source,
        "evidence_source_observed_noisy": p2d_data.get("p2c_gate", {}).get("evidence_source")
        == "observed_noisy_labels_plus_unsupervised_feature_anomaly",
        "cicids_exact_dedup_applied": int(p2d_data.get("p2c_gate", {}).get("cicids_exact_dedup_removed", 0)) > 0,
        "split_crossing_edges_zero": int(p2d_data.get("p2c_gate", {}).get("split_crossing_edges", -1)) == 0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "note": "Run tests/test_no_oracle_leakage.py as the executable A0 guard.",
    }


def _run_tension_gate(
    configs: Path,
    reports: Path,
    figures: Path,
    train_size: int,
    test_size: int,
    rates: tuple[float, ...],
    theta: float,
) -> dict[str, Any]:
    scale = d5.write_scale_policy_report(reports)
    rows: list[dict[str, Any]] = []
    hist: list[dict[str, Any]] = []
    for dataset_name in DATASETS:
        sample = p2d._sample_bundle(dataset_name, 42, configs, reports, scale, train_size, test_size)
        tail_labels = _tail_malicious_labels(sample.y_train, sample.benign_class)
        for rate in rates:
            noisy, flip = _inject_tail_asymmetric(sample.y_train, tail_labels, sample.benign_class, rate, seed=42)
            evidence = compute_evidence(
                noisy,
                {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
                anomaly=sample.anomaly,
            )
            cdm = d5._cdm_from_observed_labels(noisy, evidence, sample.graph, sample.num_classes)
            clean = ~flip
            rare = np.isin(sample.y_train, tail_labels)
            clean_rare = clean & rare
            noisy_flipped = flip
            tension_rate = float(np.mean(cdm[clean_rare] > theta)) if clean_rare.any() else 0.0
            noisy_flag_rate = float(np.mean(cdm[noisy_flipped] > theta)) if noisy_flipped.any() else 0.0
            row = {
                "dataset": sample.dataset_key,
                "reported_as": sample.reported_as,
                "sample_policy": sample.sample_policy,
                "train_rows": int(sample.X_train.shape[0]),
                "num_classes": int(sample.num_classes),
                "tail_labels": [int(label) for label in tail_labels],
                "tail_label_names": _label_names(sample, tail_labels),
                "noise_type": "tail_asymmetric",
                "tail_flip_rate": float(rate),
                "effective_flip_rate_train": float(np.mean(flip)),
                "theta": float(theta),
                "clean_rare_count": int(clean_rare.sum()),
                "flipped_count": int(noisy_flipped.sum()),
                "tension_rate_clean_rare_cdm_gt_theta": tension_rate,
                "noisy_flag_rate_cdm_gt_theta": noisy_flag_rate,
                "clean_rare_cdm_mean": _mean(cdm[clean_rare]),
                "noisy_flipped_cdm_mean": _mean(cdm[noisy_flipped]),
                "clean_rare_cdm_q90": _quantile(cdm[clean_rare], 0.90),
                "noisy_flipped_cdm_q90": _quantile(cdm[noisy_flipped], 0.90),
            }
            rows.append(row)
            hist.append(
                {
                    "dataset": sample.reported_as,
                    "rate": rate,
                    "clean_rare_cdm": cdm[clean_rare].copy(),
                    "noisy_flipped_cdm": cdm[noisy_flipped].copy(),
                    "theta": theta,
                }
            )
    frame = pd.DataFrame(rows)
    out_csv = reports / "p2e_tension_gate.csv"
    frame.to_csv(out_csv, index=False)
    figure_path = figures / "fig_p2e_cdm_tension.pdf"
    _plot_tension(hist, figure_path)
    max_tension = float(frame["tension_rate_clean_rare_cdm_gt_theta"].max()) if not frame.empty else 0.0
    pooled_clean_rare = int(frame["clean_rare_count"].sum()) if not frame.empty else 0
    weighted_tension = (
        float(np.average(frame["tension_rate_clean_rare_cdm_gt_theta"], weights=frame["clean_rare_count"]))
        if pooled_clean_rare > 0
        else 0.0
    )
    gate_passed = bool(max_tension >= TENSION_THRESHOLD)
    return {
        "stage": "P2e-A1",
        "completed": True,
        "real_data_only": True,
        "scale_policy": f"tail_preserving_real_audit_window_train_{train_size}_test_{test_size}",
        "datasets": [p2d.DATASET_LABELS[name] for name in DATASETS],
        "rates": [float(rate) for rate in rates],
        "theta": float(theta),
        "gate_threshold": TENSION_THRESHOLD,
        "gate_rule": "pass if any real dataset/rate has clean-rare GraphCDM tension >= 5%",
        "gate_passed": gate_passed,
        "max_tension_rate": max_tension,
        "pooled_clean_rare_weighted_tension_rate": weighted_tension,
        "clean_rare_total": pooled_clean_rare,
        "figure": str(figure_path).replace("\\", "/"),
        "csv": str(out_csv).replace("\\", "/"),
        "rows": rows,
    }


def _run_part_b(
    configs: Path,
    out: Path,
    reports: Path,
    tables: Path,
    figures: Path,
    train_size: int,
    test_size: int,
) -> dict[str, Any]:
    scale = d5.write_scale_policy_report(reports)
    rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    weight_samples: list[pd.DataFrame] = []
    for dataset_name in DATASETS:
        for seed in d5.SEEDS:
            sample = p2d._sample_bundle(dataset_name, seed, configs, reports, scale, train_size, test_size)
            tail_labels = _tail_malicious_labels(sample.y_train, sample.benign_class)
            for rate in PART_B_RATES:
                print(f"[p2e] {dataset_name} seed={seed} tail_asymmetric={rate:.0%}", flush=True)
                noisy, flip = _inject_tail_asymmetric(sample.y_train, tail_labels, sample.benign_class, rate, seed=seed)
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
                    weights, train_labels, training_info = _method_training_plan(
                        method,
                        sample,
                        noisy,
                        cdm,
                        evidence,
                        hard_weights,
                        soft_rescue,
                        seed,
                    )
                    y_train_pred, y_test_pred = _fit_predict_train_test(
                        sample.X_train,
                        train_labels,
                        sample.X_test,
                        weights,
                        method,
                        seed,
                    )
                    y_train_eval_pred = _crossfit_train_predictions(sample.X_train, train_labels, weights, method, seed)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    runtime_sec = time.perf_counter() - start
                    row = _part_b_row(
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
                        y_train_eval_pred,
                        y_test_pred,
                        training_info,
                        runtime_sec,
                        peak / (1024 * 1024),
                    )
                    rows.append(row)
                    tail_rows.extend(
                        _tail_breakdown_rows(
                            sample,
                            method,
                            rate,
                            seed,
                            tail_labels,
                            flip,
                            cdm,
                            evidence,
                            weights,
                            y_train_eval_pred,
                            y_test_pred,
                        )
                    )
                    if seed == 0:
                        weight_samples.append(_weight_sample_frame(method, sample.reported_as, rate, weights))
    frame = pd.DataFrame(rows)
    tail_frame = pd.DataFrame(tail_rows)
    results_csv = out / "p2e_tail_salvage.csv"
    tail_csv = tables / "table_p2e_tail_breakdown.csv"
    frame.to_csv(results_csv, index=False)
    tail_frame.to_csv(tail_csv, index=False)

    tests = _success_tests(frame)
    tests_csv = tables / "table_p2e_success_tests.csv"
    pd.DataFrame(tests["tests"]).to_csv(tests_csv, index=False)
    _plot_tail_recovery(frame, figures / "fig_p2e_tail_recovery.pdf")
    _plot_weight_hist(weight_samples, figures / "fig_p2e_weight_hist.pdf")
    stats_report = {
        "completed": True,
        "real_data_only": True,
        "pre_registration": str(reports / "p2e_preregistration.md"),
        "scale_policy": f"tail_preserving_real_salvage_window_train_{train_size}_test_{test_size}",
        "datasets": [p2d.DATASET_LABELS[name] for name in DATASETS],
        "noise": {
            "type": "tail_asymmetric",
            "rates": [float(rate) for rate in PART_B_RATES],
            "seeds": list(d5.SEEDS),
        },
        "methods": list(METHODS),
        "soft_weight": SOFT_CFG["evidence_preserving"],
        "outputs": {
            "tail_salvage_csv": str(results_csv).replace("\\", "/"),
            "tail_breakdown_csv": str(tail_csv).replace("\\", "/"),
            "success_tests_csv": str(tests_csv).replace("\\", "/"),
            "tail_recovery_figure": "figures/fig_p2e_tail_recovery.pdf",
            "weight_hist_figure": "figures/fig_p2e_weight_hist.pdf",
        },
        "row_count": int(len(frame)),
        "tail_breakdown_rows": int(len(tail_frame)),
        "summary": _part_b_summary(frame),
        "success_tests": tests,
    }
    (reports / "p2e_salvage_stats.json").write_text(json.dumps(_jsonable(stats_report), indent=2, allow_nan=False), encoding="utf-8")
    return stats_report


def _method_training_plan(
    method: str,
    sample: Any,
    noisy: np.ndarray,
    cdm: np.ndarray,
    evidence: np.ndarray,
    hard_weights: np.ndarray,
    soft_rescue: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if method == "CoLD":
        return d5._weights_for_cold(cdm, evidence), noisy.copy(), {"training_mode": "cold_hard"}
    if method == "ablation_hard":
        return hard_weights.copy(), noisy.copy(), {"training_mode": "hard_deletion"}
    if method in {"Graph-CoLD-soft", "Graph-CoLD"}:
        return soft_rescue.copy(), noisy.copy(), {"training_mode": "scalar_soft"}
    if method == "Graph-CoLD-semisup":
        semisup_labels, info = _evidence_semisup_labels(sample, noisy, cdm, evidence, soft_rescue, seed)
        return soft_rescue.copy(), semisup_labels, info
    raise ValueError(f"Unknown P2e method: {method}")


def _evidence_semisup_labels(
    sample: Any,
    noisy: np.ndarray,
    cdm: np.ndarray,
    evidence: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    theta = float(SOFT_CFG["evidence_preserving"]["theta"])
    high_evidence = evidence >= _safe_quantile(evidence, 0.75)
    candidates = (cdm > theta) & high_evidence & (weights >= RETENTION_THRESHOLD)
    reliable = cdm <= theta
    teacher = ExtraTreesClassifier(n_estimators=24, random_state=seed + 271, class_weight="balanced", n_jobs=-1)
    if reliable.sum() >= 2 and np.unique(noisy[reliable]).size >= 2:
        teacher.fit(sample.X_train[reliable], noisy[reliable])
    else:
        teacher.fit(sample.X_train, noisy)
    pseudo = teacher.predict(sample.X_train)
    confidence = np.ones(sample.X_train.shape[0], dtype=np.float64)
    if hasattr(teacher, "predict_proba"):
        proba = teacher.predict_proba(sample.X_train)
        confidence = np.max(proba, axis=1)
    selected = candidates & (confidence >= 0.50)
    labels = noisy.copy()
    labels[selected] = pseudo[selected]
    return labels, {
        "training_mode": "evidence_semisup",
        "semisup_candidate_count": int(candidates.sum()),
        "semisup_selected_count": int(selected.sum()),
        "semisup_selected_rate": float(selected.mean()),
        "semisup_confidence_mean": _mean(confidence[selected]),
    }


def _fit_predict_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    weights: np.ndarray,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = _fit_model(X_train, y_train, weights, method, seed)
    return model.predict(X_train), model.predict(X_test)


def _crossfit_train_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    labels, counts = np.unique(y_train, return_counts=True)
    n_splits = int(min(3, counts.min(initial=0))) if labels.size >= 2 else 0
    if n_splits < 2:
        return _fit_model(X_train, y_train, weights, method, seed + 19).predict(X_train)
    pred = np.empty(y_train.shape[0], dtype=np.int64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1901)
    for fold, (fit_idx, eval_idx) in enumerate(splitter.split(X_train, y_train)):
        model = _fit_model(X_train[fit_idx], y_train[fit_idx], weights[fit_idx], method, seed + 37 + fold)
        pred[eval_idx] = model.predict(X_train[eval_idx])
    return pred


def _fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    method: str,
    seed: int,
) -> ExtraTreesClassifier:
    y_train = np.asarray(y_train, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if method in {"CoLD", "ablation_hard"}:
        keep = weights >= 0.5
        model = ExtraTreesClassifier(n_estimators=24, random_state=seed, class_weight="balanced", n_jobs=-1)
        if keep.sum() >= 2 and np.unique(y_train[keep]).size >= 2:
            model.fit(X_train[keep], y_train[keep])
        else:
            model.fit(X_train, y_train)
    else:
        retained_weight = np.where(weights >= RETENTION_THRESHOLD, weights, 0.0)
        sample_weight = np.clip(retained_weight, 0.0, 1.0) * _class_balance_weights(y_train)
        if float(np.sum(sample_weight)) <= 1e-12:
            sample_weight = _class_balance_weights(y_train)
        model = ExtraTreesClassifier(n_estimators=24, random_state=seed, class_weight=None, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _part_b_row(
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
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    training_info: dict[str, Any],
    runtime_sec: float,
    memory_mb: float,
) -> dict[str, Any]:
    clean = ~np.asarray(flip, dtype=bool)
    binary_err = _binary_err(weights, evidence, clean, sample.y_train, flip)
    degree_err = degree_evidence_retention_components(weights, evidence, clean, sample.y_train)
    recovery = rare_evidence_recovery_rate(
        weights,
        sample.y_train,
        y_train_pred,
        clean,
        suspicious,
        tail_labels,
        retention_threshold=RETENTION_THRESHOLD,
    )
    retained = np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD
    clean_rare_suspicious = clean & suspicious & np.isin(sample.y_train, tail_labels)
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
        "tail_macro_f1": _tail_macro_f1(sample.y_test, y_test_pred, tail_labels),
        "tail_recall": _tail_recall(sample.y_test, y_test_pred, tail_labels),
        "rare_recovery_rate": recovery["rare_recovery_rate"],
        "rare_retained_rate": recovery["rare_retained_rate"],
        "rare_clean_suspicious_count": recovery["rare_clean_suspicious_count"],
        "rare_recovered_count": recovery["rare_recovered_count"],
        "rare_recovery_eval": "out_of_fold_training_predictions",
        "rare_retained_count": recovery["rare_retained_count"],
        "err": binary_err["err"],
        "err_tail": binary_err["err_tail"],
        "err_final": binary_err["err_final"],
        "err_degree": degree_err["err_degree"],
        "err_tail_degree": degree_err["err_tail_degree"],
        "err_final_degree": degree_err["err_final_degree"],
        "mean_weight": float(np.mean(weights)),
        "retained_fraction": float(np.mean(retained)),
        "clean_rare_suspicious_retained_fraction": float(np.mean(retained[clean_rare_suspicious])) if clean_rare_suspicious.any() else 0.0,
        "soft_retained_hard_deleted_count": int(np.sum((weights >= RETENTION_THRESHOLD) & (d5._weights_for_hard(cdm, evidence) < RETENTION_THRESHOLD))),
        "semisup_candidate_count": int(training_info.get("semisup_candidate_count", 0)),
        "semisup_selected_count": int(training_info.get("semisup_selected_count", 0)),
        "semisup_selected_rate": float(training_info.get("semisup_selected_rate", 0.0)),
        "semisup_confidence_mean": float(training_info.get("semisup_confidence_mean", 0.0)),
        "runtime_sec": float(runtime_sec),
        "memory_mb": float(memory_mb),
        "tail_labels": json.dumps([int(label) for label in tail_labels]),
        "tail_label_names": json.dumps(_label_names(sample, tail_labels)),
        "active_views": sample.active_views,
        "source_verified": sample.source_verified,
        "replacement_for": sample.replacement_for,
    }


def _tail_breakdown_rows(
    sample: Any,
    method: str,
    rate: float,
    seed: int,
    tail_labels: np.ndarray,
    flip: np.ndarray,
    cdm: np.ndarray,
    evidence: np.ndarray,
    weights: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean = ~np.asarray(flip, dtype=bool)
    retained = np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD
    suspicious = cdm > float(SOFT_CFG["evidence_preserving"]["theta"])
    names = _label_names(sample, tail_labels)
    for label, name in zip(tail_labels, names):
        train_mask = sample.y_train == int(label)
        test_mask = sample.y_test == int(label)
        clean_suspicious = train_mask & clean & suspicious
        recovered = clean_suspicious & retained & (y_train_pred == sample.y_train)
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
                "clean_suspicious_tail_count": int(clean_suspicious.sum()),
                "clean_suspicious_tail_retained_rate": float(np.mean(retained[clean_suspicious])) if clean_suspicious.any() else 0.0,
                "clean_suspicious_tail_recovery_rate": float(np.mean(recovered[clean_suspicious])) if clean_suspicious.any() else 0.0,
                "test_tail_recall": float(np.mean(y_test_pred[test_mask] == sample.y_test[test_mask])) if test_mask.any() else np.nan,
                "mean_tail_evidence": _mean(evidence[train_mask]),
                "mean_tail_cdm": _mean(cdm[train_mask]),
                "mean_tail_weight": _mean(weights[train_mask]),
            }
        )
    return rows


def _success_tests(frame: pd.DataFrame) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    candidate_methods = ("Graph-CoLD-soft", "Graph-CoLD-semisup")
    metrics = ("tail_macro_f1", "rare_recovery_rate")
    for dataset in [*sorted(frame["reported_as"].unique()), "pooled"]:
        part = frame if dataset == "pooled" else frame[frame["reported_as"] == dataset]
        for method in candidate_methods:
            no_regression = _no_macro_regression(part, method)
            for metric in metrics:
                stat = _paired_metric_test(part, method, "ablation_hard", metric)
                stat.update(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "no_significant_aggregate_macro_f1_regression": no_regression["passed"],
                        "aggregate_macro_f1_regression_p_less": no_regression["p_less"],
                        "aggregate_macro_f1_delta_vs_hard": no_regression["mean_delta"],
                    }
                )
                stat["meets_success"] = bool(
                    dataset != "pooled"
                    and metric in metrics
                    and stat["mean_delta"] > 0.0
                    and stat["p_value_greater"] < 0.05
                    and stat["cohens_dz"] >= 0.3
                    and no_regression["passed"]
                )
                tests.append(stat)
    success = any(item["meets_success"] for item in tests)
    positive_partial = any(
        item["dataset"] != "pooled"
        and item["method"] in candidate_methods
        and item["metric"] in metrics
        and item["mean_delta"] > 0.0
        and item["cohens_dz"] >= 0.3
        for item in tests
    )
    verdict = "salvaged" if success else "partial" if positive_partial else "null"
    return {
        "pre_registered_success_criterion": (
            "Evidence-semisup or scalar soft beats hard deletion on tail Macro-F1 or rare-evidence recovery "
            "with paired p<0.05 and Cohen dz>=0.3 on at least one real dataset under asymmetric >=40%, "
            "and without significant aggregate Macro-F1 regression."
        ),
        "verdict": verdict,
        "success": bool(success),
        "partial_signal": bool(positive_partial),
        "tests": tests,
    }


def _paired_metric_test(frame: pd.DataFrame, method: str, baseline: str, metric: str) -> dict[str, Any]:
    pivot = frame[frame["method"].isin([method, baseline])].pivot_table(
        index=["reported_as", "tail_flip_rate", "seed"],
        columns="method",
        values=metric,
    )
    if method not in pivot.columns or baseline not in pivot.columns:
        return _empty_stat()
    pivot = pivot[[method, baseline]].dropna()
    if pivot.empty:
        return _empty_stat()
    a = pivot[method].to_numpy(dtype=float)
    b = pivot[baseline].to_numpy(dtype=float)
    diff = a - b
    p_value = _paired_pvalue(a, b, alternative="greater")
    dz = _cohens_dz(diff)
    return {
        "paired_rows": int(len(pivot)),
        "method_mean": float(np.mean(a)),
        "baseline_mean": float(np.mean(b)),
        "mean_delta": float(np.mean(diff)),
        "p_value_greater": p_value,
        "cohens_dz": dz,
    }


def _no_macro_regression(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    pivot = frame[frame["method"].isin([method, "ablation_hard"])].pivot_table(
        index=["reported_as", "tail_flip_rate", "seed"],
        columns="method",
        values="macro_f1",
    )
    if method not in pivot.columns or "ablation_hard" not in pivot.columns:
        return {"passed": False, "p_less": 1.0, "mean_delta": 0.0}
    pivot = pivot[[method, "ablation_hard"]].dropna()
    if pivot.empty:
        return {"passed": False, "p_less": 1.0, "mean_delta": 0.0}
    a = pivot[method].to_numpy(dtype=float)
    b = pivot["ablation_hard"].to_numpy(dtype=float)
    diff = a - b
    p_less = _paired_pvalue(a, b, alternative="less")
    significant_regression = bool(float(np.mean(diff)) < 0.0 and p_less < 0.05)
    return {"passed": not significant_regression, "p_less": p_less, "mean_delta": float(np.mean(diff))}


def _part_b_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summary = (
        frame.groupby(["reported_as", "method"], dropna=False)[
            ["macro_f1", "tail_macro_f1", "tail_recall", "rare_recovery_rate", "rare_retained_rate", "err_final_degree"]
        ]
        .mean()
        .reset_index()
    )
    return summary.to_dict(orient="records")


def _plot_tail_recovery(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby(["tail_flip_rate", "method"], dropna=False)["rare_recovery_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for method, sub in summary.groupby("method", sort=False):
        ax.plot(sub["tail_flip_rate"], sub["rare_recovery_rate"], marker="o", label=method)
    ax.set_xlabel("Tail-concentrated asymmetric noise rate")
    ax.set_ylabel("Rare-evidence recovery rate")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_weight_hist(weight_samples: list[pd.DataFrame], path: Path) -> None:
    if not weight_samples:
        return
    frame = pd.concat(weight_samples, ignore_index=True)
    methods = [method for method in METHODS if method in set(frame["method"])]
    fig, axes = plt.subplots(len(methods), 1, figsize=(7.0, max(2.2, 1.65 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        vals = frame.loc[frame["method"] == method, "weight"].to_numpy(dtype=float)
        ax.hist(vals, bins=np.linspace(0.0, 1.0, 26), color="#4e79a7", alpha=0.78)
        ax.axvline(RETENTION_THRESHOLD, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_ylabel(method)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Training weight")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _weight_sample_frame(method: str, dataset: str, rate: float, weights: np.ndarray) -> pd.DataFrame:
    weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(42)
    if weights.shape[0] > 1200:
        idx = rng.choice(weights.shape[0], size=1200, replace=False)
        weights = weights[idx]
    return pd.DataFrame({"method": method, "dataset": dataset, "tail_flip_rate": float(rate), "weight": weights})


def _binary_err(weights: np.ndarray, evidence: np.ndarray, clean: np.ndarray, y: np.ndarray, flip: np.ndarray) -> dict[str, float]:
    if not np.asarray(flip, dtype=bool).any():
        return {"err": 1.0, "err_tail": 1.0, "err_final": 1.0}
    return d5.evidence_retention_components(weights, evidence, clean, y, retention_threshold=RETENTION_THRESHOLD)


def _tail_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, tail_labels: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    mask = np.isin(y_true, tail_labels)
    if not mask.any():
        return 0.0
    labels = sorted(int(label) for label in np.unique(y_true[mask]))
    return float(f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0))


def _tail_recall(y_true: np.ndarray, y_pred: np.ndarray, tail_labels: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    mask = np.isin(y_true, tail_labels)
    if not mask.any():
        return 0.0
    return float(np.mean(y_pred[mask] == y_true[mask]))


def _class_balance_weights(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train)
    labels, counts = np.unique(y, return_counts=True)
    weights = {label: y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float64)


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
    p_value = float(stats.ttest_rel(a, b, alternative=alternative).pvalue)
    return float(max(p_value, 1e-12))


def _cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if diff.shape[0] < 2:
        return 0.0
    std = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    if std <= 1e-12:
        return 99.0 if mean > 0.0 else -99.0 if mean < 0.0 else 0.0
    return float(mean / std)


def _empty_stat() -> dict[str, Any]:
    return {"paired_rows": 0, "method_mean": 0.0, "baseline_mean": 0.0, "mean_delta": 0.0, "p_value_greater": 1.0, "cohens_dz": 0.0}


def _safe_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _write_preregistration(reports: Path) -> None:
    path = reports / "p2e_preregistration.md"
    if path.exists():
        return
    lines = [
        "# P2e Preregistration",
        "",
        "This file is written before final redesigned comparison numbers are read.",
        "",
        "## Scope",
        "",
        "- Datasets: CICIDS-2017, CESNET-TLS-Year22, UNSW-NB15 real local audit windows.",
        "- Noise: tail-concentrated asymmetric rates 40%, 60%, 80%.",
        "- Seeds: 0, 1, 2.",
        "- Methods: CoLD, ablation_hard, Graph-CoLD-soft, Graph-CoLD-semisup, Graph-CoLD.",
        "",
        "## Success Criterion",
        "",
        "Success requires Graph-CoLD-soft or Graph-CoLD-semisup to beat ablation_hard on tail Macro-F1 or rare-evidence recovery with paired p<0.05 and Cohen dz>=0.3 on at least one real dataset under asymmetric noise >=40%, with no significant aggregate Macro-F1 regression.",
        "",
        "Partial success means a positive effect-size signal without the full significance/regression gate. Null means no positive pre-registered signal.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _decision(proceed: bool, part_b: dict[str, Any] | None) -> str:
    if not proceed:
        return "stop_recommend_fallback_A"
    if part_b is None:
        return "proceed_to_part_b"
    return f"part_b_{part_b.get('success_tests', {}).get('verdict', 'unknown')}"


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


def _tail_malicious_labels(y: np.ndarray, benign_class: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    labels, counts = np.unique(y[y != int(benign_class)], return_counts=True)
    if labels.size == 0:
        return np.asarray([], dtype=np.int64)
    threshold = np.quantile(counts, 0.5)
    tail = labels[counts <= threshold]
    return np.asarray(tail, dtype=np.int64)


def _inject_tail_asymmetric(
    y: np.ndarray,
    tail_labels: np.ndarray,
    benign_class: int,
    rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    noisy = y.copy()
    flip = np.zeros(y.shape[0], dtype=bool)
    candidates = np.flatnonzero(np.isin(y, tail_labels) & (y != int(benign_class)))
    if candidates.size == 0 or rate <= 0:
        return noisy, flip
    rng = np.random.default_rng(seed)
    n_flip = min(candidates.size, int(np.floor(float(rate) * candidates.size)))
    if n_flip <= 0:
        return noisy, flip
    selected = rng.choice(candidates, size=n_flip, replace=False)
    noisy[selected] = int(benign_class)
    flip[selected] = True
    return noisy, flip


def _label_names(sample: Any, labels: np.ndarray) -> list[str]:
    names = list(getattr(sample, "class_names", []) or [])
    if not names:
        names = list(getattr(sample, "meta", {}).get("class_names", [])) if hasattr(sample, "meta") else []
    if not names and hasattr(sample, "dataset"):
        names = list(getattr(sample.dataset, "meta", {}).get("class_names", []))
    out = []
    for label in labels:
        idx = int(label)
        out.append(str(names[idx]) if 0 <= idx < len(names) else str(idx))
    return out


def _plot_tension(hist: list[dict[str, Any]], path: Path) -> None:
    datasets = list(dict.fromkeys(item["dataset"] for item in hist))
    rates = list(dict.fromkeys(float(item["rate"]) for item in hist))
    fig, axes = plt.subplots(len(datasets), len(rates), figsize=(4.3 * len(rates), 2.7 * len(datasets)), squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, rate in enumerate(rates):
            ax = axes[row_idx, col_idx]
            item = next((x for x in hist if x["dataset"] == dataset and np.isclose(float(x["rate"]), rate)), None)
            if item is None:
                ax.axis("off")
                continue
            clean_vals = np.asarray(item["clean_rare_cdm"], dtype=np.float64)
            noisy_vals = np.asarray(item["noisy_flipped_cdm"], dtype=np.float64)
            bins = np.linspace(0.0, 1.0, 31)
            if clean_vals.size:
                ax.hist(clean_vals, bins=bins, alpha=0.62, label="clean rare", density=True, color="#4e79a7")
            if noisy_vals.size:
                ax.hist(noisy_vals, bins=bins, alpha=0.50, label="flipped", density=True, color="#e15759")
            ax.axvline(float(item["theta"]), color="#333333", linewidth=1.0, linestyle="--")
            ax.set_title(f"{dataset}, tail asym {rate:.0%}")
            ax.set_xlabel("GraphCDM")
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _load_model_cfg(configs: Path) -> dict[str, Any]:
    path = configs / "model.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else 0.0


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if values.size else 0.0


def _tension_md(report: dict[str, Any]) -> str:
    frame = pd.DataFrame(report["rows"])
    lines = [
        "# P2e Tension Gate",
        "",
        f"- Completed: {report['completed']}",
        f"- Real data only: {report['real_data_only']}",
        f"- Scale policy: `{report['scale_policy']}`",
        f"- Gate rule: {report['gate_rule']}",
        f"- Gate threshold: {report['gate_threshold']:.2%}",
        f"- Gate passed: {report['gate_passed']}",
        f"- Max tension rate: {report['max_tension_rate']:.4f}",
        f"- Pooled clean-rare weighted tension rate: {report['pooled_clean_rare_weighted_tension_rate']:.4f}",
        f"- Figure: `{report['figure']}`",
        f"- CSV: `{report['csv']}`",
        "",
        "## Rows",
        "",
        _frame_to_md(frame) if not frame.empty else "_No rows._",
        "",
    ]
    if not report["gate_passed"]:
        lines.extend(
            [
                "## Decision",
                "",
                "The measured tension is below the pre-redesign threshold. There is no visible pool of clean rare samples that Graph-CDM marks suspicious, so Part B should not be run. Recommend fallback A: write the method as an audit/boundary result rather than claiming evidence-preserving rescue.",
                "",
            ]
        )
    return "\n".join(lines)


def _frame_to_md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{float(value):.6f}")
        else:
            display[col] = display[col].map(lambda value: json.dumps(value) if isinstance(value, list) else str(value))
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def _salvage_md(report: dict[str, Any]) -> str:
    tension = report["a1_tension_gate"]
    lines = [
        "# P2e Salvage Report",
        "",
        "## 1. A0 Gate",
        "",
        f"- Passed: {report['a0_gate']['passed']}",
        "- Executable guard: `python -m pytest tests/test_no_oracle_leakage.py -q`",
        "",
        "## 2. A1 Tension Gate",
        "",
        f"- Gate passed: {tension['gate_passed']}",
        f"- Max tension rate: {tension['max_tension_rate']:.4f}",
        f"- Pooled clean-rare weighted tension rate: {tension['pooled_clean_rare_weighted_tension_rate']:.4f}",
        f"- Figure: `{tension['figure']}`",
        "",
        "## Decision",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Reason: {report['reason']}",
        "",
    ]
    if not tension["gate_passed"]:
        lines.extend(
            [
                "## Honest Verdict",
                "",
                "- Verdict: `null` at the pre-redesign tension gate.",
                "- Recommendation: fallback A. Do not run Part B or tune a rescue mechanism unless a later real-data gate finds clean rare samples in the suspicious CDM region.",
                "- Reject-risk estimate: high if the manuscript keeps claiming evidence-preserving rescue; lower only if reframed as a de-oracled audit/boundary paper.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 3. Part B Salvage Evaluation",
                "",
            ]
        )
        part_b = report.get("part_b")
        if part_b:
            success = part_b["success_tests"]
            summary = pd.DataFrame(part_b.get("summary", []))
            tests = pd.DataFrame(success.get("tests", []))
            lines.extend(
                [
                    f"- Pre-registration: `{part_b['pre_registration']}`",
                    f"- Verdict: `{success['verdict']}`",
                    f"- Success: {success['success']}",
                    f"- Partial signal: {success['partial_signal']}",
                    f"- Results: `{part_b['outputs']['tail_salvage_csv']}`",
                    f"- Tail breakdown: `{part_b['outputs']['tail_breakdown_csv']}`",
                    f"- Success tests: `{part_b['outputs']['success_tests_csv']}`",
                    "",
                    "### Part B Summary",
                    "",
                    _frame_to_md(summary) if not summary.empty else "_No summary rows._",
                    "",
                    "### Pre-registered Tests",
                    "",
                    _frame_to_md(tests) if not tests.empty else "_No test rows._",
                    "",
                    "### Honest Verdict",
                    "",
                    _verdict_text(success["verdict"]),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "- Verdict: tension exists. Part B must write `reports/p2e_preregistration.md` before reading any final redesigned comparison numbers.",
                    "- Current report intentionally does not include redesigned weights or final tail metrics.",
                    "",
                ]
            )
    lines.extend(["## Reproduction Commands", ""])
    lines.extend(f"- `{cmd}`" for cmd in report["reproduction_commands"])
    lines.append("")
    return "\n".join(lines)


def _verdict_text(verdict: str) -> str:
    if verdict == "salvaged":
        return (
            "The pre-registered P2e salvage criterion is met: evidence-preserving soft/semi-supervised retention "
            "improves a tail evidence metric against hard deletion without a significant aggregate Macro-F1 regression."
        )
    if verdict == "partial":
        return (
            "The run found a positive tail-side signal, but it did not satisfy the full pre-registered significance "
            "and regression gate. The paper should frame this as partial operational support, not a settled win."
        )
    return (
        "The redesigned rescue did not meet the pre-registered salvage criterion. The manuscript should use fallback A "
        "and avoid claiming that evidence preservation beats hard deletion on these real-data slices."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--tension-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            _jsonable(run_p2e_salvage(
                configs_dir=args.configs,
                out_dir=args.out,
                reports_dir=args.reports,
                tables_dir=args.tables,
                figures_dir=args.figures,
                train_size=args.train_size,
                test_size=args.test_size,
                run_part_b=not args.tension_only,
            )),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
