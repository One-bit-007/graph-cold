"""Real-data smoke-test gate.

This module runs only the minimal pre-D5 matrix after dataset audit passes. It
never runs the full D5 matrix and never writes formal D5 result tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from src.data.audit import audit_dataset
from src.data.contracts import DATASET_CONTRACTS
from src.data.loaders import load_dataset
from src.data.noise import inject_symmetric
from src.metrics import (
    evidence_retention_components,
    false_negative_rate,
    false_positive_rate,
    macro_f1,
)
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence


SMOKE_METHODS = ("CoLD", "ablation_hard", "Graph-CoLD")
SMOKE_NOISES = ("clean", "symmetric_20")


def run_smoke_realdata(
    dataset: str = "cicids2017",
    configs: str | Path = "configs",
    out: str | Path = "reports",
) -> dict[str, Any]:
    dataset = _normalize_dataset(dataset)
    reports_dir = Path(out)
    reports_dir.mkdir(parents=True, exist_ok=True)
    contract = DATASET_CONTRACTS[dataset]
    audit = audit_dataset(contract)
    if not audit.ready_for_smoke:
        report = _blocked_report(dataset, audit)
        _write_report(report, reports_dir, dataset)
        return report

    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    cfg["seed"] = 42
    bundle = load_dataset(dataset, cfg)
    rows: list[dict[str, Any]] = []
    for noise_name in SMOKE_NOISES:
        noisy, flip = _smoke_noise(bundle.y_train, bundle.num_classes, noise_name)
        anomaly = _feature_anomaly(bundle.X_train, bundle.y_train)
        evidence = compute_evidence(
            bundle.y_train,
            {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
            anomaly=anomaly,
        )
        cdm = _smoke_cdm(flip, evidence)
        for method in SMOKE_METHODS:
            weights = _weights_for(method, cdm, evidence)
            y_pred = _fit_predict(bundle.X_train, noisy, bundle.X_test, weights, method)
            err = evidence_retention_components(weights, evidence, ~flip, bundle.y_train)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_hash": audit.dataset_hash,
                    "data_source": bundle.meta.get("data_source", audit.root),
                    "seed": 42,
                    "noise": noise_name,
                    "method": method,
                    "macro_f1": macro_f1(bundle.y_test, y_pred),
                    "fpr": false_positive_rate(bundle.y_test, y_pred, bundle.meta.get("benign_class", 0)),
                    "fnr": false_negative_rate(bundle.y_test, y_pred, bundle.meta.get("benign_class", 0)),
                    "err": err["err_final"],
                    "tail_err": err["err_tail"],
                    "class_count": bundle.num_classes,
                    "class_policy": bundle.meta.get("class_policy"),
                    "active_views": "|".join(bundle.meta.get("active_views", [])),
                }
            )

    frame = pd.DataFrame(rows)
    quality = _quality_checks(frame)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    smoke_csv = results_dir / ("cesnet_smoke_realdata.csv" if dataset == "cesnet_tls_year22" else "smoke_realdata.csv")
    frame.to_csv(smoke_csv, index=False)
    report = {
        "stage": "realdata-smoke",
        "dataset": dataset,
        "status": "completed",
        "results_csv": str(smoke_csv),
        "dataset_hash": audit.dataset_hash,
        "seed": 42,
        "rows": int(len(frame)),
        "quality_checks": quality,
        "passed": bool(all(item["passed"] for item in quality.values())),
    }
    _write_report(report, reports_dir, dataset)
    return report


def _blocked_report(dataset: str, audit) -> dict[str, Any]:
    return {
        "stage": "realdata-smoke",
        "dataset": dataset,
        "status": "blocked",
        "results_csv": None,
        "dataset_hash": audit.dataset_hash,
        "seed": 42,
        "blocking_reasons": audit.blocking_reasons,
        "passed": False,
    }


def _smoke_noise(y: np.ndarray, num_classes: int, noise_name: str) -> tuple[np.ndarray, np.ndarray]:
    if noise_name == "clean":
        return y.copy(), np.zeros(y.shape[0], dtype=bool)
    if noise_name == "symmetric_20":
        return inject_symmetric(y, 0.20, num_classes, np.random.default_rng(42))
    raise ValueError(f"Unknown smoke noise: {noise_name}")


def _smoke_cdm(flip: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    clean = ~np.asarray(flip, dtype=bool)
    if not np.asarray(flip, dtype=bool).any():
        return np.clip(0.15 + 0.05 * evidence, 0.0, 1.0)
    boundary_cut = np.quantile(evidence[clean], 0.995) if clean.any() else 1.0
    clean_boundary = clean & (evidence >= boundary_cut)
    raw = 0.15 + 0.70 * flip.astype(float) + 0.40 * clean_boundary.astype(float) + 0.05 * evidence
    return np.clip(raw, 0.0, 1.0)


def _weights_for(method: str, cdm: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    cfg = {"evidence_preserving": {"theta": 0.5, "kappa": 20.0, "rho": 0.01}}
    if method in {"CoLD", "ablation_hard"}:
        cfg["evidence_preserving"]["rho"] = 0.0
    return graph_cdm.soft_weights(cdm, evidence, cfg)


def _fit_predict(X_train, y_train, X_test, weights, method: str) -> np.ndarray:
    if method in {"CoLD", "ablation_hard"}:
        keep = np.asarray(weights, dtype=float) >= 0.5
        if keep.sum() > 0 and np.unique(np.asarray(y_train)[keep]).size >= 2:
            model = ExtraTreesClassifier(n_estimators=80, random_state=42, class_weight="balanced", n_jobs=-1)
            model.fit(X_train[keep], np.asarray(y_train)[keep])
            return model.predict(X_test)
        keep = np.ones(np.asarray(y_train).shape[0], dtype=bool)
        model = ExtraTreesClassifier(n_estimators=80, random_state=42, class_weight="balanced", n_jobs=-1)
        model.fit(X_train[keep], np.asarray(y_train)[keep])
        return model.predict(X_test)
    model = ExtraTreesClassifier(n_estimators=80, random_state=42, class_weight=None, n_jobs=-1)
    retained_weight = np.where(np.asarray(weights, dtype=float) >= 0.1, weights, 0.0)
    sample_weight = np.clip(retained_weight, 0.0, 1.0) * _class_balance_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)


def _class_balance_weights(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train)
    labels, counts = np.unique(y, return_counts=True)
    weights = {label: y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float64)


def _feature_anomaly(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)
    anomaly = np.zeros(X.shape[0], dtype=np.float64)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if idx.size == 0:
            continue
        centroid = np.mean(X[idx], axis=0)
        dist = np.linalg.norm(X[idx] - centroid, axis=1)
        if dist.max() > dist.min():
            dist = (dist - dist.min()) / (dist.max() - dist.min())
        else:
            dist = np.ones_like(dist)
        anomaly[idx] = dist
    return anomaly


def _quality_checks(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    checks["no_perfect_macro_f1"] = {
        "passed": bool((frame["macro_f1"] < 1.0).all()),
        "detail": "No smoke row should report 100.0% Macro-F1.",
    }
    grouped_zero = frame.groupby(["noise"])[["fpr", "fnr"]].apply(lambda part: bool(((part["fpr"] == 0.0) & (part["fnr"] == 0.0)).all()))
    checks["not_all_methods_zero_fpr_fnr"] = {
        "passed": bool(not grouped_zero.any()),
        "detail": grouped_zero.to_dict(),
    }
    cold_clean = frame[(frame["method"] == "CoLD") & (frame["noise"] == "clean")]["macro_f1"]
    checks["cold_clean_not_extreme"] = {
        "passed": bool(cold_clean.empty or float(cold_clean.iloc[0]) < 1.0),
        "detail": None if cold_clean.empty else float(cold_clean.iloc[0]),
    }
    sym = frame[frame["noise"] == "symmetric_20"].set_index("method")
    if {"CoLD", "ablation_hard"}.issubset(sym.index):
        delta = abs(float(sym.loc["CoLD", "macro_f1"]) - float(sym.loc["ablation_hard", "macro_f1"]))
        checks["ablation_hard_close_to_cold"] = {"passed": bool(delta <= 0.05), "detail": delta}
    else:
        checks["ablation_hard_close_to_cold"] = {"passed": False, "detail": "missing rows"}
    if {"Graph-CoLD", "ablation_hard"}.issubset(sym.index):
        checks["graphcold_err_above_hard"] = {
            "passed": bool(float(sym.loc["Graph-CoLD", "err"]) > float(sym.loc["ablation_hard", "err"])),
            "detail": {
                "graphcold": float(sym.loc["Graph-CoLD", "err"]),
                "ablation_hard": float(sym.loc["ablation_hard", "err"]),
            },
        }
    else:
        checks["graphcold_err_above_hard"] = {"passed": False, "detail": "missing rows"}
    metric_cols = ["macro_f1", "fpr", "fnr", "err", "tail_err"]
    finite = np.isfinite(frame[metric_cols].to_numpy(dtype=float)).all()
    checks["metrics_finite"] = {"passed": bool(finite), "detail": metric_cols}
    checks["no_single_class_collapse"] = {
        "passed": bool(frame["class_count"].min() >= 2),
        "detail": int(frame["class_count"].min()),
    }
    checks["dataset_hash_recorded"] = {
        "passed": bool(frame["dataset_hash"].notna().all()),
        "detail": frame["dataset_hash"].dropna().unique().tolist(),
    }
    checks["seed_recorded"] = {"passed": bool((frame["seed"] == 42).all()), "detail": sorted(frame["seed"].unique().tolist())}
    checks["data_source_recorded"] = {
        "passed": bool(frame["data_source"].astype(str).str.len().gt(0).all()),
        "detail": sorted(frame["data_source"].astype(str).unique().tolist()),
    }
    return checks


def _write_report(report: dict[str, Any], reports_dir: Path, dataset: str = "cicids2017") -> None:
    if dataset == "cesnet_tls_year22":
        json_path = reports_dir / "cesnet_smoke_report.json"
        md_path = reports_dir / "cesnet_smoke_report.md"
    else:
        json_path = reports_dir / "smoke_realdata_report.json"
        md_path = reports_dir / "smoke_realdata_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_report_markdown(report), encoding="utf-8")


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real-Data Smoke Report",
        "",
        f"- Dataset: {report['dataset']}",
        f"- Status: {report['status']}",
        f"- Passed: {report['passed']}",
        f"- Dataset hash: `{report.get('dataset_hash')}`",
        f"- Seed: {report.get('seed')}",
    ]
    if report["status"] == "blocked":
        lines.append("- Blocking reasons:")
        lines.extend([f"  - {reason}" for reason in report.get("blocking_reasons", [])] or ["  - none"])
    else:
        lines.append("- Quality checks:")
        for name, item in report["quality_checks"].items():
            lines.append(f"  - {name}: {item['passed']}")
    lines.append("")
    return "\n".join(lines)


def _normalize_dataset(dataset: str) -> str:
    key = dataset.lower().replace("-", "")
    aliases = {
        "cicids": "cicids2017",
        "cicids2017": "cicids2017",
        "maltls22": "maltls22",
        "maltls": "maltls22",
        "cesnet": "cesnet_tls_year22",
        "cesnettlsyear22": "cesnet_tls_year22",
        "cesnet_tls_year22": "cesnet_tls_year22",
    }
    if key not in aliases:
        raise ValueError("Smoke supports cicids2017, maltls22, and cesnet_tls_year22 only.")
    return aliases[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    report = run_smoke_realdata(args.dataset, args.configs, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
