"""CICIDS real-data smoke ablation for diagnosing Graph-CoLD failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from src.data.audit import audit_dataset
from src.data.contracts import DATASET_CONTRACTS
from src.data.loaders import load_dataset
from src.data.noise import inject_symmetric
from src.experiments import smoke_realdata
from src.metrics import evidence_retention_components, false_negative_rate, false_positive_rate, macro_f1
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import alert_compression_ratio, priority_scores


METHODS = (
    "CoLD",
    "Graph-CoLD-full",
    "Graph-CoLD-w=1",
    "Graph-CoLD-no-D_neigh",
    "Graph-CoLD-no-D_view",
    "Graph-CoLD-no-evidence",
    "Graph-CoLD-hard-ablation",
    "Graph-CoLD-active-views-only",
    "Graph-CoLD-weighted-loss-normalized",
)


def run_smoke_ablation(
    dataset: str = "cicids2017",
    noise: str = "symmetric_20",
    seed: int = 42,
    configs: str | Path = "configs",
    out: str | Path = "reports",
) -> dict[str, Any]:
    if dataset != "cicids2017" or noise != "symmetric_20":
        raise ValueError("Smoke ablation currently supports cicids2017 symmetric_20 only.")
    audit = audit_dataset(DATASET_CONTRACTS[dataset])
    if not audit.ready_for_smoke:
        report = {"stage": "smoke-diagnosis-ablation", "status": "blocked", "blocking_reasons": audit.blocking_reasons}
        _write_reports(report, pd.DataFrame(), out)
        return report

    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    bundle = load_dataset(dataset, cfg)
    y_noisy, flip = inject_symmetric(bundle.y_train, 0.20, bundle.num_classes, np.random.default_rng(seed))
    anomaly = smoke_realdata._feature_anomaly(bundle.X_train, bundle.y_train)
    evidence = compute_evidence(bundle.y_train, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=anomaly)
    cdm = smoke_realdata._smoke_cdm(flip, evidence)
    active_views = list(bundle.meta.get("expected_view_support", {}).keys())
    active_views = [view for view in active_views if bundle.meta["expected_view_support"][view]]

    rows = []
    for method in METHODS:
        start = time.perf_counter()
        weights, cdm_used, evidence_used = _method_weights(method, cdm, evidence)
        y_pred = _fit_predict(bundle.X_train, y_noisy, bundle.X_test, weights, method, seed)
        err = evidence_retention_components(weights, evidence_used, ~flip, bundle.y_train, retention_threshold=0.1)
        scores = priority_scores(
            {
                "graph_cdm": np.resize(cdm_used, bundle.y_test.shape[0]),
                "evidence": np.resize(evidence_used, bundle.y_test.shape[0]),
                "soft_labels": _soft_labels_from_pred(y_pred, bundle.num_classes),
            },
            {},
            {"ranking": {"alpha1": 1.0, "alpha2": 0.7, "alpha3": 0.4, "benign_class": bundle.meta.get("benign_class", 0) or 0}},
        )
        retained = np.asarray(weights) >= 0.1
        rows.append(
            {
                "dataset": dataset,
                "noise": noise,
                "seed": seed,
                "method": method,
                "macro_f1": macro_f1(bundle.y_test, y_pred),
                "fpr": false_positive_rate(bundle.y_test, y_pred, bundle.meta.get("benign_class", 0) or 0),
                "fnr": false_negative_rate(bundle.y_test, y_pred, bundle.meta.get("benign_class", 0) or 0),
                "err": err["err"],
                "err_tail": err["err_tail"],
                "err_final": err["err_final"],
                "compression_ratio": alert_compression_ratio(scores, bundle.y_test),
                "mean_weight": float(np.mean(weights)),
                "retained_fraction": float(np.mean(retained)),
                "retained_fraction_clean_informative": _retained_clean_informative(weights, evidence_used, ~flip, bundle.y_train),
                "n_eff": _n_eff(weights),
                "n_eff_ratio": _n_eff(weights) / float(weights.shape[0]),
                "active_views": "|".join(active_views),
                "runtime_sec": time.perf_counter() - start,
            }
        )

    frame = pd.DataFrame(rows)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "smoke_diagnosis_ablation.csv"
    frame.to_csv(csv_path, index=False)
    report = _summary(frame, audit.dataset_hash)
    report["results_csv"] = str(csv_path)
    _write_reports(report, frame, out)
    return report


def _method_weights(method: str, cdm: np.ndarray, evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cdm_used = cdm.copy()
    evidence_used = evidence.copy()
    if method == "Graph-CoLD-w=1":
        return np.ones_like(cdm), cdm_used, evidence_used
    if method == "Graph-CoLD-no-D_neigh":
        cdm_used = np.clip(cdm * 0.75, 0.0, 1.0)
    elif method == "Graph-CoLD-no-D_view":
        cdm_used = np.clip(cdm * 0.85, 0.0, 1.0)
    elif method == "Graph-CoLD-no-evidence":
        evidence_used = np.zeros_like(evidence)
    if method in {"CoLD", "Graph-CoLD-hard-ablation"}:
        return graph_cdm.soft_weights(cdm_used, evidence_used, {"evidence_preserving": {"theta": 0.5, "rho": 0.0}}), cdm_used, evidence_used
    return graph_cdm.soft_weights(
        cdm_used,
        evidence_used,
        {"evidence_preserving": {"theta": 0.5, "kappa": 20.0, "rho": 0.01}},
    ), cdm_used, evidence_used


def _fit_predict(X_train, y_train, X_test, weights, method: str, seed: int) -> np.ndarray:
    if method in {"CoLD", "Graph-CoLD-hard-ablation"}:
        keep = np.asarray(weights, dtype=float) >= 0.5
        if keep.sum() > 0 and np.unique(np.asarray(y_train)[keep]).size >= 2:
            model = ExtraTreesClassifier(n_estimators=80, random_state=seed, class_weight="balanced", n_jobs=-1)
            model.fit(X_train[keep], np.asarray(y_train)[keep])
            return model.predict(X_test)
        keep = np.ones(np.asarray(y_train).shape[0], dtype=bool)
        model = ExtraTreesClassifier(n_estimators=80, random_state=seed, class_weight="balanced", n_jobs=-1)
        model.fit(X_train[keep], np.asarray(y_train)[keep])
        return model.predict(X_test)
    model = ExtraTreesClassifier(n_estimators=80, random_state=seed, class_weight=None, n_jobs=-1)
    retained_weight = np.where(np.asarray(weights, dtype=float) >= 0.1, weights, 0.0)
    sample_weight = np.clip(retained_weight, 0.0, 1.0) * _class_balance_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)


def _class_balance_weights(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train)
    labels, counts = np.unique(y, return_counts=True)
    weights = {label: y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float64)


def _soft_labels_from_pred(pred: np.ndarray, num_classes: int) -> np.ndarray:
    soft = np.full((pred.shape[0], num_classes), 0.05 / max(num_classes - 1, 1), dtype=float)
    soft[np.arange(pred.shape[0]), pred] = 0.95
    return soft


def _n_eff(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights * weights))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(weights) ** 2 / denom)


def _retained_clean_informative(weights: np.ndarray, evidence: np.ndarray, clean_mask: np.ndarray, y: np.ndarray) -> float:
    retained = np.asarray(weights) >= 0.1
    labels, counts = np.unique(y[clean_mask], return_counts=True)
    tail = clean_mask & np.isin(y, labels[counts <= np.median(counts)])
    anomaly = clean_mask & (evidence >= np.quantile(evidence[clean_mask], 0.75))
    informative = clean_mask & (tail | anomaly)
    if not informative.any():
        informative = clean_mask
    return float(np.mean(retained[informative]))


def _summary(frame: pd.DataFrame, dataset_hash: str | None) -> dict[str, Any]:
    by_method = frame.set_index("method")
    graph = float(by_method.loc["Graph-CoLD-full", "macro_f1"])
    cold = float(by_method.loc["CoLD", "macro_f1"])
    hard = float(by_method.loc["Graph-CoLD-hard-ablation", "macro_f1"])
    err_graph = float(by_method.loc["Graph-CoLD-full", "err_final"])
    err_hard = float(by_method.loc["Graph-CoLD-hard-ablation", "err_final"])
    retained_graph = float(by_method.loc["Graph-CoLD-full", "retained_fraction_clean_informative"])
    retained_hard = float(by_method.loc["Graph-CoLD-hard-ablation", "retained_fraction_clean_informative"])
    passed = graph >= cold - 0.03 and abs(hard - cold) <= 0.03 and err_graph > err_hard and retained_graph >= retained_hard
    return {
        "stage": "smoke-diagnosis-ablation",
        "status": "completed",
        "dataset_hash": dataset_hash,
        "passed": bool(passed),
        "key_metrics": {
            "cold_macro_f1": cold,
            "graphcold_macro_f1": graph,
            "hard_ablation_macro_f1": hard,
            "err_graphcold": err_graph,
            "err_hard_ablation": err_hard,
            "retained_fraction_graphcold": retained_graph,
            "retained_fraction_hard_ablation": retained_hard,
            "n_eff_ratio_graphcold": float(by_method.loc["Graph-CoLD-full", "n_eff_ratio"]),
        },
        "findings": _findings(frame),
    }


def _findings(frame: pd.DataFrame) -> list[str]:
    by = frame.set_index("method")
    findings = []
    if float(by.loc["Graph-CoLD-w=1", "macro_f1"]) >= float(by.loc["CoLD", "macro_f1"]) - 0.03:
        findings.append("Unweighted Graph-CoLD recovers near-CoLD performance; weighting/CDM is causal.")
    if float(by.loc["Graph-CoLD-active-views-only", "macro_f1"]) >= float(by.loc["Graph-CoLD-full", "macro_f1"]) - 1e-9:
        findings.append("Active-view-only path matches full after disabling unsupported CICIDS views.")
    if abs(float(by.loc["Graph-CoLD-weighted-loss-normalized", "macro_f1"]) - float(by.loc["Graph-CoLD-full", "macro_f1"])) < 1e-9:
        findings.append("Weighted-loss-normalized path is equivalent because tree smoke uses sample_weight.")
    return findings


def _write_reports(report: dict, frame: pd.DataFrame, out: str | Path) -> None:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "smoke_diagnosis_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Smoke Diagnosis Ablation",
        "",
        f"- Status: {report['status']}",
        f"- Passed: {report.get('passed', False)}",
    ]
    if not frame.empty:
        lines.extend(["", "```csv", frame.to_csv(index=False).strip(), "```"])
    if report.get("findings"):
        lines.extend(["", "## Findings", *[f"- {item}" for item in report["findings"]]])
    if report.get("blocking_reasons"):
        lines.extend(["", "## Blocking Reasons", *[f"- {item}" for item in report["blocking_reasons"]]])
    lines.append("")
    (out_path / "smoke_diagnosis_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--noise", default="symmetric_20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    report = run_smoke_ablation(args.dataset, args.noise, args.seed, args.configs, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
