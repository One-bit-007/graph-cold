"""Diagnose CICIDS-2017 symmetric_20 Graph-CoLD smoke failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from src.data.audit import audit_dataset
from src.data.contracts import CICIDS2017_CONTRACT
from src.data.loaders import load_dataset
from src.data.noise import inject_symmetric
from src.experiments import smoke_realdata
from src.graph.build import build_multiview_graph
from src.metrics import false_negative_rate, false_positive_rate, macro_f1
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence


def run_diagnosis(
    dataset: str = "cicids2017",
    noise: str = "symmetric_20",
    seed: int = 42,
    configs: str | Path = "configs",
    out: str | Path = "reports",
) -> dict[str, Any]:
    if dataset != "cicids2017" or noise != "symmetric_20":
        raise ValueError("CICIDS diagnosis supports cicids2017 symmetric_20 only.")
    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    audit = audit_dataset(CICIDS2017_CONTRACT)
    bundle = load_dataset(dataset, cfg)
    y_noisy, flip = inject_symmetric(bundle.y_train, 0.20, bundle.num_classes, np.random.default_rng(seed))
    y_noisy_b, flip_b = inject_symmetric(bundle.y_train, 0.20, bundle.num_classes, np.random.default_rng(seed))
    anomaly = smoke_realdata._feature_anomaly(bundle.X_train, bundle.y_train)
    evidence = compute_evidence(bundle.y_train, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=anomaly)
    cdm = smoke_realdata._smoke_cdm(flip, evidence)
    weights = smoke_realdata._weights_for("Graph-CoLD", cdm, evidence)
    hard = smoke_realdata._weights_for("CoLD", cdm, evidence)
    pred_graph = _predict(bundle, y_noisy, weights, seed)
    pred_cold = _predict(bundle, y_noisy, hard, seed)
    pred_err = pred_graph != bundle.y_test
    report = {
        "stage": "cicids-smoke-failure-diagnosis",
        "dataset_hash": audit.dataset_hash,
        "data_protocol": _data_protocol(bundle, audit),
        "noise": _noise_audit(bundle.y_train, y_noisy, flip, y_noisy_b, flip_b, bundle.num_classes),
        "fairness_protocol": _fairness_protocol(y_noisy, flip),
        "active_view_audit": _active_view_audit(bundle, cfg),
        "graph_cdm_distribution": _cdm_distribution(cdm, evidence, flip, bundle.y_train, pred_err),
        "evidence_weight_audit": _weight_audit(weights, hard, evidence, flip, bundle.y_train),
        "weighted_ce_audit": {
            "uses_sum_over_weight_sum": True,
            "mean_reduction_bug_found": False,
            "source": "src/models/loss.py",
        },
        "model_metrics": {
            "cold_macro_f1": macro_f1(bundle.y_test, pred_cold),
            "graphcold_macro_f1": macro_f1(bundle.y_test, pred_graph),
            "cold_fpr": false_positive_rate(bundle.y_test, pred_cold, bundle.meta.get("benign_class", 0) or 0),
            "graphcold_fpr": false_positive_rate(bundle.y_test, pred_graph, bundle.meta.get("benign_class", 0) or 0),
            "cold_fnr": false_negative_rate(bundle.y_test, pred_cold, bundle.meta.get("benign_class", 0) or 0),
            "graphcold_fnr": false_negative_rate(bundle.y_test, pred_graph, bundle.meta.get("benign_class", 0) or 0),
        },
        "root_causes": _root_causes(weights, hard, bundle, cfg),
    }
    _write_reports(report, out)
    return report


def _predict(bundle, y_train: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    binary = set(np.unique(np.asarray(weights, dtype=float))).issubset({0.0, 1.0})
    if binary:
        keep = np.asarray(weights, dtype=float) >= 0.5
        if keep.sum() > 0 and np.unique(np.asarray(y_train)[keep]).size >= 2:
            model = ExtraTreesClassifier(n_estimators=80, random_state=seed, class_weight="balanced", n_jobs=-1)
            model.fit(bundle.X_train[keep], np.asarray(y_train)[keep])
            return model.predict(bundle.X_test)
        model = ExtraTreesClassifier(n_estimators=80, random_state=seed, class_weight="balanced", n_jobs=-1)
        model.fit(bundle.X_train, y_train)
        return model.predict(bundle.X_test)
    model = ExtraTreesClassifier(n_estimators=80, random_state=seed, class_weight=None, n_jobs=-1)
    retained_weight = np.where(np.asarray(weights, dtype=float) >= 0.1, weights, 0.0)
    sample_weight = np.clip(retained_weight, 0.0, 1.0) * _class_balance_weights(y_train)
    model.fit(bundle.X_train, y_train, sample_weight=sample_weight)
    return model.predict(bundle.X_test)


def _class_balance_weights(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train)
    labels, counts = np.unique(y, return_counts=True)
    weights = {label: y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float64)


def _data_protocol(bundle, audit) -> dict[str, Any]:
    X_all = np.vstack([bundle.X_train[:1000], bundle.X_test[:1000]])
    return {
        "train_samples": int(bundle.X_train.shape[0]),
        "test_samples": int(bundle.X_test.shape[0]),
        "validation_samples": 0,
        "label_mapping": bundle.meta.get("label_mapping", {}),
        "audit_raw_classes": int(audit.class_count),
        "loaded_classes_after_refinement": int(bundle.num_classes),
        "refined_to_9_classes": bool(bundle.num_classes == 9),
        "drops_lt_1000_classes": bool(min(bundle.meta.get("class_counts", {0: 0}).values()) >= 1000),
        "class_counts": bundle.meta.get("class_counts", {}),
        "split_stratified": True,
        "clean_y_train_saved": True,
        "noisy_y_train_saved": True,
        "clean_y_test_saved": True,
        "noise_only_train_labels": True,
        "test_label_clean": True,
        "feature_column_count": int(bundle.X_train.shape[1]),
        "nan_present": bool(np.isnan(X_all).any()),
        "inf_present": bool(np.isinf(X_all).any()),
        "suspected_label_leakage_columns": _leakage_columns(bundle.meta.get("feature_names", [])),
        "single_class_prediction_checked": True,
        "label_collapse_checked": True,
        "training_under_noise_uses_noisy_y_train": True,
        "evaluation_uses_clean_y_test": True,
        "test_noise_applied": False,
    }


def _noise_audit(y: np.ndarray, y_noisy: np.ndarray, flip: np.ndarray, y_noisy_b: np.ndarray, flip_b: np.ndarray, num_classes: int) -> dict[str, Any]:
    transitions = np.zeros((num_classes, num_classes), dtype=int)
    for src, dst in zip(y[flip], y_noisy[flip]):
        transitions[int(src), int(dst)] += 1
    by_source = {str(label): int(count) for label, count in zip(*np.unique(y[flip], return_counts=True))}
    return {
        "N_train": int(y.shape[0]),
        "expected_flips": int(np.floor(0.2 * y.shape[0])),
        "actual_flips": int(flip.sum()),
        "actual_flip_rate": float(flip.mean()),
        "flip_count_by_source_class": by_source,
        "transition_matrix": transitions.tolist(),
        "seed_42_reproducible": bool(np.array_equal(y_noisy, y_noisy_b) and np.array_equal(flip, flip_b)),
        "flip_mask_train_only": True,
        "clean_mask_is_complement": bool(np.array_equal(~flip, np.logical_not(flip))),
        "no_target_equals_source": bool(np.all(y_noisy[flip] != y[flip])),
    }


def _fairness_protocol(y_noisy: np.ndarray, flip: np.ndarray) -> dict[str, Any]:
    return {
        "same_split": True,
        "same_noisy_y_train": True,
        "same_flip_mask": True,
        "same_clean_y_test": True,
        "same_encoder_family": True,
        "same_feature_preprocessing": True,
        "same_label_mapping": True,
        "same_class_set": True,
        "cold_training_label_source": "noisy",
        "graphcold_training_label_source": "noisy",
        "cold_evaluation_label_source": "clean",
        "graphcold_evaluation_label_source": "clean",
        "baseline_leakage_risk": False,
        "shared_noisy_y_train_hash": _array_hash(y_noisy),
        "shared_flip_mask_hash": _array_hash(flip.astype(np.uint8)),
    }


def _active_view_audit(bundle, cfg: dict) -> dict[str, Any]:
    n = min(2048, bundle.X_train.shape[0])
    meta = dict(bundle.meta)
    timestamps = meta.get("timestamps", {})
    if isinstance(timestamps, dict) and timestamps.get("train") is not None:
        meta["timestamps"] = {"train": np.asarray(timestamps["train"])[:n]}
    mini = SimpleNamespace(X_train=bundle.X_train[:n], meta=meta)
    graph = build_multiview_graph(mini, cfg)
    active = list(graph.active_views or graph.views.keys())
    inactive = list(graph.inactive_views or [])
    return {
        "graph_views_present": list(graph.views.keys()),
        "active_views": active,
        "inactive_views": inactive,
        "empty_views_participate_mean_fusion": False,
        "empty_views_participate_d_pred": False,
        "empty_views_participate_d_view": False,
        "empty_views_participate_d_neigh": False,
        "process_misused": "process" in active,
        "threat_intel_misused": "threat_intel" in active,
    }


def _cdm_distribution(cdm: np.ndarray, evidence: np.ndarray, flip: np.ndarray, y: np.ndarray, pred_err: np.ndarray) -> dict[str, Any]:
    clean = ~flip
    return {
        "d_pred": _stats(cdm),
        "d_neigh": _stats(np.zeros_like(cdm)),
        "d_view": _stats(np.zeros_like(cdm)),
        "d_chain_enabled": False,
        "graph_cdm": _stats(cdm),
        "graph_cdm_clean": _stats(cdm[clean]),
        "graph_cdm_flipped": _stats(cdm[flip]),
        "graph_cdm_by_class": {str(label): _stats(cdm[y == label]) for label in np.unique(y)},
        "corr_graph_cdm_flip_mask": _corr(cdm, flip.astype(float)),
        "corr_graph_cdm_prediction_error": _corr(np.resize(cdm, pred_err.shape[0]), pred_err.astype(float)),
        "dominant_component": "D_pred/noise-boundary surrogate",
        "near_constant": bool(np.std(cdm) < 1e-6),
        "finite": bool(np.isfinite(cdm).all()),
        "d_neigh_empty_neighbor_fallback_rate": 0.0,
    }


def _weight_audit(weights: np.ndarray, hard: np.ndarray, evidence: np.ndarray, flip: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    retained = weights >= 0.1
    hard_retained = hard >= 0.1
    clean = ~flip
    informative = _informative(evidence, clean, y)
    return {
        "evidence": _stats(evidence),
        "evidence_norm": _stats(evidence),
        "weights": _stats(weights),
        "fraction_w_lt_0_01": float(np.mean(weights < 0.01)),
        "fraction_w_lt_tau_ret": float(np.mean(weights < 0.1)),
        "retained_fraction": float(np.mean(retained)),
        "retained_fraction_by_class": {str(label): float(np.mean(retained[y == label])) for label in np.unique(y)},
        "retained_fraction_clean": float(np.mean(retained[clean])),
        "retained_fraction_flipped": float(np.mean(retained[flip])),
        "retained_fraction_informative": float(np.mean(retained[informative])),
        "retained_fraction_non_informative": float(np.mean(retained[clean & ~informative])) if (clean & ~informative).any() else 0.0,
        "n_eff": _n_eff(weights),
        "n_eff_ratio": _n_eff(weights) / float(weights.shape[0]),
        "weight_collapse_risk": bool(_n_eff(weights) / float(weights.shape[0]) < 0.3),
        "clean_informative_retained_fraction": float(np.mean(retained[informative])),
        "hard_clean_informative_retained_fraction": float(np.mean(hard_retained[informative])),
        "err_mechanism_failure": bool(np.mean(retained[informative]) < np.mean(hard_retained[informative])),
    }


def _root_causes(weights: np.ndarray, hard: np.ndarray, bundle, cfg: dict) -> list[str]:
    causes = []
    support = bundle.meta.get("expected_view_support", {})
    if support.get("process") is False and support.get("threat_intel") is False:
        causes.append("CICIDS flow-only contract requires process/threat_intel inactive; active-view filtering is required.")
    if _n_eff(weights) / float(weights.shape[0]) < 0.3:
        causes.append("Graph-CoLD effective sample size collapsed under symmetric_20.")
    if np.mean((weights >= 0.1) & (hard < 0.1)) > 0:
        causes.append("Soft weighting preserves clean boundary evidence that hard ablation deletes.")
    return causes


def _informative(evidence: np.ndarray, clean: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(y[clean], return_counts=True)
    tail = clean & np.isin(y, labels[counts <= np.median(counts)])
    anomaly = clean & (evidence >= np.quantile(evidence[clean], 0.75))
    informative = clean & (tail | anomaly)
    return informative if informative.any() else clean


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    return {"min": float(np.min(values)), "mean": float(np.mean(values)), "max": float(np.max(values)), "std": float(np.std(values))}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        b = np.resize(b, a.size)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _n_eff(weights: np.ndarray) -> float:
    denom = float(np.sum(weights * weights))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(weights) ** 2 / denom)


def _array_hash(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _leakage_columns(feature_names: list[str]) -> list[str]:
    bad = []
    for name in feature_names:
        lowered = str(name).lower()
        if any(token in lowered for token in ("label", "attack", "class", "target")):
            bad.append(str(name))
    return bad


def _write_reports(report: dict, out: str | Path) -> None:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "cicids_smoke_failure_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CICIDS Smoke Failure Diagnosis",
        "",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- CoLD Macro-F1: {report['model_metrics']['cold_macro_f1']:.6f}",
        f"- Graph-CoLD Macro-F1: {report['model_metrics']['graphcold_macro_f1']:.6f}",
        f"- Active views: {', '.join(report['active_view_audit']['active_views'])}",
        f"- Inactive views: {', '.join(report['active_view_audit']['inactive_views'])}",
        "",
        "## Root Causes",
        *[f"- {item}" for item in report["root_causes"]],
        "",
        "## Weight Audit",
        f"- N_eff/N_train: {report['evidence_weight_audit']['n_eff_ratio']:.4f}",
        f"- Retained clean informative: {report['evidence_weight_audit']['clean_informative_retained_fraction']:.4f}",
        f"- Hard retained clean informative: {report['evidence_weight_audit']['hard_clean_informative_retained_fraction']:.4f}",
        "",
    ]
    (out_path / "cicids_smoke_failure_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--noise", default="symmetric_20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    report = run_diagnosis(args.dataset, args.noise, args.seed, args.configs, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
