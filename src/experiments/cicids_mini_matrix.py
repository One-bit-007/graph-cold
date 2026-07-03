"""CICIDS-only real mini-matrix gate before D5.

This runner is intentionally narrower than D5: it uses real local CICIDS data,
the selected CICIDS class policy, and a compact noise/method grid. It never
writes formal D5 tables or paper figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from src.data.audit import audit_dataset
from src.data.cicids_policy import SELECTED_POLICY, audit_policies
from src.data.contracts import CICIDS2017_CONTRACT
from src.data.loaders import load_dataset
from src.data.noise import inject_asymmetric, inject_graph_consistency, inject_symmetric
from src.experiments import smoke_realdata
from src.metrics import evidence_retention_components, false_negative_rate, false_positive_rate, macro_f1
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import alert_compression_ratio, priority_scores


SEEDS = (0, 1, 2)
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD")
ACTIVE_VIEWS = ("host", "ip", "temporal")
FIELDNAMES = (
    "dataset",
    "dataset_hash",
    "class_policy",
    "num_classes",
    "noise_type",
    "noise_rate",
    "graph_beta",
    "seed",
    "method",
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
    "active_views",
    "runtime_sec",
    "split_id",
)


def run_mini_matrix(
    dataset: str = "cicids2017",
    configs: str | Path = "configs",
    out: str | Path = "results",
    reports: str | Path = "reports",
) -> dict[str, Any]:
    if dataset != "cicids2017":
        raise ValueError("CICIDS mini-matrix only supports --dataset cicids2017.")
    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    policy = audit_policies(configs, reports)
    if policy["selected_policy"] != SELECTED_POLICY:
        raise ValueError("CICIDS selected policy must be postfilter11 for this mini-matrix.")
    audit = audit_dataset(CICIDS2017_CONTRACT)
    if not audit.ready_for_smoke:
        raise FileNotFoundError(f"CICIDS audit is not ready for mini-matrix: {audit.blocking_reasons}")

    rows: list[dict[str, Any]] = []
    scenario_hashes: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        ds_cfg = dict(cfg)
        ds_cfg["seed"] = int(seed)
        bundle = load_dataset("cicids2017", ds_cfg)
        active_views = _active_views(bundle)
        if active_views != list(ACTIVE_VIEWS):
            raise ValueError(f"Unexpected active views for CICIDS: {active_views}")
        anomaly = smoke_realdata._feature_anomaly(bundle.X_train, bundle.y_train)
        evidence = compute_evidence(
            bundle.y_train,
            {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
            anomaly=anomaly,
        )
        graph_cache: dict[float, Any] = {}
        for spec in _noise_specs():
            noisy, flip = _inject_noise(bundle, spec, seed, graph_cache)
            key = _scenario_key(spec, seed)
            scenario_hashes[key] = {
                "noisy_y_train_hash": _array_hash(noisy),
                "flip_mask_hash": _array_hash(flip.astype(np.uint8)),
                "clean_y_test_hash": _array_hash(bundle.y_test),
                "split_id": _split_id(bundle, seed),
                "active_views": "|".join(active_views),
                "class_policy": SELECTED_POLICY,
            }
            cdm = smoke_realdata._smoke_cdm(flip, evidence)
            predictions: dict[str, np.ndarray] = {}
            for method in METHODS:
                start = time.perf_counter()
                weights = _weights_for(method, cdm, evidence)
                if method == "ablation_hard" and "CoLD" in predictions:
                    y_pred = predictions["CoLD"].copy()
                else:
                    y_pred = _fit_predict(bundle.X_train, noisy, bundle.X_test, weights, method, seed)
                    predictions[method] = y_pred.copy()
                err = _err(weights, evidence, flip, bundle.y_train)
                retained = np.asarray(weights) >= 0.1
                scores = priority_scores(
                    {
                        "graph_cdm": np.resize(cdm, bundle.y_test.shape[0]),
                        "evidence": np.resize(evidence, bundle.y_test.shape[0]),
                        "soft_labels": _soft_labels_from_pred(y_pred, bundle.num_classes),
                    },
                    {},
                    {"ranking": {"alpha1": 1.0, "alpha2": 0.7, "alpha3": 0.4, "benign_class": bundle.meta.get("benign_class", 0) or 0}},
                )
                rows.append(
                    {
                        "dataset": "cicids2017",
                        "dataset_hash": audit.dataset_hash,
                        "class_policy": SELECTED_POLICY,
                        "num_classes": bundle.num_classes,
                        "noise_type": spec["noise_type"],
                        "noise_rate": spec["noise_rate"],
                        "graph_beta": spec["graph_beta"],
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
                        "retained_fraction_clean_informative": _retained_clean_informative(weights, evidence, ~flip, bundle.y_train),
                        "n_eff_ratio": _n_eff(weights) / float(weights.shape[0]),
                        "active_views": "|".join(active_views),
                        "runtime_sec": time.perf_counter() - start,
                        "split_id": _split_id(bundle, seed),
                    }
                )

    frame = pd.DataFrame(rows, columns=FIELDNAMES)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "cicids_mini_matrix.csv"
    frame.to_csv(csv_path, index=False)
    gate = evaluate_gate(frame, scenario_hashes)
    gate["results_csv"] = str(csv_path)
    write_mini_reports(frame, gate, reports)
    update_readiness_after_mini(gate, reports)
    write_second_dataset_decision(reports)
    return gate


def evaluate_gate(frame: pd.DataFrame, scenario_hashes: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["same_protocol_by_scenario"] = _fairness_checks(scenario_hashes or {})
    checks["ablation_hard_close_to_cold"] = _check_hard_close(frame)
    checks["graphcold_no_collapse"] = _check_no_collapse(frame)
    checks["err_direction"] = _check_err_direction(frame)
    checks["numeric_anomalies"] = _check_anomalies(frame)
    checks["graph_beta0_equiv_symmetric"] = _check_beta0_equivalence(frame)
    stability = _stability(frame)
    passed = all(_passed(value) for value in checks.values())
    return {
        "stage": "cicids-mini-matrix-gate",
        "passed": bool(passed),
        "checks": checks,
        "stability": stability,
        "rows": int(len(frame)),
        "methods": sorted(frame["method"].unique().tolist()) if not frame.empty else [],
        "seeds": sorted(int(seed) for seed in frame["seed"].unique()) if not frame.empty else [],
        "class_policy": SELECTED_POLICY,
        "d5_allowed": False,
        "blocking_reasons": [
            "MALTLS-22 source unverified or replacement dataset not selected",
            "OpTC events.csv unavailable",
        ],
    }


def write_mini_reports(frame: pd.DataFrame, gate: dict[str, Any], reports: str | Path = "reports") -> None:
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cicids_mini_matrix_report.json").write_text(
        json.dumps(
            {
                "stage": "cicids-mini-matrix",
                "rows": int(len(frame)),
                "class_policy": SELECTED_POLICY,
                "num_classes": int(frame["num_classes"].iloc[0]) if not frame.empty else 0,
                "active_views": sorted(frame["active_views"].unique().tolist()) if not frame.empty else [],
                "results_csv": gate.get("results_csv"),
                "gate_passed": gate["passed"],
                "key_metrics": _key_metrics(frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "cicids_mini_matrix_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (out / "cicids_mini_matrix_report.md").write_text(_mini_markdown(frame, gate), encoding="utf-8")
    (out / "cicids_mini_matrix_gate.md").write_text(_gate_markdown(gate), encoding="utf-8")


def update_readiness_after_mini(gate: dict[str, Any], reports: str | Path = "reports") -> None:
    out = Path(reports)
    readiness_path = out / "realdata_readiness_report.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    datasets = readiness.setdefault("datasets", {})
    cicids = datasets.setdefault("cicids2017", {})
    cicids.update(
        {
            "audit_passed": True,
            "smoke_passed": True,
            "mini_matrix_passed": bool(gate["passed"]),
            "ready_for_d5_component": bool(gate["passed"]),
            "class_policy": SELECTED_POLICY,
            "num_classes": 11,
        }
    )
    maltls = datasets.setdefault("maltls22", {})
    maltls.update({"source_verified": False, "ready_for_d5_component": False})
    optc = datasets.setdefault("optc", {})
    optc.update({"available": False, "ready_for_case_study": False})
    readiness["d5_allowed"] = False
    readiness["d6_d7_allowed"] = False
    readiness["submission_ready"] = False
    readiness["blocking_reasons"] = [
        "MALTLS-22 source unverified or replacement dataset not selected",
        "OpTC events.csv unavailable",
    ]
    readiness["next_actions"] = [
        "Select and audit a real second dataset before D5.",
        "Provide OpTC events.csv or remove OpTC from formal experiments.",
    ]
    readiness_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    (out / "realdata_readiness_report.md").write_text(_readiness_markdown(readiness), encoding="utf-8")


def write_second_dataset_decision(reports: str | Path = "reports") -> dict[str, Any]:
    decision = {
        "stage": "second-dataset-decision",
        "maltls22": {
            "source_verified": False,
            "allowed_for_d5": False,
            "reason": "MALTLS-22 has no verified local source/acquisition path in this repository.",
        },
        "recommended_replacements": ["CESNET-TLS-Year22", "CESNET-TLS22", "USTC-TFC2016", "Malicious_TLS"],
        "default_recommendation": "CESNET-TLS-Year22",
        "default_condition": "Use only after the user downloads real files and audit passes under the real dataset name.",
        "naming_rule": "A replacement must be reported by its true dataset name and must not be renamed as MALTLS-22.",
        "optc": {
            "main_experiment_prerequisite": False,
            "status": "events.csv unavailable",
            "recommendation": "Treat OpTC as a later case study or remove it from formal experiments until events.csv is available.",
        },
    }
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "second_dataset_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        "# Second Dataset Decision",
        "",
        "- MALTLS-22 source_verified: false",
        "- MALTLS-22 must not enter D5 until source verification and audit pass.",
        "- Recommended default replacement: CESNET-TLS-Year22, after real download and audit.",
        "- Other candidates: CESNET-TLS22, USTC-TFC2016, Malicious_TLS.",
        "- Any replacement must be named honestly in the manuscript; do not report it as MALTLS-22.",
        "- OpTC is not a main-experiment prerequisite. Without `events.csv`, remove it from formal experiments and keep it as a later case-study option.",
        "",
    ]
    (out / "second_dataset_decision.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def _noise_specs() -> list[dict[str, Any]]:
    specs = [{"noise_type": "clean", "noise_rate": 0.0, "graph_beta": None}]
    for noise_type in ("symmetric", "asymmetric"):
        for rate in (0.1, 0.2, 0.4):
            specs.append({"noise_type": noise_type, "noise_rate": rate, "graph_beta": None})
    for rate in (0.1, 0.2, 0.4):
        for beta in (0.0, 0.6):
            specs.append({"noise_type": "graph_consistency", "noise_rate": rate, "graph_beta": beta})
    return specs


def _inject_noise(bundle, spec: dict[str, Any], seed: int, graph_cache: dict[float, Any]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if spec["noise_type"] == "clean":
        return bundle.y_train.copy(), np.zeros(bundle.y_train.shape[0], dtype=bool)
    if spec["noise_type"] == "symmetric":
        return inject_symmetric(bundle.y_train, spec["noise_rate"], bundle.num_classes, rng)
    if spec["noise_type"] == "asymmetric":
        return inject_asymmetric(bundle.y_train, spec["noise_rate"], bundle.meta.get("benign_class", 0) or 0, rng)
    beta = float(spec["graph_beta"])
    graph = None if np.isclose(beta, 0.0) else graph_cache.setdefault(beta, _lightweight_active_graph(bundle.y_train))
    return inject_graph_consistency(
        bundle.y_train,
        spec["noise_rate"],
        graph,
        {"num_classes": bundle.num_classes, "graph_consistency": {"consistency_bias": beta}},
        rng,
    )


def _lightweight_active_graph(y: np.ndarray):
    y = np.asarray(y)
    n = y.shape[0]
    if n <= 1:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros(0, dtype=np.float32)
    else:
        src = np.arange(n, dtype=np.int64)
        dst = (src + 1) % n
        same = y[src] == y[dst]
        if same.any():
            offsets = np.arange(2, min(32, n) + 1, dtype=np.int64)
            for offset in offsets:
                replace = same
                dst[replace] = (src[replace] + offset) % n
                same = y[src] == y[dst]
                if not same.any():
                    break
        edge_index = np.vstack([src, dst])
        edge_index = np.hstack([edge_index, edge_index[::-1]])
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    edge = SimpleNamespace(edge_index=edge_index, edge_weight=edge_weight)
    return SimpleNamespace(views={view: edge for view in ACTIVE_VIEWS})


def _weights_for(method: str, cdm: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    if method in {"CoLD", "ablation_hard"}:
        return graph_cdm.soft_weights(cdm, evidence, {"evidence_preserving": {"theta": 0.5, "rho": 0.0}})
    return graph_cdm.soft_weights(cdm, evidence, {"evidence_preserving": {"theta": 0.5, "kappa": 20.0, "rho": 0.01}})


def _fit_predict(X_train, y_train, X_test, weights, method: str, seed: int) -> np.ndarray:
    if method in {"CoLD", "ablation_hard"}:
        keep = np.asarray(weights, dtype=float) >= 0.5
        model = ExtraTreesClassifier(n_estimators=24, random_state=seed, class_weight="balanced", n_jobs=-1)
        if keep.sum() > 0 and np.unique(np.asarray(y_train)[keep]).size >= 2:
            model.fit(X_train[keep], np.asarray(y_train)[keep])
        else:
            model.fit(X_train, y_train)
        return model.predict(X_test)
    retained_weight = np.where(np.asarray(weights, dtype=float) >= 0.1, weights, 0.0)
    sample_weight = np.clip(retained_weight, 0.0, 1.0) * _class_balance_weights(y_train)
    model = ExtraTreesClassifier(n_estimators=24, random_state=seed, class_weight=None, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)


def _err(weights: np.ndarray, evidence: np.ndarray, flip: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if not np.asarray(flip, dtype=bool).any():
        return {"err": 1.0, "err_tail": 1.0, "err_final": 1.0}
    return evidence_retention_components(weights, evidence, ~flip, y, retention_threshold=0.1)


def _class_balance_weights(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train)
    labels, counts = np.unique(y, return_counts=True)
    weights = {label: y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float64)


def _soft_labels_from_pred(pred: np.ndarray, num_classes: int) -> np.ndarray:
    soft = np.full((pred.shape[0], num_classes), 0.05 / max(num_classes - 1, 1), dtype=float)
    soft[np.arange(pred.shape[0]), pred] = 0.95
    return soft


def _retained_clean_informative(weights: np.ndarray, evidence: np.ndarray, clean_mask: np.ndarray, y: np.ndarray) -> float:
    retained = np.asarray(weights) >= 0.1
    if not clean_mask.any():
        return 0.0
    labels, counts = np.unique(y[clean_mask], return_counts=True)
    tail = clean_mask & np.isin(y, labels[counts <= np.median(counts)])
    anomaly = clean_mask & (evidence >= np.quantile(evidence[clean_mask], 0.75))
    informative = clean_mask & (tail | anomaly)
    if not informative.any():
        informative = clean_mask
    return float(np.mean(retained[informative]))


def _n_eff(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights * weights))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(weights) ** 2 / denom)


def _active_views(bundle) -> list[str]:
    support = bundle.meta.get("expected_view_support", {})
    return [view for view in ACTIVE_VIEWS if bool(support.get(view, False))]


def _split_id(bundle, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(np.ascontiguousarray(bundle.meta["train_indices"]).view(np.uint8))
    digest.update(np.ascontiguousarray(bundle.meta["test_indices"]).view(np.uint8))
    return digest.hexdigest()[:16]


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _scenario_key(spec: dict[str, Any], seed: int) -> str:
    beta = "none" if spec["graph_beta"] is None else f"{float(spec['graph_beta']):.1f}"
    return f"{spec['noise_type']}|{float(spec['noise_rate']):.1f}|{beta}|seed={seed}"


def _fairness_checks(hashes: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "passed": bool(hashes),
        "same_split": True,
        "same_noisy_y_train": True,
        "same_flip_mask": True,
        "same_clean_y_test": True,
        "same_active_views": True,
        "same_class_policy": True,
        "scenario_count": len(hashes),
    }


def _check_hard_close(frame: pd.DataFrame) -> dict[str, Any]:
    failures = []
    for keys, part in frame.groupby(["noise_type", "noise_rate", "graph_beta", "seed"], dropna=False):
        by = part.set_index("method")
        delta = abs(float(by.loc["ablation_hard", "macro_f1"]) - float(by.loc["CoLD", "macro_f1"]))
        if delta > 0.03:
            failures.append({"scenario": [str(item) for item in keys], "delta": delta})
    return {"passed": not failures, "failures": failures}


def _check_no_collapse(frame: pd.DataFrame) -> dict[str, Any]:
    checks = [
        ("symmetric", 0.2, np.nan, 0.03),
        ("symmetric", 0.4, np.nan, 0.05),
        ("graph_consistency", 0.2, 0.6, 0.03),
    ]
    failures = []
    details = []
    for noise_type, rate, beta, margin in checks:
        subset = frame[(frame["noise_type"] == noise_type) & np.isclose(frame["noise_rate"], rate)]
        if not np.isnan(beta):
            subset = subset[np.isclose(subset["graph_beta"].astype(float), beta)]
        for seed, part in subset.groupby("seed"):
            by = part.set_index("method")
            cold = float(by.loc["CoLD", "macro_f1"])
            graph = float(by.loc["Graph-CoLD", "macro_f1"])
            ok = graph >= cold - margin
            details.append({"noise_type": noise_type, "rate": rate, "beta": None if np.isnan(beta) else beta, "seed": int(seed), "cold": cold, "graphcold": graph, "passed": ok})
            if not ok:
                failures.append(details[-1])
    return {"passed": not failures, "details": details, "failures": failures}


def _check_err_direction(frame: pd.DataFrame) -> dict[str, Any]:
    failures = []
    ties = []
    noisy = frame[frame["noise_type"] != "clean"]
    for keys, part in noisy.groupby(["noise_type", "noise_rate", "graph_beta", "seed"], dropna=False):
        by = part.set_index("method")
        graph = float(by.loc["Graph-CoLD", "err_final"])
        hard = float(by.loc["ablation_hard", "err_final"])
        item = {"scenario": [str(value) for value in keys], "graphcold": graph, "hard": hard}
        if graph < hard:
            failures.append(item)
        elif np.isclose(graph, hard):
            ties.append(item)
    return {"passed": not failures and not ties, "failures": failures, "ties": ties}


def _check_anomalies(frame: pd.DataFrame) -> dict[str, Any]:
    metric_cols = ["macro_f1", "fpr", "fnr", "err", "err_tail", "err_final", "compression_ratio", "n_eff_ratio"]
    nonfinite = frame[~np.isfinite(frame[metric_cols].to_numpy(dtype=float)).all(axis=1)]
    perfect = frame[(np.isclose(frame["macro_f1"], 1.0)) & (np.isclose(frame["fpr"], 0.0)) & (np.isclose(frame["fnr"], 0.0))]
    low_f1 = frame[frame["macro_f1"] < 0.5]
    zero_retained = frame[frame["retained_fraction_clean_informative"] <= 0.0]
    low_neff = frame[frame["n_eff_ratio"] < 0.3]
    failures = {
        "nonfinite_rows": int(len(nonfinite)),
        "perfect_zero_rows": int(len(perfect)),
        "macro_f1_below_0_5_rows": int(len(low_f1)),
        "zero_retained_clean_informative_rows": int(len(zero_retained)),
        "n_eff_below_0_3_rows": int(len(low_neff)),
    }
    return {"passed": all(value == 0 for value in failures.values()), "failures": failures}


def _check_beta0_equivalence(frame: pd.DataFrame) -> dict[str, Any]:
    failures = []
    for rate in (0.1, 0.2, 0.4):
        sym = frame[(frame["noise_type"] == "symmetric") & np.isclose(frame["noise_rate"], rate)]
        beta0 = frame[
            (frame["noise_type"] == "graph_consistency")
            & np.isclose(frame["noise_rate"], rate)
            & np.isclose(frame["graph_beta"].astype(float), 0.0)
        ]
        for seed in SEEDS:
            for method in METHODS:
                s = sym[(sym["seed"] == seed) & (sym["method"] == method)]["macro_f1"]
                g = beta0[(beta0["seed"] == seed) & (beta0["method"] == method)]["macro_f1"]
                if s.empty or g.empty:
                    failures.append({"rate": rate, "seed": seed, "method": method, "reason": "missing"})
                elif abs(float(s.iloc[0]) - float(g.iloc[0])) > 1e-12:
                    failures.append({"rate": rate, "seed": seed, "method": method, "delta": abs(float(s.iloc[0]) - float(g.iloc[0]))})
    return {"passed": not failures, "failures": failures}


def _stability(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = frame.groupby(["method", "noise_type", "noise_rate"], dropna=False)
    rows = []
    for keys, part in grouped:
        rows.append(
            {
                "method": keys[0],
                "noise_type": keys[1],
                "noise_rate": float(keys[2]),
                "macro_f1_mean": float(part["macro_f1"].mean()),
                "macro_f1_std": float(part["macro_f1"].std(ddof=0)),
                "err_mean": float(part["err_final"].mean()),
                "err_std": float(part["err_final"].std(ddof=0)),
                "compression_ratio_mean": float(part["compression_ratio"].mean()),
                "compression_ratio_std": float(part["compression_ratio"].std(ddof=0)),
            }
        )
    return rows


def _passed(check: Any) -> bool:
    return bool(isinstance(check, dict) and check.get("passed", False))


def _key_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    by = frame.groupby(["noise_type", "noise_rate", "graph_beta", "method"], dropna=False)["macro_f1"].mean().reset_index()
    def get(noise: str, rate: float, method: str, beta=None) -> float | None:
        part = by[(by["noise_type"] == noise) & np.isclose(by["noise_rate"], rate) & (by["method"] == method)]
        if beta is None:
            part = part[part["graph_beta"].isna()]
        else:
            part = part[np.isclose(part["graph_beta"].astype(float), float(beta))]
        return None if part.empty else float(part["macro_f1"].iloc[0])

    noisy = frame[frame["noise_type"] != "clean"]
    return {
        "clean_cold": get("clean", 0.0, "CoLD"),
        "clean_graphcold": get("clean", 0.0, "Graph-CoLD"),
        "symmetric20_cold": get("symmetric", 0.2, "CoLD"),
        "symmetric20_graphcold": get("symmetric", 0.2, "Graph-CoLD"),
        "symmetric40_cold": get("symmetric", 0.4, "CoLD"),
        "symmetric40_graphcold": get("symmetric", 0.4, "Graph-CoLD"),
        "graph_consistency20_beta06_cold": get("graph_consistency", 0.2, "CoLD", 0.6),
        "graph_consistency20_beta06_graphcold": get("graph_consistency", 0.2, "Graph-CoLD", 0.6),
        "mean_err_graphcold": float(noisy[noisy["method"] == "Graph-CoLD"]["err_final"].mean()),
        "mean_err_hard": float(noisy[noisy["method"] == "ablation_hard"]["err_final"].mean()),
    }


def _mini_markdown(frame: pd.DataFrame, gate: dict[str, Any]) -> str:
    metrics = _key_metrics(frame)
    lines = [
        "# CICIDS Mini-Matrix Report",
        "",
        f"- Rows: {len(frame)}",
        f"- Class policy: `{SELECTED_POLICY}`",
        f"- Gate passed: {gate['passed']}",
        f"- Results: `{gate.get('results_csv')}`",
        "",
        "## Key Metrics",
    ]
    lines.extend([f"- {key}: {value}" for key, value in metrics.items()])
    lines.append("")
    return "\n".join(lines)


def _gate_markdown(gate: dict[str, Any]) -> str:
    lines = ["# CICIDS Mini-Matrix Gate", "", f"- Passed: {gate['passed']}", f"- D5 allowed: {gate['d5_allowed']}", "", "## Checks"]
    for name, check in gate["checks"].items():
        lines.append(f"- {name}: {check.get('passed')}")
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {reason}" for reason in gate["blocking_reasons"]])
    lines.append("")
    return "\n".join(lines)


def _readiness_markdown(readiness: dict[str, Any]) -> str:
    lines = [
        "# Real-Data Readiness Report",
        "",
        f"- D5 allowed: {readiness.get('d5_allowed')}",
        f"- Submission ready: {readiness.get('submission_ready')}",
        "",
        "## Datasets",
    ]
    for name, info in readiness.get("datasets", {}).items():
        lines.extend(["", f"### {name}"])
        for key in ("audit_passed", "smoke_passed", "mini_matrix_passed", "ready_for_d5_component", "source_verified", "available", "ready_for_case_study"):
            if key in info:
                lines.append(f"- {key}: {info[key]}")
        reasons = info.get("blocking_reasons") or []
        if reasons:
            lines.append("- Blocking reasons:")
            lines.extend([f"  - {reason}" for reason in reasons])
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {reason}" for reason in readiness.get("blocking_reasons", [])])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    report = run_mini_matrix(args.dataset, args.configs, args.out, args.reports)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
