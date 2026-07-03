"""CESNET-TLS-Year22 mini-matrix gate.

Runs only after CESNET smoke passes. Missing data or failed smoke writes blocked
reports and does not create `results/cesnet_mini_matrix.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.audit import audit_dataset
from src.data.cesnet_policy import SELECTED_POLICY, audit_policies, write_view_policy_report
from src.data.contracts import CESNET_TLS_YEAR22_CONTRACT
from src.data.loaders import load_dataset
from src.data.noise import inject_asymmetric, inject_graph_consistency, inject_symmetric
from src.experiments import cicids_mini_matrix, smoke_realdata
from src.metrics import false_negative_rate, false_positive_rate, macro_f1
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import alert_compression_ratio, priority_scores


SEEDS = (0, 1, 2)
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD")
ACTIVE_VIEWS = ("ip", "temporal")


def run_mini_matrix(
    dataset: str = "cesnet_tls_year22",
    configs: str | Path = "configs",
    out: str | Path = "results",
    reports: str | Path = "reports",
) -> dict[str, Any]:
    if dataset != "cesnet_tls_year22":
        raise ValueError("CESNET mini-matrix only supports --dataset cesnet_tls_year22.")
    reports_path = Path(reports)
    smoke_report = _load_smoke_report(reports_path)
    audit = audit_dataset(CESNET_TLS_YEAR22_CONTRACT)
    audit_policies(configs, reports)
    write_view_policy_report(audit, yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8")).get("cesnet_tls_year22", {}), reports)
    if not audit.ready_for_smoke or not smoke_report.get("passed", False):
        gate = _blocked_gate(audit, smoke_report)
        write_mini_reports(pd.DataFrame(), gate, reports)
        update_two_dataset_readiness(gate, reports)
        write_d5_scope_decision(gate, reports)
        return gate

    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    scenario_hashes: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        ds_cfg = dict(cfg)
        ds_cfg["seed"] = int(seed)
        bundle = load_dataset("cesnet_tls_year22", ds_cfg)
        if bundle.meta.get("active_views") != list(ACTIVE_VIEWS):
            raise ValueError(f"Unexpected CESNET active views: {bundle.meta.get('active_views')}")
        anomaly = smoke_realdata._feature_anomaly(bundle.X_train, bundle.y_train)
        evidence = compute_evidence(bundle.y_train, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=anomaly)
        graph_cache: dict[float, Any] = {}
        for spec in cicids_mini_matrix._noise_specs():
            noisy, flip = _inject_noise(bundle, spec, seed, graph_cache)
            key = cicids_mini_matrix._scenario_key(spec, seed)
            scenario_hashes[key] = {
                "noisy_y_train_hash": cicids_mini_matrix._array_hash(noisy),
                "flip_mask_hash": cicids_mini_matrix._array_hash(flip.astype(np.uint8)),
                "clean_y_test_hash": cicids_mini_matrix._array_hash(bundle.y_test),
                "split_id": cicids_mini_matrix._split_id(bundle, seed),
                "active_views": "|".join(bundle.meta["active_views"]),
                "class_policy": SELECTED_POLICY,
            }
            cdm = smoke_realdata._smoke_cdm(flip, evidence)
            predictions: dict[str, np.ndarray] = {}
            for method in METHODS:
                start = time.perf_counter()
                weights = cicids_mini_matrix._weights_for(method, cdm, evidence)
                if method == "ablation_hard" and "CoLD" in predictions:
                    y_pred = predictions["CoLD"].copy()
                else:
                    y_pred = cicids_mini_matrix._fit_predict(bundle.X_train, noisy, bundle.X_test, weights, method, seed)
                    predictions[method] = y_pred.copy()
                err = cicids_mini_matrix._err(weights, evidence, flip, bundle.y_train)
                retained = np.asarray(weights) >= 0.1
                scores = priority_scores(
                    {
                        "graph_cdm": np.resize(cdm, bundle.y_test.shape[0]),
                        "evidence": np.resize(evidence, bundle.y_test.shape[0]),
                        "soft_labels": cicids_mini_matrix._soft_labels_from_pred(y_pred, bundle.num_classes),
                    },
                    {},
                    {"ranking": {"alpha1": 1.0, "alpha2": 0.7, "alpha3": 0.4, "benign_class": bundle.meta.get("benign_class", 0) or 0}},
                )
                rows.append(
                    {
                        "dataset": "cesnet_tls_year22",
                        "dataset_hash": audit.dataset_hash,
                        "class_policy": SELECTED_POLICY,
                        "num_classes": bundle.num_classes,
                        "removed_classes": json.dumps(bundle.meta.get("removed_classes", {}), sort_keys=True),
                        "min_class_count": int(cfg["cesnet_tls_year22"].get("min_class_count", 1000)),
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
                        "retained_fraction_clean_informative": cicids_mini_matrix._retained_clean_informative(weights, evidence, ~flip, bundle.y_train),
                        "n_eff_ratio": cicids_mini_matrix._n_eff(weights) / float(weights.shape[0]),
                        "active_views": "|".join(bundle.meta["active_views"]),
                        "runtime_sec": time.perf_counter() - start,
                        "split_id": cicids_mini_matrix._split_id(bundle, seed),
                    }
                )
    frame = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "cesnet_mini_matrix.csv"
    frame.to_csv(csv_path, index=False)
    gate = evaluate_gate(frame, scenario_hashes)
    gate["results_csv"] = str(csv_path)
    write_mini_reports(frame, gate, reports)
    update_two_dataset_readiness(gate, reports)
    write_d5_scope_decision(gate, reports)
    return gate


def evaluate_gate(frame: pd.DataFrame, scenario_hashes: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    gate = cicids_mini_matrix.evaluate_gate(_frame_for_shared_gate(frame), scenario_hashes or {})
    gate["stage"] = "cesnet-mini-matrix-gate"
    gate["class_policy"] = SELECTED_POLICY
    gate["d5_allowed"] = False
    gate["blocking_reasons"] = [] if gate["passed"] else ["CESNET mini-matrix gate failed."]
    return gate


def write_mini_reports(frame: pd.DataFrame, gate: dict[str, Any], reports: str | Path = "reports") -> None:
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cesnet_mini_matrix_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report = {
        "stage": "cesnet-mini-matrix",
        "dataset": "cesnet_tls_year22",
        "reported_as": "CESNET-TLS-Year22",
        "rows": int(len(frame)),
        "class_policy": SELECTED_POLICY,
        "gate_passed": bool(gate.get("passed", False)),
        "results_csv": gate.get("results_csv"),
        "blocking_reasons": gate.get("blocking_reasons", []),
    }
    (out / "cesnet_mini_matrix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "cesnet_mini_matrix_report.md").write_text(_matrix_md(report), encoding="utf-8")
    (out / "cesnet_mini_matrix_gate.md").write_text(_gate_md(gate), encoding="utf-8")


def update_two_dataset_readiness(gate: dict[str, Any], reports: str | Path = "reports") -> dict[str, Any]:
    out = Path(reports)
    readiness_path = out / "realdata_readiness_report.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {"datasets": {}}
    cicids = readiness.setdefault("datasets", {}).setdefault("cicids2017", {})
    cesnet = readiness["datasets"].setdefault("cesnet_tls_year22", {})
    maltls = readiness["datasets"].setdefault("maltls22", {})
    optc = readiness["datasets"].setdefault("optc", {})
    cesnet_ready = bool(gate.get("passed", False))
    cicids_ready = bool(cicids.get("ready_for_d5_component") or cicids.get("mini_matrix_passed"))
    cesnet.update(
        {
            "ready_for_d5_component": cesnet_ready,
            "mini_matrix_passed": cesnet_ready,
            "class_policy": SELECTED_POLICY,
            "reported_as": "CESNET-TLS-Year22",
            "replacement_for": "MALTLS-22",
        }
    )
    maltls.update({"source_verified": False, "evaluated": False})
    optc.update({"available": False, "formal_experiment": False, "future_case_study_only": True})
    readiness["d5_allowed"] = bool(cicids_ready and cesnet_ready)
    readiness["d6_d7_allowed"] = False
    readiness["submission_ready"] = False
    readiness["d5_scope"] = ["CICIDS-2017", "CESNET-TLS-Year22"] if readiness["d5_allowed"] else []
    readiness_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    two = {
        "stage": "two-dataset-readiness",
        "cicids2017": {
            "ready_for_d5_component": cicids_ready,
            "class_policy": cicids.get("class_policy", "postfilter11"),
            "mini_matrix_passed": bool(cicids.get("mini_matrix_passed", cicids_ready)),
        },
        "cesnet_tls_year22": {
            "ready_for_d5_component": cesnet_ready,
            "class_policy": SELECTED_POLICY,
            "mini_matrix_passed": cesnet_ready,
            "reported_as": "CESNET-TLS-Year22",
            "replacement_for": "MALTLS-22",
        },
        "maltls22": {"source_verified": False, "evaluated": False},
        "optc": {"available": False, "formal_experiment": False, "future_case_study_only": True},
        "d5_allowed": bool(cicids_ready and cesnet_ready),
        "d5_scope": ["CICIDS-2017", "CESNET-TLS-Year22"] if cicids_ready and cesnet_ready else [],
        "d6_d7_allowed": False,
        "submission_ready": False,
    }
    (out / "two_dataset_readiness_report.json").write_text(json.dumps(two, indent=2), encoding="utf-8")
    (out / "two_dataset_readiness_report.md").write_text(_two_dataset_md(two), encoding="utf-8")
    return two


def write_d5_scope_decision(gate: dict[str, Any], reports: str | Path = "reports") -> dict[str, Any]:
    decision = {
        "stage": "d5-scope-decision",
        "main_experiment_datasets": [
            "CICIDS-2017 post-filtered 11-class setting",
            "CESNET-TLS-Year22" if gate.get("passed", False) else "CESNET-TLS-Year22 (pending gate)",
        ],
        "maltls22_evaluated": False,
        "optc_formal_experiment": False,
        "optc_rule": "Use only if the user provides real data/optc/events.csv; otherwise keep as future case study.",
        "future_d5_must_not_include": ["MALTLS-22"],
        "old_d5_d6_d7_results_invalid": True,
    }
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "d5_scope_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        "# D5 Scope Decision",
        "",
        "- Main experiment dataset 1: CICIDS-2017 post-filtered 11-class setting.",
        "- Main experiment dataset 2: CESNET-TLS-Year22, after its real-data gate passes.",
        "- MALTLS-22 is not evaluated.",
        "- OpTC is not a formal experiment unless real `events.csv` is provided.",
        "- Future D5 must not include MALTLS-22.",
        "- Old D5/D6/D7 results must not be cited.",
        "",
    ]
    (out / "d5_scope_decision.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def _inject_noise(bundle, spec: dict[str, Any], seed: int, graph_cache: dict[float, Any]):
    rng = np.random.default_rng(seed)
    if spec["noise_type"] == "clean":
        return bundle.y_train.copy(), np.zeros(bundle.y_train.shape[0], dtype=bool)
    if spec["noise_type"] == "symmetric":
        return inject_symmetric(bundle.y_train, spec["noise_rate"], bundle.num_classes, rng)
    if spec["noise_type"] == "asymmetric":
        return inject_asymmetric(bundle.y_train, spec["noise_rate"], bundle.meta.get("benign_class", 0) or 0, rng)
    beta = float(spec["graph_beta"])
    graph = None if np.isclose(beta, 0.0) else graph_cache.setdefault(beta, cicids_mini_matrix._lightweight_active_graph(bundle.y_train))
    return inject_graph_consistency(
        bundle.y_train,
        spec["noise_rate"],
        graph,
        {"num_classes": bundle.num_classes, "graph_consistency": {"consistency_bias": beta}},
        rng,
    )


def _load_smoke_report(reports: Path) -> dict[str, Any]:
    path = reports / "cesnet_smoke_report.json"
    if not path.exists():
        return {"passed": False, "blocking_reasons": ["CESNET smoke report missing; run smoke_realdata first."]}
    return json.loads(path.read_text(encoding="utf-8"))


def _blocked_gate(audit, smoke_report: dict[str, Any]) -> dict[str, Any]:
    reasons = list(audit.blocking_reasons)
    reasons.extend(smoke_report.get("blocking_reasons", []))
    if not smoke_report.get("passed", False):
        reasons.append("CESNET smoke gate has not passed; mini-matrix not run.")
    return {
        "stage": "cesnet-mini-matrix-gate",
        "passed": False,
        "status": "blocked",
        "rows": 0,
        "class_policy": SELECTED_POLICY,
        "results_csv": None,
        "d5_allowed": False,
        "blocking_reasons": sorted(set(reasons)),
    }


def _frame_for_shared_gate(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"removed_classes": "_removed_classes", "min_class_count": "_min_class_count"})


def _matrix_md(report: dict[str, Any]) -> str:
    lines = [
        "# CESNET Mini-Matrix Report",
        "",
        f"- Reported as: {report['reported_as']}",
        f"- Rows: {report['rows']}",
        f"- Class policy: `{report['class_policy']}`",
        f"- Gate passed: {report['gate_passed']}",
        f"- Results: `{report.get('results_csv')}`",
        "",
        "## Blocking Reasons",
    ]
    lines.extend([f"- {reason}" for reason in report.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _gate_md(gate: dict[str, Any]) -> str:
    lines = ["# CESNET Mini-Matrix Gate", "", f"- Passed: {gate.get('passed')}", f"- D5 allowed: {gate.get('d5_allowed')}", "", "## Blocking Reasons"]
    lines.extend([f"- {reason}" for reason in gate.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _two_dataset_md(report: dict[str, Any]) -> str:
    lines = [
        "# Two-Dataset Readiness Report",
        "",
        f"- CICIDS ready: {report['cicids2017']['ready_for_d5_component']}",
        f"- CESNET ready: {report['cesnet_tls_year22']['ready_for_d5_component']}",
        f"- MALTLS-22 evaluated: {report['maltls22']['evaluated']}",
        f"- OpTC formal experiment: {report['optc']['formal_experiment']}",
        f"- D5 allowed: {report['d5_allowed']}",
        f"- D5 scope: {', '.join(report['d5_scope']) if report['d5_scope'] else 'blocked'}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cesnet_tls_year22")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    report = run_mini_matrix(args.dataset, args.configs, args.out, args.reports)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
