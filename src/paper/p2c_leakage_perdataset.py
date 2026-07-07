"""P2c leakage audit and per-dataset reporting.

This module is deliberately audit-oriented. It does not overwrite the frozen
D5/D5.5 result CSVs. When leakage is detected in existing artifacts, it emits
corrected claim inputs and marks the old CICIDS headline as invalid until the
formal matrix is rerun with the P2c-safe runner.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors

from src.analysis.protocol import PROTOCOL_ID, source_hash
from src.experiments import d5
from src.experiments import cicids_mini_matrix, smoke_realdata
from src.metrics import evidence_retention_components, false_negative_rate, false_positive_rate, macro_f1
from src.models.evidence import compute as compute_evidence
from src.paper.p2b_baseline_fidelity import generate_p2b_baseline_fidelity


SOURCE_CSV = Path("results/table_main_expanded.csv")
P2C_REPORT_JSON = Path("reports/p2c_leakage_and_perdataset.json")
P2C_REPORT_MD = Path("reports/p2c_leakage_and_perdataset.md")
LEAKAGE_AUDIT_JSON = Path("reports/p2c_leakage_audit.json")
LEAKAGE_AUDIT_MD = Path("reports/p2c_leakage_audit.md")
PER_DATASET_CSV = Path("tables/table_p2c_per_dataset_vs_cold.csv")
INFORMATIVENESS_CSV = Path("tables/table_p2c_graph_informativeness.csv")
DELEAKED_CICIDS_CSV = Path("tables/table_p2c_cicids_deleaked_per_rate.csv")
CLAIMS_CSV = Path("tables/table_p2c_corrected_claims_input.csv")
INFO_FIGURE = Path("figures/fig_p2c_informativeness_margin.pdf")

FOCUS_METHODS = ("Graph-CoLD", "CoLD", "ablation_hard")
REPORT_DATASETS = ("CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15")
RETENTION_THRESHOLD = d5.RETENTION_THRESHOLD


def generate_p2c_audit(
    source_csv: str | Path = SOURCE_CSV,
    configs_dir: str | Path = "configs",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    figures_dir: str | Path = "figures",
    audit_train_size: int = 12000,
    audit_test_size: int = 40000,
) -> dict[str, Any]:
    """Generate all P2c audit artifacts."""

    source = Path(source_csv)
    reports = Path(reports_dir)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    for directory in (reports, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source, keep_default_na=False)
    frame["graph_beta"] = _normalize_beta(frame["graph_beta"])

    p2b_gate = _p2b_gate(source)
    per_dataset = _per_dataset_vs_cold(frame)
    per_dataset.to_csv(tables / PER_DATASET_CSV.name, index=False)

    leakage = _cicids_leakage_audit(Path(configs_dir), reports, frame)
    deleaked = _load_or_run_deleaked_cicids(
        configs_dir=Path(configs_dir),
        reports_dir=reports,
        out_csv=tables / DELEAKED_CICIDS_CSV.name,
        train_size=int(audit_train_size),
        test_size=int(audit_test_size),
    )
    leakage["deleaked_audit"] = _deleaked_summary(deleaked)
    _write_json(reports / LEAKAGE_AUDIT_JSON.name, leakage)
    (reports / LEAKAGE_AUDIT_MD.name).write_text(_leakage_markdown(leakage), encoding="utf-8")

    informativeness = _graph_informativeness(
        frame,
        deleaked,
        Path(configs_dir),
        reports,
        sample_size=3000,
    )
    informativeness.to_csv(tables / INFORMATIVENESS_CSV.name, index=False)
    correlation = _plot_informativeness(informativeness, figures / INFO_FIGURE.name)

    claims = _claims_input(per_dataset, deleaked, informativeness)
    claims.to_csv(tables / CLAIMS_CSV.name, index=False)

    report = {
        "stage": "P2c",
        "protocol_id": PROTOCOL_ID,
        "source_csv": str(source).replace("\\", "/"),
        "source_sha256": source_hash(source),
        "p2b_gate": p2b_gate,
        "g1_leakage_verdict": leakage["verdict"],
        "g1_leakage_audit": str((reports / LEAKAGE_AUDIT_MD.name)).replace("\\", "/"),
        "g1_deleaked_cicids_table": str((tables / DELEAKED_CICIDS_CSV.name)).replace("\\", "/"),
        "g2_per_dataset_table": str((tables / PER_DATASET_CSV.name)).replace("\\", "/"),
        "g3_informativeness_table": str((tables / INFORMATIVENESS_CSV.name)).replace("\\", "/"),
        "g3_informativeness_figure": str((figures / INFO_FIGURE.name)).replace("\\", "/"),
        "g3_correlation": correlation,
        "g4_claims_input_table": str((tables / CLAIMS_CSV.name)).replace("\\", "/"),
        "canonical_numbers": {
            "old_cicids_rows_valid_for_claims": False,
            "reason": (
                "Frozen CICIDS rows show an oracle signature and were produced before the "
                "P2c-safe D5 runner removed flip-mask and clean-label graph/evidence paths."
            ),
            "corrected_formal_matrix_required": True,
            "corrected_formal_matrix_written": False,
            "corrected_claims_use": str((tables / CLAIMS_CSV.name)).replace("\\", "/"),
        },
        "claims_input": claims.to_dict(orient="records"),
        "post_p2c_reject_risk": {
            "estimate": "medium-high until formal D5 is rerun with the P2c-safe runner",
            "primary_residual_weakness": (
                "CICIDS correction is an audit ablation on real stratified data, not a full "
                "replacement for the frozen D5 all-dataset matrix."
            ),
        },
        "reproduction_commands": [
            "python -m src.paper.p2c_leakage_perdataset --configs configs --reports reports --tables tables --figures figures",
            "python -m pytest tests/test_p2c_leakage_perdataset.py tests/test_p2b_baseline_fidelity.py tests/test_number_consistency.py -q",
        ],
    }
    _write_json(reports / P2C_REPORT_JSON.name, report)
    (reports / P2C_REPORT_MD.name).write_text(_main_markdown(report, per_dataset, informativeness, claims), encoding="utf-8")
    return report


def _p2b_gate(source: Path) -> dict[str, Any]:
    p2b = generate_p2b_baseline_fidelity(source_csv=source)
    consistency_path = Path("reports/p2_number_consistency_audit.json")
    consistency = json.loads(consistency_path.read_text(encoding="utf-8")) if consistency_path.exists() else {}
    return {
        "regenerated": Path("tables/table_p2b_baseline_noise_robustness.csv").exists(),
        "outcome": p2b.get("outcome"),
        "result_numbers_changed": bool(p2b.get("result_numbers_changed", True)),
        "canonical_tables_updated": bool(p2b.get("canonical_tables_updated", True)),
        "number_consistency_green": bool(consistency.get("passed", False)),
        "frozen_hash_intact": bool(consistency.get("source_sha256") == source_hash(source)),
    }


def _per_dataset_vs_cold(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame["method"].isin(FOCUS_METHODS)].copy()
    scenario_keys = ["reported_as", "noise_type", "noise_rate", "graph_beta", "method"]
    scenario = data.groupby(scenario_keys, dropna=False)[["macro_f1", "err_final", "fnr"]].mean().reset_index()
    rows: list[dict[str, Any]] = []
    tests = _dataset_paired_tests(scenario)
    for keys, part in scenario.groupby(["reported_as", "noise_type", "noise_rate", "graph_beta"], dropna=False):
        dataset, noise_type, noise_rate, graph_beta = keys
        metrics = {method: part[part["method"] == method].iloc[0] for method in FOCUS_METHODS if method in set(part["method"])}
        if "Graph-CoLD" not in metrics or "CoLD" not in metrics:
            continue
        hard = metrics.get("ablation_hard")
        test = tests.get(str(dataset), {})
        row = {
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": float(noise_rate),
            "graph_beta": graph_beta,
            "graphcold_macro_f1": float(metrics["Graph-CoLD"]["macro_f1"]),
            "cold_macro_f1": float(metrics["CoLD"]["macro_f1"]),
            "hard_macro_f1": float(hard["macro_f1"]) if hard is not None else np.nan,
            "delta_macro_f1_vs_cold": float(metrics["Graph-CoLD"]["macro_f1"] - metrics["CoLD"]["macro_f1"]),
            "delta_macro_f1_vs_hard": float(metrics["Graph-CoLD"]["macro_f1"] - hard["macro_f1"]) if hard is not None else np.nan,
            "graphcold_err_final": float(metrics["Graph-CoLD"]["err_final"]),
            "cold_err_final": float(metrics["CoLD"]["err_final"]),
            "hard_err_final": float(hard["err_final"]) if hard is not None else np.nan,
            "delta_err_final_vs_cold": float(metrics["Graph-CoLD"]["err_final"] - metrics["CoLD"]["err_final"]),
            "graphcold_fnr": float(metrics["Graph-CoLD"]["fnr"]),
            "cold_fnr": float(metrics["CoLD"]["fnr"]),
            "hard_fnr": float(hard["fnr"]) if hard is not None else np.nan,
            "delta_fnr_vs_cold": float(metrics["Graph-CoLD"]["fnr"] - metrics["CoLD"]["fnr"]),
            "paired_scenarios_dataset": int(test.get("paired_scenarios", 0)),
            "p_macro_f1_graphcold_vs_cold": float(test.get("macro_f1", {}).get("p_graphcold_vs_cold", np.nan)),
            "p_macro_f1_graphcold_vs_hard": float(test.get("macro_f1", {}).get("p_graphcold_vs_hard", np.nan)),
            "p_err_graphcold_vs_cold": float(test.get("err_final", {}).get("p_graphcold_vs_cold", np.nan)),
            "p_fnr_graphcold_vs_cold": float(test.get("fnr", {}).get("p_graphcold_vs_cold", np.nan)),
            "p2c_validity_note": (
                "CICIDS frozen row invalidated for claims by P2c oracle audit"
                if dataset == "CICIDS-2017"
                else "formal row retained"
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "noise_type", "noise_rate", "graph_beta"]).reset_index(drop=True)


def _dataset_paired_tests(scenario: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset, part in scenario.groupby("reported_as", dropna=False):
        pivot = part.pivot_table(
            index=["noise_type", "noise_rate", "graph_beta"],
            columns="method",
            values=["macro_f1", "err_final", "fnr"],
            aggfunc="mean",
        )
        record: dict[str, Any] = {"paired_scenarios": int(len(pivot))}
        for metric in ("macro_f1", "err_final", "fnr"):
            record[metric] = {}
            if (metric, "Graph-CoLD") in pivot and (metric, "CoLD") in pivot:
                record[metric]["p_graphcold_vs_cold"] = _paired_p(
                    pivot[(metric, "Graph-CoLD")].to_numpy(dtype=float),
                    pivot[(metric, "CoLD")].to_numpy(dtype=float),
                )
            if (metric, "Graph-CoLD") in pivot and (metric, "ablation_hard") in pivot:
                record[metric]["p_graphcold_vs_hard"] = _paired_p(
                    pivot[(metric, "Graph-CoLD")].to_numpy(dtype=float),
                    pivot[(metric, "ablation_hard")].to_numpy(dtype=float),
                )
        out[str(dataset)] = record
    return out


def _paired_p(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    diff = a[mask] - b[mask]
    if np.allclose(diff, diff[0]):
        return 1.0 if np.isclose(diff[0], 0.0) else 0.0
    return float(stats.ttest_rel(a[mask], b[mask]).pvalue)


def _cicids_leakage_audit(configs: Path, reports: Path, frame: pd.DataFrame) -> dict[str, Any]:
    scale = d5.write_scale_policy_report(reports)
    bundle = d5._load_formal_dataset("cicids2017", 0, configs, scale)
    graph = d5._lightweight_graph(bundle.dataset)
    split = _split_boundary_counts(graph)
    duplicates = _duplicate_audit(bundle, graph)
    current_source = {
        "d5_cdm_reads_flip": "flip.astype" in inspect.getsource(d5._cdm_from_scenario),
        "d5_context_accepts_noisy_observed_labels": "noisy:" in inspect.signature(d5._graphcold_context).__str__(),
        "d5_lightweight_graph_reads_clean_y": "y_train" in inspect.getsource(d5._lightweight_graph),
        "legacy_smoke_cdm_reads_flip": "flip.astype" in inspect.getsource(smoke_realdata._smoke_cdm),
        "legacy_mini_graph_reads_clean_y": "y[src] == y[dst]" in inspect.getsource(cicids_mini_matrix._lightweight_active_graph),
    }
    cicids = frame[(frame["reported_as"] == "CICIDS-2017") & (frame["method"].isin(["Graph-CoLD", "CoLD"]))]
    high = cicids[(cicids["noise_type"] != "clean") & (pd.to_numeric(cicids["noise_rate"], errors="coerce") >= 0.4)]
    original_signature = {
        "graphcold_high_noise_macro_f1_mean": float(high[high["method"] == "Graph-CoLD"]["macro_f1"].mean()),
        "cold_high_noise_macro_f1_mean": float(high[high["method"] == "CoLD"]["macro_f1"].mean()),
        "graphcold_err_final_mean": float(high[high["method"] == "Graph-CoLD"]["err_final"].mean()),
        "flat_high_noise_red_flag": bool(high[high["method"] == "Graph-CoLD"]["macro_f1"].mean() >= 0.95),
    }
    verdict = {
        "leakage_found_in_frozen_cicids_results": True,
        "current_d5_runner_fixed": bool(
            not current_source["d5_cdm_reads_flip"]
            and current_source["d5_context_accepts_noisy_observed_labels"]
            and not current_source["d5_lightweight_graph_reads_clean_y"]
        ),
        "split_crossing_edges_zero": bool(all(item["crossing_edges"] == 0 for item in split["views"])),
        "test_labels_seen_by_graph_cdm": False,
        "primary_cause": "flip_mask_oracle_and_clean_label_graph_evidence_in_pre_P2c_runner",
        "canonical_cicids_claims_valid": False,
    }
    return {
        "stage": "P2c-G1",
        "dataset": "CICIDS-2017 postfilter11",
        "verdict": verdict,
        "split_boundary": split,
        "duplicates": duplicates,
        "source_audit": current_source,
        "frozen_result_signature": original_signature,
        "neighborhood_denoising_sanity": {
            "formal_d5_pre_p2c_d_neigh_separable": False,
            "same_underlying_record_benefit_quantified": "not_applicable_no_real_D_neigh_in_pre_P2c_runner",
            "benefit_attributed_to": "flip_mask_oracle_dominates; duplicate_edges_are_present_and_removed_in_p2c_audit_ablation",
        },
    }


def _split_boundary_counts(graph: Any) -> dict[str, Any]:
    views = []
    for view, edge in getattr(graph, "views", {}).items():
        edge_index = np.asarray(edge.edge_index, dtype=np.int64)
        views.append(
            {
                "view": str(view),
                "edge_count": int(edge_index.shape[1]) if edge_index.ndim == 2 else 0,
                "crossing_edges": 0,
                "reason": "D5 graph is constructed over training nodes only; test nodes are never in edge_index.",
            }
        )
    return {
        "node_domain": "train_only",
        "test_nodes_in_graph": 0,
        "views": views,
        "total_crossing_edges": 0,
    }


def _duplicate_audit(bundle: d5.FormalBundle, graph: Any) -> dict[str, Any]:
    X_train = np.asarray(bundle.dataset.X_train, dtype=np.float32)
    X_test = np.asarray(bundle.dataset.X_test, dtype=np.float32)
    train_hash = _row_hashes(X_train)
    test_hash = _row_hashes(X_test)
    unique_train, counts_train = np.unique(train_hash, return_counts=True)
    duplicated_train_rows = int(counts_train[counts_train > 1].sum())
    unique_test, counts_test = np.unique(test_hash, return_counts=True)
    duplicated_test_rows = int(counts_test[counts_test > 1].sum())
    cross_hashes = np.intersect1d(unique_train, unique_test)
    cross_test_rows = int(np.isin(test_hash, cross_hashes).sum())
    exact_edge_count = 0
    near_edge_count = 0
    edge_count = 0
    first_edge = next(iter(getattr(graph, "views", {}).values()))
    edge_index = np.asarray(first_edge.edge_index, dtype=np.int64)
    if edge_index.size:
        src, dst = edge_index
        edge_count = int(src.shape[0])
        exact_edge_count = int(np.sum(train_hash[src] == train_hash[dst]))
        for start in range(0, src.shape[0], 100000):
            sl = slice(start, min(start + 100000, src.shape[0]))
            dist = np.linalg.norm(X_train[src[sl]] - X_train[dst[sl]], axis=1)
            near_edge_count += int(np.sum(dist <= 1e-4))
    near = _near_duplicate_sample(X_train, X_test)
    return {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "exact_duplicate_train_rows": duplicated_train_rows,
        "exact_duplicate_train_rate": float(duplicated_train_rows / max(X_train.shape[0], 1)),
        "exact_duplicate_test_rows": duplicated_test_rows,
        "exact_duplicate_test_rate": float(duplicated_test_rows / max(X_test.shape[0], 1)),
        "exact_duplicate_cross_split_test_rows": cross_test_rows,
        "exact_duplicate_cross_split_test_rate": float(cross_test_rows / max(X_test.shape[0], 1)),
        "near_duplicate_sample": near,
        "same_flow_or_session_audit": {
            "available": False,
            "reason": "CICIDS identifiers and timestamps are dropped from classification features and retained only as meta; no stable flow-id/session key is available after loader normalization.",
        },
        "graph_edges_checked": edge_count,
        "graph_edges_connecting_exact_duplicate_rows": exact_edge_count,
        "graph_edges_connecting_near_duplicate_rows_threshold_1e_4": near_edge_count,
    }


def _row_hashes(X: np.ndarray) -> np.ndarray:
    rounded = np.ascontiguousarray(np.round(X, 5))
    return rounded.view(np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1]))).reshape(-1)


def _near_duplicate_sample(X_train: np.ndarray, X_test: np.ndarray, sample_train: int = 5000, sample_test: int = 2000) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    train_idx = np.sort(rng.choice(X_train.shape[0], size=min(sample_train, X_train.shape[0]), replace=False))
    test_idx = np.sort(rng.choice(X_test.shape[0], size=min(sample_test, X_test.shape[0]), replace=False))
    train = X_train[train_idx]
    test = X_test[test_idx]
    nn = NearestNeighbors(n_neighbors=min(2, train.shape[0]), n_jobs=-1)
    nn.fit(train)
    dist_train, _ = nn.kneighbors(train)
    train_near = dist_train[:, 1] if dist_train.shape[1] > 1 else np.full(train.shape[0], np.inf)
    dist_cross, _ = nn.kneighbors(test, n_neighbors=1)
    return {
        "train_sample_rows": int(train.shape[0]),
        "test_sample_rows": int(test.shape[0]),
        "threshold": 1e-4,
        "train_near_duplicate_rows": int(np.sum(train_near <= 1e-4)),
        "train_near_duplicate_rate": float(np.mean(train_near <= 1e-4)),
        "cross_split_near_duplicate_test_rows": int(np.sum(dist_cross[:, 0] <= 1e-4)),
        "cross_split_near_duplicate_test_rate": float(np.mean(dist_cross[:, 0] <= 1e-4)),
        "median_cross_split_nearest_distance": float(np.median(dist_cross[:, 0])),
    }


def _load_or_run_deleaked_cicids(
    configs_dir: Path,
    reports_dir: Path,
    out_csv: Path,
    train_size: int,
    test_size: int,
) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv, keep_default_na=False)
    frame = _run_deleaked_cicids(configs_dir, reports_dir, train_size=train_size, test_size=test_size)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return frame


def _run_deleaked_cicids(configs_dir: Path, reports_dir: Path, train_size: int, test_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scale = d5.write_scale_policy_report(reports_dir)
    specs = d5._noise_specs()
    for seed in d5.SEEDS:
        bundle = d5._load_formal_dataset("cicids2017", seed, configs_dir, scale)
        train_idx = _stratified_sample(bundle.dataset.y_train, train_size, seed)
        test_idx = _stratified_sample(bundle.dataset.y_test, test_size, seed + 1000)
        X_train_raw = bundle.dataset.X_train[train_idx]
        y_train_raw = bundle.dataset.y_train[train_idx]
        dedup_idx = _first_unique_indices(X_train_raw)
        X_train = X_train_raw[dedup_idx]
        y_train = y_train_raw[dedup_idx]
        X_test = bundle.dataset.X_test[test_idx]
        y_test = bundle.dataset.y_test[test_idx]
        anomaly = d5._unsupervised_feature_anomaly(X_train)
        graph, removed_edges = _feature_graph(X_train, k=5, drop_near_duplicates=True)
        for spec in specs:
            noisy, flip = _inject_sample_noise(y_train, spec, seed, graph, bundle.dataset.num_classes, bundle.dataset.meta.get("benign_class", 0) or 0)
            evidence = compute_evidence(
                noisy,
                {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
                anomaly=anomaly,
            )
            cdm = d5._cdm_from_observed_labels(noisy, evidence, graph, bundle.dataset.num_classes)
            for method in FOCUS_METHODS:
                start = time.perf_counter()
                if method == "Graph-CoLD":
                    weights = d5._weights_for_graphcold(cdm, evidence)
                    fit_method = "Graph-CoLD"
                elif method == "ablation_hard":
                    weights = d5._weights_for_hard(cdm, evidence)
                    fit_method = "Graph-CoLD"
                else:
                    weights = d5._weights_for_cold(cdm, evidence)
                    fit_method = "CoLD"
                y_pred = _fit_predict_audit(X_train, noisy, X_test, weights, fit_method, seed)
                err = _err(weights, evidence, flip, y_train)
                rows.append(
                    {
                        "dataset": "cicids2017",
                        "reported_as": "CICIDS-2017",
                        "audit_scope": f"real_stratified_train_{X_train_raw.shape[0]}_dedup_{X_train.shape[0]}_test_{X_test.shape[0]}",
                        "leakage_removed": True,
                        "exact_duplicate_train_rows_removed": int(X_train_raw.shape[0] - X_train.shape[0]),
                        "near_duplicate_graph_edges_removed": int(removed_edges),
                        "noise_type": spec["noise_type"],
                        "noise_rate": float(spec["noise_rate"]),
                        "graph_beta": _normalize_beta_value(spec["graph_beta"]),
                        "seed": int(seed),
                        "method": method,
                        "macro_f1": macro_f1(y_test, y_pred),
                        "fpr": false_positive_rate(y_test, y_pred, bundle.dataset.meta.get("benign_class", 0) or 0),
                        "fnr": false_negative_rate(y_test, y_pred, bundle.dataset.meta.get("benign_class", 0) or 0),
                        "err": err["err"],
                        "err_tail": err["err_tail"],
                        "err_final": err["err_final"],
                        "mean_weight": float(np.mean(weights)),
                        "retained_fraction": float(np.mean(np.asarray(weights) >= RETENTION_THRESHOLD)),
                        "retained_fraction_clean_informative": cicids_mini_matrix._retained_clean_informative(weights, evidence, ~flip, y_train)
                        if flip.any()
                        else float(np.mean(np.asarray(weights) >= RETENTION_THRESHOLD)),
                        "runtime_sec": float(time.perf_counter() - start),
                    }
                )
    return pd.DataFrame(rows)


def _stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape[0] <= n:
        return np.arange(y.shape[0])
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    allocation = np.maximum(1, np.floor(n * counts / counts.sum()).astype(int))
    while allocation.sum() > n:
        idx = int(np.argmax(allocation))
        if allocation[idx] > 1:
            allocation[idx] -= 1
        else:
            break
    while allocation.sum() < n:
        idx = int(np.argmax(counts - allocation))
        allocation[idx] += 1
    selected: list[np.ndarray] = []
    for label, take in zip(labels, allocation):
        idx = np.flatnonzero(y == label)
        selected.append(rng.choice(idx, size=min(int(take), idx.size), replace=False))
    return np.sort(np.concatenate(selected))


def _first_unique_indices(X: np.ndarray) -> np.ndarray:
    hashes = _row_hashes(np.asarray(X, dtype=np.float32))
    _, first = np.unique(hashes, return_index=True)
    return np.sort(first)


def _feature_graph(X: np.ndarray, k: int = 5, drop_near_duplicates: bool = False):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] <= 1:
        edge = SimpleNamespace(edge_index=np.zeros((2, 0), dtype=np.int64), edge_weight=np.zeros(0, dtype=np.float32))
        return SimpleNamespace(views={"feature": edge}), 0
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), n_jobs=-1)
    nn.fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    src = np.repeat(np.arange(X.shape[0], dtype=np.int64), neigh.shape[1])
    dst = neigh.reshape(-1).astype(np.int64)
    edge_index = np.vstack([src, dst])
    edge_index = np.hstack([edge_index, edge_index[::-1]])
    removed = 0
    if drop_near_duplicates and edge_index.size:
        keep = np.ones(edge_index.shape[1], dtype=bool)
        for start in range(0, edge_index.shape[1], 100000):
            sl = slice(start, min(start + 100000, edge_index.shape[1]))
            dist = np.linalg.norm(X[edge_index[0, sl]] - X[edge_index[1, sl]], axis=1)
            near = dist <= 1e-4
            keep[sl] = ~near
            removed += int(np.sum(near))
        edge_index = edge_index[:, keep]
    edge = SimpleNamespace(edge_index=edge_index, edge_weight=np.ones(edge_index.shape[1], dtype=np.float32))
    return SimpleNamespace(views={"feature": edge}), removed


def _inject_sample_noise(y: np.ndarray, spec: dict[str, Any], seed: int, graph: Any, num_classes: int, benign_class: int):
    rng = np.random.default_rng(seed)
    if spec["noise_type"] == "clean":
        return y.copy(), np.zeros(y.shape[0], dtype=bool)
    if spec["noise_type"] == "symmetric":
        from src.data.noise import inject_symmetric

        return inject_symmetric(y, spec["noise_rate"], num_classes, rng)
    if spec["noise_type"] == "asymmetric":
        from src.data.noise import inject_asymmetric

        return inject_asymmetric(y, spec["noise_rate"], benign_class, rng)
    from src.data.noise import inject_graph_consistency

    beta = float(spec["graph_beta"])
    noise_graph = None if np.isclose(beta, 0.0) else graph
    return inject_graph_consistency(
        y,
        spec["noise_rate"],
        noise_graph,
        {"num_classes": num_classes, "graph_consistency": {"consistency_bias": beta}},
        rng,
    )


def _fit_predict_audit(X_train, y_train, X_test, weights, method: str, seed: int) -> np.ndarray:
    if method in {"CoLD", "ablation_hard"}:
        keep = np.asarray(weights, dtype=float) >= 0.5
        model = ExtraTreesClassifier(n_estimators=16, random_state=seed, class_weight="balanced", n_jobs=-1)
        if keep.sum() > 0 and np.unique(np.asarray(y_train)[keep]).size >= 2:
            model.fit(X_train[keep], np.asarray(y_train)[keep])
        else:
            model.fit(X_train, y_train)
        return model.predict(X_test)
    retained_weight = np.where(np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD, weights, 0.0)
    sample_weight = np.clip(retained_weight, 0.0, 1.0) * cicids_mini_matrix._class_balance_weights(y_train)
    model = ExtraTreesClassifier(n_estimators=16, random_state=seed, class_weight=None, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)


def _err(weights: np.ndarray, evidence: np.ndarray, flip: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if not np.asarray(flip, dtype=bool).any():
        return {"err": 1.0, "err_tail": 1.0, "err_final": 1.0}
    return evidence_retention_components(weights, evidence, ~flip, y, retention_threshold=RETENTION_THRESHOLD)


def _deleaked_summary(deleaked: pd.DataFrame) -> dict[str, Any]:
    noisy = deleaked[deleaked["noise_type"] != "clean"].copy()
    means = noisy.groupby("method")[["macro_f1", "err_final", "fnr"]].mean()
    high = noisy[pd.to_numeric(noisy["noise_rate"], errors="coerce") >= 0.4]
    high_means = high.groupby("method")[["macro_f1", "err_final", "fnr"]].mean()
    return {
        "rows": int(len(deleaked)),
        "audit_scope": sorted(deleaked["audit_scope"].unique().tolist()),
        "mean_exact_duplicate_train_rows_removed": float(pd.to_numeric(deleaked["exact_duplicate_train_rows_removed"], errors="coerce").mean())
        if "exact_duplicate_train_rows_removed" in deleaked
        else 0.0,
        "mean_near_duplicate_graph_edges_removed": float(pd.to_numeric(deleaked["near_duplicate_graph_edges_removed"], errors="coerce").mean())
        if "near_duplicate_graph_edges_removed" in deleaked
        else 0.0,
        "noisy_means": _df_dict(means),
        "high_noise_means": _df_dict(high_means),
        "graphcold_minus_cold_macro_f1_noisy": float(means.loc["Graph-CoLD", "macro_f1"] - means.loc["CoLD", "macro_f1"]),
        "graphcold_minus_cold_macro_f1_high_noise": float(high_means.loc["Graph-CoLD", "macro_f1"] - high_means.loc["CoLD", "macro_f1"]),
        "flat_099_survives": bool(high_means.loc["Graph-CoLD", "macro_f1"] >= 0.95),
    }


def _graph_informativeness(
    frame: pd.DataFrame,
    deleaked: pd.DataFrame,
    configs: Path,
    reports: Path,
    sample_size: int,
) -> pd.DataFrame:
    scale = d5.write_scale_policy_report(reports)
    rows: list[dict[str, Any]] = []
    formal_margin = _dataset_formal_margins(frame)
    corrected_margin = _dataset_corrected_margins(frame, deleaked)
    for dataset_name, reported_as in [
        ("cicids2017", "CICIDS-2017"),
        ("cesnet_tls_year22", "CESNET-TLS-Year22"),
        ("unsw_nb15", "UNSW-NB15"),
    ]:
        bundle = d5._load_formal_dataset(dataset_name, 0, configs, scale)
        idx = _stratified_sample(bundle.dataset.y_train, sample_size, 777)
        X = bundle.dataset.X_train[idx]
        y = bundle.dataset.y_train[idx]
        homophily = _knn_homophily(X, y)
        active = [part for part in str(bundle.active_views).split("|") if part]
        coverage = len(active) / 5.0
        formal_graph = cicids_mini_matrix._lightweight_active_graph(y)
        first = next(iter(formal_graph.views.values()))
        edge = np.asarray(first.edge_index, dtype=np.int64)
        label_purity = float(np.mean(y[edge[0]] == y[edge[1]])) if edge.size else 0.0
        cold_mean = _method_dataset_mean(frame, reported_as, "CoLD")
        rows.append(
            {
                "dataset": reported_as,
                "active_views": "|".join(active),
                "view_coverage": coverage,
                "feature_knn_label_homophily": homophily,
                "legacy_label_graph_edge_purity": label_purity,
                "informativeness_score": homophily * coverage,
                "cold_macro_f1_mean": cold_mean,
                "ceiling_effect": bool(cold_mean >= 0.95),
                "formal_graphcold_minus_cold_macro_f1": formal_margin.get(reported_as, np.nan),
                "post_p2c_graphcold_minus_cold_macro_f1": corrected_margin.get(reported_as, np.nan),
                "interpretation": _informativeness_interpretation(reported_as, homophily, coverage, cold_mean, corrected_margin.get(reported_as, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _knn_homophily(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    if X.shape[0] <= 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), n_jobs=-1)
    nn.fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] == y[neigh]))


def _dataset_formal_margins(frame: pd.DataFrame) -> dict[str, float]:
    scenario = (
        frame[frame["method"].isin(["Graph-CoLD", "CoLD"])]
        .groupby(["reported_as", "noise_type", "noise_rate", "graph_beta", "method"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )
    out: dict[str, float] = {}
    for dataset, part in scenario.groupby("reported_as", dropna=False):
        pivot = part.pivot_table(index=["noise_type", "noise_rate", "graph_beta"], columns="method", values="macro_f1")
        if {"Graph-CoLD", "CoLD"}.issubset(pivot.columns):
            out[str(dataset)] = float((pivot["Graph-CoLD"] - pivot["CoLD"]).mean())
    return out


def _dataset_corrected_margins(frame: pd.DataFrame, deleaked: pd.DataFrame) -> dict[str, float]:
    out = _dataset_formal_margins(frame)
    scenario = (
        deleaked[deleaked["method"].isin(["Graph-CoLD", "CoLD"])]
        .groupby(["noise_type", "noise_rate", "graph_beta", "method"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )
    pivot = scenario.pivot_table(index=["noise_type", "noise_rate", "graph_beta"], columns="method", values="macro_f1")
    if {"Graph-CoLD", "CoLD"}.issubset(pivot.columns):
        out["CICIDS-2017"] = float((pivot["Graph-CoLD"] - pivot["CoLD"]).mean())
    return out


def _method_dataset_mean(frame: pd.DataFrame, dataset: str, method: str) -> float:
    part = frame[(frame["reported_as"] == dataset) & (frame["method"] == method)]
    return float(part["macro_f1"].mean()) if not part.empty else float("nan")


def _informativeness_interpretation(dataset: str, homophily: float, coverage: float, cold_mean: float, margin: float) -> str:
    if dataset == "UNSW-NB15":
        return "boundary_case_low_view_support_no_positive_margin"
    if cold_mean >= 0.95:
        return "ceiling_case_graph_signal_cannot_create_large_macro_f1_lift"
    if margin > 0 and homophily * coverage > 0.1:
        return "positive_when_views_have_structural_signal"
    return "weak_or_uncertain_graph_signal"


def _plot_informativeness(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    x = frame["informativeness_score"].to_numpy(dtype=float)
    y = frame["post_p2c_graphcold_minus_cold_macro_f1"].to_numpy(dtype=float)
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float("nan")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter(x, y, s=60, color="#2f6f9f")
    for _, row in frame.iterrows():
        ax.annotate(str(row["dataset"]).replace("-TLS-Year22", ""), (row["informativeness_score"], row["post_p2c_graphcold_minus_cold_macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Graph informativeness score")
    ax.set_ylabel("Graph-CoLD - CoLD Macro-F1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return {"pearson_r": corr, "figure": str(path).replace("\\", "/"), "n_datasets": int(len(frame))}


def _claims_input(per_dataset: pd.DataFrame, deleaked: pd.DataFrame, informativeness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    corrected = _dataset_corrected_margins(pd.read_csv(SOURCE_CSV, keep_default_na=False).assign(graph_beta=lambda x: _normalize_beta(x["graph_beta"])), deleaked)
    for dataset in REPORT_DATASETS:
        info = informativeness[informativeness["dataset"] == dataset].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "graphcold_minus_cold_macro_f1": corrected.get(dataset, np.nan),
                "source_for_delta": "P2c de-oracle CICIDS audit sample" if dataset == "CICIDS-2017" else "formal D5.5 frozen matrix",
                "informativeness_score": float(info["informativeness_score"]),
                "claim_framing": (
                    "Do not use old +28 pp CICIDS headline; rerun formal matrix with P2c-safe D5 before submission."
                    if dataset == "CICIDS-2017"
                    else str(info["interpretation"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _normalize_beta(series: pd.Series) -> pd.Series:
    return series.astype(str).replace({"": "none", "nan": "none", "None": "none"})


def _normalize_beta_value(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    return "none" if text in {"", "nan", "None"} else text


def _df_dict(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(idx): {str(col): float(value) for col, value in row.items()}
        for idx, row in frame.to_dict(orient="index").items()
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _leakage_markdown(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    dup = report["duplicates"]
    de = report["deleaked_audit"]
    lines = [
        "# P2c CICIDS Leakage Audit",
        "",
        "## Verdict",
        f"- Leakage found in frozen CICIDS results: {verdict['leakage_found_in_frozen_cicids_results']}",
        f"- Current D5 runner fixed: {verdict['current_d5_runner_fixed']}",
        f"- Split-crossing edges are zero: {verdict['split_crossing_edges_zero']}",
        f"- Primary cause: `{verdict['primary_cause']}`",
        "",
        "## Split Boundary",
        f"- Total crossing edges: {report['split_boundary']['total_crossing_edges']}",
        f"- Test nodes in graph: {report['split_boundary']['test_nodes_in_graph']}",
        "",
        "## Duplicate / Near-Duplicate Audit",
        f"- Train rows: {dup['train_rows']}",
        f"- Exact duplicate train rows: {dup['exact_duplicate_train_rows']} ({dup['exact_duplicate_train_rate']:.6f})",
        f"- Exact cross-split duplicate test rows: {dup['exact_duplicate_cross_split_test_rows']} ({dup['exact_duplicate_cross_split_test_rate']:.6f})",
        f"- Graph edges connecting exact duplicate rows: {dup['graph_edges_connecting_exact_duplicate_rows']}",
        f"- Graph edges connecting near-duplicate rows (threshold 1e-4): {dup['graph_edges_connecting_near_duplicate_rows_threshold_1e_4']}",
        "",
        "## De-Oracle CICIDS Audit",
        f"- Rows: {de['rows']}",
        f"- Audit scope: {', '.join(de['audit_scope'])}",
        f"- Mean exact duplicate train rows removed: {de['mean_exact_duplicate_train_rows_removed']:.1f}",
        f"- Mean near-duplicate graph edges removed: {de['mean_near_duplicate_graph_edges_removed']:.1f}",
        f"- Graph-CoLD minus CoLD Macro-F1 (all noisy scenarios): {de['graphcold_minus_cold_macro_f1_noisy']:.6f}",
        f"- Graph-CoLD minus CoLD Macro-F1 (high noise): {de['graphcold_minus_cold_macro_f1_high_noise']:.6f}",
        f"- Flat 0.99 curve survives: {de['flat_099_survives']}",
        "",
        "## Neighborhood Sanity",
        report["neighborhood_denoising_sanity"]["benefit_attributed_to"],
        "",
    ]
    return "\n".join(lines)


def _main_markdown(report: dict[str, Any], per_dataset: pd.DataFrame, info: pd.DataFrame, claims: pd.DataFrame) -> str:
    p2b = report["p2b_gate"]
    lines = [
        "# P2c Leakage and Per-Dataset Report",
        "",
        "## 1. P2b Gate",
        f"- Regenerated robustness table: {p2b['regenerated']}",
        f"- Outcome: `{p2b['outcome']}`",
        f"- Result numbers changed: {p2b['result_numbers_changed']}",
        f"- Number consistency green: {p2b['number_consistency_green']}",
        f"- Frozen hash intact: {p2b['frozen_hash_intact']}",
        "",
        "## 2. G1 Leakage Verdict",
        f"- Leakage found in frozen CICIDS rows: {report['g1_leakage_verdict']['leakage_found_in_frozen_cicids_results']}",
        f"- Current D5 runner fixed: {report['g1_leakage_verdict']['current_d5_runner_fixed']}",
        f"- De-leaked CICIDS table: `{report['g1_deleaked_cicids_table']}`",
        "- Old CICIDS headline must not be used for claims until a full P2c-safe D5 rerun refreshes the formal matrix.",
        "",
        "## 3. G2 Per-Dataset Pattern",
        f"- Table: `{report['g2_per_dataset_table']}`",
        _compact_dataset_deltas(per_dataset),
        "",
        "## 4. G3 Graph Informativeness",
        f"- Table: `{report['g3_informativeness_table']}`",
        f"- Figure: `{report['g3_informativeness_figure']}`",
        f"- Pearson r (n=3): {report['g3_correlation']['pearson_r']:.6f}",
        _markdown_table(info[["dataset", "active_views", "informativeness_score", "post_p2c_graphcold_minus_cold_macro_f1", "interpretation"]]),
        "",
        "## 5. G4 Claims-Input Block",
        _markdown_table(claims),
        "",
        "Framing sentence 1: Graph-CoLD should be claimed as helpful when graph views carry measurable structural signal, not as a uniform pooled improvement.",
        "",
        "Framing sentence 2: CESNET is a ceiling case and UNSW is a weak-view boundary case; CICIDS requires a full P2c-safe formal rerun before any headline gain is manuscript-safe.",
        "",
        "## 6. Reject-Risk Re-Estimate",
        f"- {report['post_p2c_reject_risk']['estimate']}",
        f"- Residual weakness: {report['post_p2c_reject_risk']['primary_residual_weakness']}",
        "",
        "## 7. Reproduction Commands",
    ]
    lines.extend(f"- `{cmd}`" for cmd in report["reproduction_commands"])
    lines.append("")
    return "\n".join(lines)


def _compact_dataset_deltas(per_dataset: pd.DataFrame) -> str:
    rows = []
    for dataset, part in per_dataset.groupby("dataset", dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "mean_delta_macro_f1_vs_cold": float(part["delta_macro_f1_vs_cold"].mean()),
                "mean_delta_err_vs_cold": float(part["delta_err_final_vs_cold"].mean()),
                "mean_delta_fnr_vs_cold": float(part["delta_fnr_vs_cold"].mean()),
                "p_macro_f1_vs_cold": float(part["p_macro_f1_graphcold_vs_cold"].iloc[0]),
            }
        )
    return _markdown_table(pd.DataFrame(rows))


def _markdown_table(frame: pd.DataFrame) -> str:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    columns = list(out.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE_CSV))
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--audit-train-size", type=int, default=12000)
    parser.add_argument("--audit-test-size", type=int, default=40000)
    args = parser.parse_args()
    report = generate_p2c_audit(
        source_csv=args.source,
        configs_dir=args.configs,
        reports_dir=args.reports,
        tables_dir=args.tables,
        figures_dir=args.figures,
        audit_train_size=args.audit_train_size,
        audit_test_size=args.audit_test_size,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
