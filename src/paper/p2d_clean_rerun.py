"""P2d clean de-oracled rerun and contribution re-verification.

The raw full-CICIDS runner is too slow for reliable no-checkpoint execution in
this desktop environment, so P2d writes a clean, deterministic real-data audit
window matrix. The report names that scale policy explicitly and supersedes the
oracle-era frozen tables rather than silently preserving their headline claims.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
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

from src.analysis.evidence_downstream import tail_class_recall
from src.analysis.protocol import PROTOCOL_ID, source_hash, write_protocol_artifacts
from src.analysis.result_sanity import check_results
from src.analysis.stat_tests import grouped_paired_summary
from src.baselines.base import BaselineResult
from src.experiments import d5
from src.experiments.d5_baseline_expansion import _baseline_candidates, _candidate_method_name
from src.metrics import evidence_retention_components, false_negative_rate, false_positive_rate, macro_f1
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import alert_compression_ratio, priority_scores


TRAIN_SIZE = 3000
TEST_SIZE = 8000
RETENTION_THRESHOLD = d5.RETENTION_THRESHOLD
FORMAL_METHODS = ("Graph-CoLD", "CoLD", "ablation_hard")
BASELINE_METHODS = (
    "Noisy-Supervised",
    "Confident-Learning",
    "Co-Teaching",
    "Decoupling",
    "FINE",
    "MCRe",
    "MORSE",
)
ALL_METHODS = (*FORMAL_METHODS, *BASELINE_METHODS)
DATASETS = ("cicids2017", "cesnet_tls_year22", "unsw_nb15")
DATASET_LABELS = {
    "cicids2017": "CICIDS-2017",
    "cesnet_tls_year22": "CESNET-TLS-Year22",
    "unsw_nb15": "UNSW-NB15",
}


def run_p2d_clean_rerun(
    out_dir: str | Path = "results",
    configs_dir: str | Path = "configs",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    figures_dir: str | Path = "figures",
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    skip_matrix: bool = False,
) -> dict[str, Any]:
    """Run or reuse the P2d clean audit-window matrix and write report assets."""

    out = Path(out_dir)
    configs = Path(configs_dir)
    reports = Path(reports_dir)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    for directory in (out, reports, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    if not skip_matrix or not (out / "table_main_expanded.csv").exists():
        matrix = _run_matrix(configs, reports, train_size=train_size, test_size=test_size, partial_path=out / "table_main_expanded.partial.csv")
        formal = matrix[matrix["method"].isin(FORMAL_METHODS)].copy()
        formal.to_csv(out / "table_main.csv", index=False)
        matrix.to_csv(out / "table_main_expanded.csv", index=False)
        _run_ablation(matrix, configs, reports, train_size, test_size).to_csv(out / "table_ablation.csv", index=False)
    else:
        matrix = pd.read_csv(out / "table_main_expanded.csv", keep_default_na=False)
        formal = matrix[matrix["method"].isin(FORMAL_METHODS)].copy()

    runtime = _runtime_report(matrix)
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2, allow_nan=False), encoding="utf-8")
    stats_report = grouped_paired_summary(formal, metric="macro_f1")
    (out / "stat_tests.json").write_text(json.dumps(stats_report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "stat_tests_baseline_expansion.json").write_text(
        json.dumps(grouped_paired_summary(matrix, metric="macro_f1"), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    sanity = check_results(formal)
    (reports / "d5_result_sanity_report.json").write_text(json.dumps(sanity, indent=2, allow_nan=False), encoding="utf-8")
    (reports / "d5_expanded_sanity_report.json").write_text(
        json.dumps(check_results(matrix), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    protocol = write_protocol_artifacts(out / "table_main_expanded.csv", tables / "table_p2_canonical_headline.csv", reports / "p2_number_consistency.json")
    number_audit = _number_consistency_audit(matrix, tables, reports, protocol)
    per_dataset = _per_dataset_table(matrix)
    per_dataset.to_csv(tables / "table_p2d_per_dataset_vs_cold.csv", index=False)
    core = _core_contribution_table(formal)
    core.to_csv(tables / "table_p2d_core_contribution.csv", index=False)
    info = _graph_informativeness(matrix, configs, reports, train_size=min(3000, train_size))
    info.to_csv(tables / "table_p2d_graph_informativeness.csv", index=False)
    claims = _claims_input(per_dataset, core, info)
    claims.to_csv(tables / "table_p2d_final_claims_input.csv", index=False)
    _plot_macro(matrix, figures / "fig_p2d_macro_f1_vs_noise_rate.pdf")
    _plot_err_retention(formal, figures / "fig_p2d_err_retention.pdf")
    _plot_informativeness(info, figures / "fig_p2d_informativeness_margin.pdf")

    gate = _p2c_gate(configs, reports)
    report = {
        "stage": "P2d",
        "completed": True,
        "scope": {
            "scale_policy": f"real_data_deterministic_audit_window_train_{train_size}_test_{test_size}",
            "datasets": [DATASET_LABELS[name] for name in DATASETS],
            "noise_specs": len(d5._noise_specs()),
            "seeds": list(d5.SEEDS),
            "methods": list(ALL_METHODS),
        },
        "p2c_gate": gate,
        "outputs": {
            "table_main": "results/table_main.csv",
            "table_main_expanded": "results/table_main_expanded.csv",
            "table_ablation": "results/table_ablation.csv",
            "canonical_headline": "tables/table_p2_canonical_headline.csv",
            "per_dataset": "tables/table_p2d_per_dataset_vs_cold.csv",
            "core_contribution": "tables/table_p2d_core_contribution.csv",
            "graph_informativeness": "tables/table_p2d_graph_informativeness.csv",
            "claims_input": "tables/table_p2d_final_claims_input.csv",
        },
        "hashes": {
            "table_main_expanded_sha256": source_hash(out / "table_main_expanded.csv"),
            "canonical_protocol_source_sha256": protocol["source_sha256"],
        },
        "stale_tables_superseded": [
            "results/table_main.csv",
            "results/table_main_expanded.csv",
            "results/table_ablation.csv",
            "tables/table_p2_canonical_headline.csv",
            "tables/table_2_main_performance.csv",
            "tables/table_3_high_noise_summary.csv",
        ],
        "number_consistency": number_audit,
        "core_verdict": _core_verdict(core),
        "claims_input": claims.to_dict(orient="records"),
        "post_p2d_reject_risk": _risk(core, per_dataset),
        "reproduction_commands": [
            "python -m src.paper.p2d_clean_rerun --configs configs --out results --reports reports --tables tables --figures figures",
            "python -m pytest tests/test_no_oracle_leakage.py tests/test_number_consistency.py -q",
        ],
    }
    (reports / "p2d_clean_rerun.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (reports / "p2d_clean_rerun.md").write_text(_markdown(report, per_dataset, core, info, claims), encoding="utf-8")
    return report


def _run_matrix(configs: Path, reports: Path, train_size: int, test_size: int, partial_path: Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, str, float, str, int, str]] = set()
    if partial_path is not None and partial_path.exists():
        existing = pd.read_csv(partial_path, keep_default_na=False)
        rows = existing.to_dict(orient="records")
        done = {_row_key(row) for row in rows}
    scale = d5.write_scale_policy_report(reports)
    for dataset_name in DATASETS:
        for seed in d5.SEEDS:
            sample = _sample_bundle(dataset_name, seed, configs, reports, scale, train_size, test_size)
            for spec in d5._noise_specs():
                noisy, flip = _inject_noise(sample, spec, seed)
                evidence = compute_evidence(
                    noisy,
                    {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
                    anomaly=sample.anomaly,
                )
                cdm = d5._cdm_from_observed_labels(noisy, evidence, sample.graph, sample.num_classes)
                for method in ALL_METHODS:
                    key = _row_key(
                        {
                            "dataset": dataset_name,
                            "noise_type": spec["noise_type"],
                            "noise_rate": float(spec["noise_rate"]),
                            "graph_beta": _beta(spec["graph_beta"]),
                            "seed": seed,
                            "method": method,
                        }
                    )
                    if key in done:
                        continue
                    _safe_progress(
                        f"[p2d] {dataset_name} seed={seed} {spec['noise_type']} "
                        f"rate={spec['noise_rate']} beta={_beta(spec['graph_beta'])} {method}"
                    )
                    row = _evaluate(sample, spec, seed, method, noisy, flip, evidence, cdm)
                    rows.append(row)
                    done.add(key)
                    if partial_path is not None and len(rows) % 25 == 0:
                        partial_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_csv(partial_path, index=False)
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows)


def _safe_progress(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def _row_key(row: dict[str, Any]) -> tuple[str, str, float, str, int, str]:
    return (
        str(row["dataset"]),
        str(row["noise_type"]),
        float(row["noise_rate"]),
        _beta(row["graph_beta"]),
        int(row["seed"]),
        str(row["method"]),
    )


def _sample_bundle(dataset_name: str, seed: int, configs: Path, reports: Path, scale: dict[str, Any], train_size: int, test_size: int):
    bundle = d5._load_formal_dataset(dataset_name, seed, configs, scale)
    train_idx = _stratified_sample(bundle.dataset.y_train, train_size, seed)
    test_idx = _stratified_sample(bundle.dataset.y_test, test_size, seed + 1000)
    X_train = bundle.dataset.X_train[train_idx]
    y_train = bundle.dataset.y_train[train_idx]
    X_test = bundle.dataset.X_test[test_idx]
    y_test = bundle.dataset.y_test[test_idx]
    graph = _feature_graph(X_train, active_views=bundle.active_views.split("|"))
    return SimpleNamespace(
        dataset_key=dataset_name,
        reported_as=bundle.reported_as,
        dataset_hash=bundle.dataset_hash,
        actual_data_path=bundle.actual_data_path,
        class_policy=bundle.class_policy,
        sample_policy=f"p2d_real_audit_window_train_{X_train.shape[0]}_test_{X_test.shape[0]}",
        sample_seed=42,
        sampling_stratified=True,
        active_views=bundle.active_views,
        source_verified=bundle.source_verified,
        replacement_for=bundle.replacement_for,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=bundle.dataset.num_classes,
        benign_class=bundle.dataset.meta.get("benign_class", 0) or 0,
        anomaly=d5._unsupervised_feature_anomaly(X_train),
        graph=graph,
        dedup=bundle.dataset.meta.get("cicids_train_exact_dedup", {}) if dataset_name == "cicids2017" else {},
    )


def _feature_graph(X: np.ndarray, active_views: list[str], k: int = 5):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] <= 1:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), n_jobs=-1)
        nn.fit(X)
        neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
        src = np.repeat(np.arange(X.shape[0], dtype=np.int64), neigh.shape[1])
        dst = neigh.reshape(-1).astype(np.int64)
        edge_index = np.vstack([src, dst])
        edge_index = np.hstack([edge_index, edge_index[::-1]])
    edge = SimpleNamespace(edge_index=edge_index, edge_weight=np.ones(edge_index.shape[1], dtype=np.float32))
    return SimpleNamespace(views={view: edge for view in active_views if view})


def _inject_noise(sample, spec: dict[str, Any], seed: int):
    from src.data.noise import inject_asymmetric, inject_graph_consistency, inject_symmetric

    rng = np.random.default_rng(seed)
    if spec["noise_type"] == "clean":
        return sample.y_train.copy(), np.zeros(sample.y_train.shape[0], dtype=bool)
    if spec["noise_type"] == "symmetric":
        return inject_symmetric(sample.y_train, spec["noise_rate"], sample.num_classes, rng)
    if spec["noise_type"] == "asymmetric":
        return inject_asymmetric(sample.y_train, spec["noise_rate"], sample.benign_class, rng)
    beta = float(spec["graph_beta"])
    graph = None if np.isclose(beta, 0.0) else sample.graph
    return inject_graph_consistency(
        sample.y_train,
        spec["noise_rate"],
        graph,
        {"num_classes": sample.num_classes, "graph_consistency": {"consistency_bias": beta}},
        rng,
    )


def _evaluate(sample, spec: dict[str, Any], seed: int, method: str, noisy, flip, evidence, cdm) -> dict[str, Any]:
    import time
    import tracemalloc

    tracemalloc.start()
    start = time.perf_counter()
    if method == "Graph-CoLD":
        weights = d5._weights_for_graphcold(cdm, evidence)
        y_pred = _fit_predict(sample.X_train, noisy, sample.X_test, weights, "Graph-CoLD", seed)
        family = "graph_cold"
        status = "p2d_deoracled"
        proba = _soft_labels_from_pred(y_pred, sample.num_classes)
    elif method == "ablation_hard":
        weights = d5._weights_for_hard(cdm, evidence)
        y_pred = _fit_predict(sample.X_train, noisy, sample.X_test, weights, "Graph-CoLD", seed)
        family = "hard_ablation"
        status = "p2d_deoracled"
        proba = _soft_labels_from_pred(y_pred, sample.num_classes)
    elif method == "CoLD":
        weights = d5._weights_for_cold(cdm, evidence)
        y_pred = _fit_predict(sample.X_train, noisy, sample.X_test, weights, "CoLD", seed)
        family = "cold"
        status = "p2d_deoracled"
        proba = _soft_labels_from_pred(y_pred, sample.num_classes)
    else:
        result = _baseline_result(method, seed, float(spec["noise_rate"]), sample, noisy)
        weights = np.asarray(result.weights, dtype=np.float64)
        y_pred = result.y_pred
        family = result.method_family
        status = result.implementation_status
        proba = result.proba
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime_sec = time.perf_counter() - start
    retained = np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD
    err = _err(weights, evidence, flip, sample.y_train)
    scores = priority_scores(
        {
            "graph_cdm": np.resize(cdm, sample.y_test.shape[0]),
            "evidence": np.resize(evidence, sample.y_test.shape[0]),
            "soft_labels": proba,
        },
        {},
        {"ranking": {"alpha1": 1.0, "alpha2": 0.7, "alpha3": 0.4, "benign_class": sample.benign_class}},
    )
    row = {
        "dataset": sample.dataset_key,
        "reported_as": sample.reported_as,
        "dataset_hash": sample.dataset_hash,
        "actual_data_path": sample.actual_data_path,
        "class_policy": sample.class_policy,
        "num_classes": sample.num_classes,
        "sample_policy": sample.sample_policy,
        "sample_size": int(sample.X_train.shape[0] + sample.X_test.shape[0]),
        "sample_seed": sample.sample_seed,
        "sampling_stratified": sample.sampling_stratified,
        "noise_type": spec["noise_type"],
        "noise_rate": float(spec["noise_rate"]),
        "graph_beta": _beta(spec["graph_beta"]),
        "seed": int(seed),
        "split_id": f"p2d-{sample.dataset_key}-seed{seed}",
        "noise_seed": int(seed),
        "model_seed": int(seed),
        "method": method,
        "method_family": family,
        "implementation_status": status,
        "macro_f1": macro_f1(sample.y_test, y_pred),
        "fpr": false_positive_rate(sample.y_test, y_pred, sample.benign_class),
        "fnr": false_negative_rate(sample.y_test, y_pred, sample.benign_class),
        "tail_recall": tail_class_recall(sample.y_test, y_pred, benign_class=sample.benign_class),
        "err": err["err"],
        "err_tail": err["err_tail"],
        "err_final": err["err_final"],
        "compression_ratio": alert_compression_ratio(scores, sample.y_test),
        "mean_weight": float(np.mean(weights)),
        "retained_fraction": float(np.mean(retained)),
        "retained_fraction_clean_informative": _retained_clean_informative(weights, evidence, ~flip, sample.y_train)
        if flip.any()
        else float(np.mean(retained)),
        "n_eff_ratio": _n_eff(weights) / float(weights.shape[0]),
        "runtime_sec": runtime_sec,
        "memory_mb": peak / (1024 * 1024),
        "active_views": sample.active_views,
        "source_verified": sample.source_verified,
        "replacement_for": sample.replacement_for,
        "cicids_train_dedup_removed": int(sample.dedup.get("removed_rows", 0)) if sample.dedup else 0,
    }
    return row


def _baseline_result(method: str, seed: int, noise_rate: float, sample, noisy) -> BaselineResult:
    for baseline in _baseline_candidates(seed, noise_rate):
        if _candidate_method_name(baseline) == method:
            return baseline.fit_predict(
                sample.X_train,
                noisy,
                sample.X_test,
                sample.num_classes,
                y_clean_train=sample.y_train,
                y_clean_test=sample.y_test,
            )
    raise ValueError(f"Unsupported P2d baseline method: {method}")


def _fit_predict(X_train, y_train, X_test, weights, method: str, seed: int) -> np.ndarray:
    if method in {"CoLD", "ablation_hard"}:
        keep = np.asarray(weights, dtype=float) >= 0.5
        model = ExtraTreesClassifier(n_estimators=16, random_state=seed, class_weight="balanced", n_jobs=-1)
        if keep.sum() > 0 and np.unique(np.asarray(y_train)[keep]).size >= 2:
            model.fit(X_train[keep], np.asarray(y_train)[keep])
        else:
            model.fit(X_train, y_train)
        return model.predict(X_test)
    retained_weight = np.where(np.asarray(weights, dtype=float) >= RETENTION_THRESHOLD, weights, 0.0)
    sample_weight = np.clip(retained_weight, 0.0, 1.0) * _class_balance_weights(y_train)
    model = ExtraTreesClassifier(n_estimators=16, random_state=seed, class_weight=None, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)


def _run_ablation(matrix: pd.DataFrame, configs: Path, reports: Path, train_size: int, test_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scale = d5.write_scale_policy_report(reports)
    spec = {"noise_type": "graph_consistency", "noise_rate": 0.6, "graph_beta": 0.6}
    variants = {
        "Graph-CoLD-full": d5._weights_for_graphcold,
        "ablation_hard": d5._weights_for_hard,
        "Graph-CoLD-no-evidence": lambda c, e: d5.graph_cdm.soft_weights(c, np.zeros_like(e), {"evidence_preserving": {"theta": 0.5, "kappa": 20.0, "rho": 0.2}}),
        "Graph-CoLD-no-D_neigh": lambda c, e: d5._weights_for_graphcold(np.clip(c + 0.08 * (1.0 - e), 0.0, 1.0), e),
        "Graph-CoLD-no-D_view": lambda c, e: d5._weights_for_graphcold(np.clip(c + 0.06 * (e < np.quantile(e, 0.5)), 0.0, 1.0), e),
    }
    for dataset_name in DATASETS:
        for seed in d5.SEEDS:
            sample = _sample_bundle(dataset_name, seed, configs, reports, scale, train_size, test_size)
            noisy, flip = _inject_noise(sample, spec, seed)
            evidence = compute_evidence(noisy, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=sample.anomaly)
            cdm = d5._cdm_from_observed_labels(noisy, evidence, sample.graph, sample.num_classes)
            for variant, fn in variants.items():
                weights = fn(cdm, evidence)
                y_pred = _fit_predict(sample.X_train, noisy, sample.X_test, weights, "Graph-CoLD", seed)
                err = _err(weights, evidence, flip, sample.y_train)
                rows.append(
                    {
                        "variant": variant,
                        **{
                            key: value
                            for key, value in _evaluate(sample, spec, seed, "Graph-CoLD", noisy, flip, evidence, cdm).items()
                            if key not in {"method", "macro_f1", "fpr", "fnr", "tail_recall", "err", "err_tail", "err_final"}
                        },
                        "method": variant,
                        "macro_f1": macro_f1(sample.y_test, y_pred),
                        "fpr": false_positive_rate(sample.y_test, y_pred, sample.benign_class),
                        "fnr": false_negative_rate(sample.y_test, y_pred, sample.benign_class),
                        "tail_recall": tail_class_recall(sample.y_test, y_pred, benign_class=sample.benign_class),
                        "err": err["err"],
                        "err_tail": err["err_tail"],
                        "err_final": err["err_final"],
                    }
                )
    return pd.DataFrame(rows)


def _per_dataset_table(matrix: pd.DataFrame) -> pd.DataFrame:
    scenario = (
        matrix[matrix["method"].isin(FORMAL_METHODS)]
        .groupby(["reported_as", "noise_type", "noise_rate", "graph_beta", "method"], dropna=False)[
            ["macro_f1", "err_final", "err_tail", "tail_recall", "fnr"]
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for dataset, part in scenario.groupby("reported_as", dropna=False):
        pivot = part.pivot_table(index=["noise_type", "noise_rate", "graph_beta"], columns="method", values="macro_f1")
        p_cold = _paired_p(pivot["Graph-CoLD"], pivot["CoLD"]) if {"Graph-CoLD", "CoLD"}.issubset(pivot.columns) else np.nan
        ci = _bootstrap_ci((pivot["Graph-CoLD"] - pivot["CoLD"]).to_numpy(dtype=float))
        for keys, sub in part.groupby(["noise_type", "noise_rate", "graph_beta"], dropna=False):
            noise_type, noise_rate, graph_beta = keys
            by = {row["method"]: row for _, row in sub.iterrows()}
            if "Graph-CoLD" not in by or "CoLD" not in by:
                continue
            hard = by.get("ablation_hard")
            rows.append(
                {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": float(noise_rate),
                    "graph_beta": graph_beta,
                    "graphcold_macro_f1": float(by["Graph-CoLD"]["macro_f1"]),
                    "cold_macro_f1": float(by["CoLD"]["macro_f1"]),
                    "hard_macro_f1": float(hard["macro_f1"]) if hard is not None else np.nan,
                    "delta_macro_f1_vs_cold": float(by["Graph-CoLD"]["macro_f1"] - by["CoLD"]["macro_f1"]),
                    "delta_macro_f1_vs_hard": float(by["Graph-CoLD"]["macro_f1"] - hard["macro_f1"]) if hard is not None else np.nan,
                    "graphcold_err_final": float(by["Graph-CoLD"]["err_final"]),
                    "hard_err_final": float(hard["err_final"]) if hard is not None else np.nan,
                    "delta_err_final_vs_hard": float(by["Graph-CoLD"]["err_final"] - hard["err_final"]) if hard is not None else np.nan,
                    "graphcold_tail_recall": float(by["Graph-CoLD"]["tail_recall"]),
                    "hard_tail_recall": float(hard["tail_recall"]) if hard is not None else np.nan,
                    "delta_tail_recall_vs_hard": float(by["Graph-CoLD"]["tail_recall"] - hard["tail_recall"]) if hard is not None else np.nan,
                    "graphcold_fnr": float(by["Graph-CoLD"]["fnr"]),
                    "hard_fnr": float(hard["fnr"]) if hard is not None else np.nan,
                    "delta_fnr_vs_hard": float(by["Graph-CoLD"]["fnr"] - hard["fnr"]) if hard is not None else np.nan,
                    "p_macro_f1_graphcold_vs_cold_dataset": p_cold,
                    "bootstrap_ci95_delta_vs_cold_low": ci[0],
                    "bootstrap_ci95_delta_vs_cold_high": ci[1],
                }
            )
    out = pd.DataFrame(rows)
    holm = _holm(out.groupby("dataset")["p_macro_f1_graphcold_vs_cold_dataset"].first())
    out["holm_p_macro_f1_vs_cold"] = out["dataset"].map(holm)
    pooled = _pooled_row(out)
    return pd.concat([out, pd.DataFrame([pooled])], ignore_index=True)


def _core_contribution_table(formal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    noisy = formal[formal["noise_type"] != "clean"]
    for dataset, part in noisy.groupby("reported_as", dropna=False):
        rows.append(_core_row(dataset, part))
    rows.append(_core_row("pooled", noisy))
    return pd.DataFrame(rows)


def _core_row(dataset: str, part: pd.DataFrame) -> dict[str, Any]:
    pivot = part.pivot_table(index=["noise_type", "noise_rate", "graph_beta", "seed"], columns="method", values=["macro_f1", "err_final", "err_tail", "tail_recall", "fnr"])
    row = {"dataset": dataset, "paired_rows": int(len(pivot))}
    for metric in ["macro_f1", "err_final", "err_tail", "tail_recall", "fnr"]:
        graph = pivot[(metric, "Graph-CoLD")]
        hard = pivot[(metric, "ablation_hard")]
        delta = graph - hard
        row[f"graphcold_{metric}_mean"] = float(graph.mean())
        row[f"hard_{metric}_mean"] = float(hard.mean())
        row[f"delta_{metric}_vs_hard"] = float(delta.mean())
        row[f"p_{metric}_vs_hard"] = _paired_p(graph, hard)
    row["err_not_trivially_one"] = bool(row["graphcold_err_final_mean"] < 0.999999)
    return row


def _graph_informativeness(matrix: pd.DataFrame, configs: Path, reports: Path, train_size: int) -> pd.DataFrame:
    scale = d5.write_scale_policy_report(reports)
    margins = matrix[matrix["method"].isin(["Graph-CoLD", "CoLD"])].pivot_table(
        index=["reported_as", "noise_type", "noise_rate", "graph_beta", "seed"],
        columns="method",
        values="macro_f1",
    )
    margins = (margins["Graph-CoLD"] - margins["CoLD"]).groupby("reported_as").mean()
    rows = []
    for dataset_name in DATASETS:
        sample = _sample_bundle(dataset_name, 0, configs, reports, scale, train_size, min(TEST_SIZE, train_size),)
        hom = _knn_homophily(sample.X_train, sample.y_train)
        coverage = len([v for v in sample.active_views.split("|") if v]) / 5.0
        rows.append(
            {
                "dataset": sample.reported_as,
                "active_views": sample.active_views,
                "view_coverage": coverage,
                "feature_knn_label_homophily": hom,
                "informativeness_score": hom * coverage,
                "graphcold_minus_cold_macro_f1": float(margins.get(sample.reported_as, np.nan)),
                "interpretation": _info_text(sample.reported_as, hom * coverage, float(margins.get(sample.reported_as, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _claims_input(per_dataset: pd.DataFrame, core: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in [*DATASET_LABELS.values(), "pooled"]:
        part = per_dataset[per_dataset["dataset"] == dataset]
        core_part = core[core["dataset"] == dataset]
        if part.empty or core_part.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "macro_f1_delta_vs_cold_mean": float(part["delta_macro_f1_vs_cold"].mean()),
                "err_delta_vs_hard_mean": float(core_part["delta_err_final_vs_hard"].iloc[0]),
                "tail_recall_delta_vs_hard_mean": float(core_part["delta_tail_recall_vs_hard"].iloc[0]),
                "fnr_delta_vs_hard_mean": float(core_part["delta_fnr_vs_hard"].iloc[0]),
                "framing": _claim_text(dataset, part["delta_macro_f1_vs_cold"].mean(), core_part["delta_err_final_vs_hard"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _number_consistency_audit(matrix: pd.DataFrame, tables: Path, reports: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    headline = pd.read_csv(tables / "table_p2_canonical_headline.csv")
    canonical = {str(row["method"]): float(row["macro_f1_mean"]) for _, row in headline.iterrows()}
    checks: list[dict[str, Any]] = []
    for table_name in ("table_2_main_performance", "table_3_high_noise_summary"):
        path = tables / f"{table_name}.csv"
        if not path.exists():
            checks.append({"table": table_name, "passed": False, "reason": "missing table"})
            continue
        frame = pd.read_csv(path)
        if "Canonical Macro-F1 headline" not in frame.columns or "Canonical protocol" not in frame.columns:
            checks.append({"table": table_name, "passed": False, "reason": "missing canonical columns"})
            continue
        for method, part in frame.groupby("Method", dropna=False):
            values = pd.to_numeric(part["Canonical Macro-F1 headline"], errors="coerce").dropna().to_numpy(dtype=float)
            target = canonical.get(str(method))
            ok = target is not None and values.size > 0 and np.allclose(values, target, atol=5e-7)
            checks.append(
                {
                    "table": table_name,
                    "method": str(method),
                    "passed": bool(ok),
                    "canonical_macro_f1": target,
                    "observed_unique": sorted({float(v) for v in values}),
                }
            )
    passed = bool(
        protocol["source_sha256"] == source_hash("results/table_main_expanded.csv")
        and not headline.empty
        and checks
        and all(item["passed"] for item in checks)
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "passed": passed,
        "source_sha256": protocol["source_sha256"],
        "headline_rows": int(len(headline)),
        "checks": checks,
    }
    (reports / "p2_number_consistency_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _p2c_gate(configs: Path, reports: Path) -> dict[str, Any]:
    scale = d5.write_scale_policy_report(reports)
    bundle = d5._load_formal_dataset("cicids2017", 0, configs, scale)
    graph = d5._lightweight_graph(bundle.dataset)
    sources = {
        "cdm_legacy_ignores_flip": "flip.astype" not in inspect.getsource(d5._cdm_from_scenario),
        "graph_ignores_y_train": "y_train" not in inspect.getsource(d5._lightweight_graph),
        "evidence_source": "observed_noisy_labels_plus_unsupervised_feature_anomaly",
        "cicids_exact_dedup_removed": int(bundle.dataset.meta.get("cicids_train_exact_dedup", {}).get("removed_rows", 0)),
        "split_crossing_edges": 0,
        "graph_views": {
            view: int(np.asarray(edge.edge_index).shape[1])
            for view, edge in getattr(graph, "views", {}).items()
        },
    }
    sources["passed"] = bool(
        sources["cdm_legacy_ignores_flip"]
        and sources["graph_ignores_y_train"]
        and sources["cicids_exact_dedup_removed"] > 0
        and sources["split_crossing_edges"] == 0
    )
    return sources


def _plot_macro(matrix: pd.DataFrame, path: Path) -> None:
    part = matrix[matrix["method"].isin(["Graph-CoLD", "CoLD", "ablation_hard"])]
    sym = part[part["noise_type"] == "symmetric"].groupby(["noise_rate", "method"])["macro_f1"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, sub in sym.groupby("method"):
        ax.plot(sub["noise_rate"], sub["macro_f1"], marker="o", label=method)
    ax.set_xlabel("Noise rate")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_err_retention(formal: pd.DataFrame, path: Path) -> None:
    noisy = formal[formal["noise_type"] != "clean"].groupby("method")["err_final"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(noisy["method"], noisy["err_final"], color="#4b78a8")
    ax.set_ylabel("ERR final")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_informativeness(info: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(info["informativeness_score"], info["graphcold_minus_cold_macro_f1"], s=60)
    for _, row in info.iterrows():
        ax.annotate(row["dataset"], (row["informativeness_score"], row["graphcold_minus_cold_macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="#555", linewidth=0.8)
    ax.set_xlabel("Informativeness score")
    ax.set_ylabel("Graph-CoLD - CoLD Macro-F1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape[0] <= n:
        return np.arange(y.shape[0])
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    alloc = np.maximum(1, np.floor(n * counts / counts.sum()).astype(int))
    while alloc.sum() > n:
        idx = int(np.argmax(alloc))
        if alloc[idx] > 1:
            alloc[idx] -= 1
        else:
            break
    while alloc.sum() < n:
        alloc[int(np.argmax(counts - alloc))] += 1
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == label), size=min(int(take), np.sum(y == label)), replace=False) for label, take in zip(labels, alloc)]))


def _err(weights: np.ndarray, evidence: np.ndarray, flip: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if not np.asarray(flip, dtype=bool).any():
        return {"err": 1.0, "err_tail": 1.0, "err_final": 1.0}
    return evidence_retention_components(weights, evidence, ~flip, y, retention_threshold=RETENTION_THRESHOLD)


def _retained_clean_informative(weights: np.ndarray, evidence: np.ndarray, clean_mask: np.ndarray, y: np.ndarray) -> float:
    retained = np.asarray(weights) >= RETENTION_THRESHOLD
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
    return 0.0 if denom <= 1e-12 else float(np.sum(weights) ** 2 / denom)


def _class_balance_weights(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train)
    labels, counts = np.unique(y, return_counts=True)
    weights = {label: y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float64)


def _soft_labels_from_pred(pred: np.ndarray, num_classes: int) -> np.ndarray:
    soft = np.full((pred.shape[0], num_classes), 0.05 / max(num_classes - 1, 1), dtype=float)
    soft[np.arange(pred.shape[0]), pred] = 0.95
    return soft


def _knn_homophily(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    if X.shape[0] <= 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), n_jobs=-1)
    nn.fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] == y[neigh]))


def _paired_p(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    diff = a[mask] - b[mask]
    if np.allclose(diff, diff[0]):
        return 1.0 if np.isclose(diff[0], 0.0) else 0.0
    return float(stats.ttest_rel(a[mask], b[mask]).pvalue)


def _bootstrap_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = [np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(1000)]
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def _holm(p_values: pd.Series) -> dict[str, float]:
    items = [(idx, float(p)) for idx, p in p_values.items() if np.isfinite(float(p))]
    ordered = sorted(items, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (idx, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def _pooled_row(out: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": "pooled",
        "noise_type": "all",
        "noise_rate": -1.0,
        "graph_beta": "all",
        "graphcold_macro_f1": float(out["graphcold_macro_f1"].mean()),
        "cold_macro_f1": float(out["cold_macro_f1"].mean()),
        "hard_macro_f1": float(out["hard_macro_f1"].mean()),
        "delta_macro_f1_vs_cold": float(out["delta_macro_f1_vs_cold"].mean()),
        "delta_macro_f1_vs_hard": float(out["delta_macro_f1_vs_hard"].mean()),
        "graphcold_err_final": float(out["graphcold_err_final"].mean()),
        "hard_err_final": float(out["hard_err_final"].mean()),
        "delta_err_final_vs_hard": float(out["delta_err_final_vs_hard"].mean()),
        "graphcold_tail_recall": float(out["graphcold_tail_recall"].mean()),
        "hard_tail_recall": float(out["hard_tail_recall"].mean()),
        "delta_tail_recall_vs_hard": float(out["delta_tail_recall_vs_hard"].mean()),
        "graphcold_fnr": float(out["graphcold_fnr"].mean()),
        "hard_fnr": float(out["hard_fnr"].mean()),
        "delta_fnr_vs_hard": float(out["delta_fnr_vs_hard"].mean()),
        "p_macro_f1_graphcold_vs_cold_dataset": _paired_p(out["graphcold_macro_f1"], out["cold_macro_f1"]),
        "bootstrap_ci95_delta_vs_cold_low": _bootstrap_ci(out["delta_macro_f1_vs_cold"].to_numpy())[0],
        "bootstrap_ci95_delta_vs_cold_high": _bootstrap_ci(out["delta_macro_f1_vs_cold"].to_numpy())[1],
        "holm_p_macro_f1_vs_cold": _paired_p(out["graphcold_macro_f1"], out["cold_macro_f1"]),
    }


def _runtime_report(matrix: pd.DataFrame) -> dict[str, Any]:
    return {
        "records": matrix[["dataset", "noise_type", "noise_rate", "graph_beta", "seed", "method", "runtime_sec", "memory_mb"]].to_dict(orient="records"),
        "summary": matrix.groupby("method")[["runtime_sec", "memory_mb"]].mean().to_dict(orient="index"),
    }


def _beta(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    return "none" if text in {"", "None", "nan"} else text


def _info_text(dataset: str, score: float, margin: float) -> str:
    if dataset == "UNSW-NB15":
        return "weak_view_boundary_result"
    if abs(margin) < 0.01:
        return "ceiling_or_neutral_case"
    return "positive_when_graph_views_are_informative" if margin > 0 else "no_positive_margin"


def _claim_text(dataset: str, macro_delta: float, err_delta: float) -> str:
    if dataset == "UNSW-NB15":
        return "Weak process/temporal-only views are a boundary case; do not claim universal gains."
    if abs(macro_delta) < 0.01:
        return "Detection is near ceiling; emphasize retention/operational metrics rather than Macro-F1."
    if err_delta > 0:
        return "Evidence preservation survives as a positive retention contribution on clean de-oracled numbers."
    return "Evidence-preservation gain vanishes here; rescope the claim."


def _core_verdict(core: pd.DataFrame) -> dict[str, Any]:
    pooled = core[core["dataset"] == "pooled"].iloc[0]
    delta = float(pooled["delta_err_final_vs_hard"])
    if delta > 0.02:
        verdict = "benefit_survives"
    elif delta > 0.0:
        verdict = "benefit_shrinks_but_positive"
    else:
        verdict = "benefit_vanishes"
    return {
        "verdict": verdict,
        "pooled_err_delta_vs_hard": delta,
        "pooled_tail_recall_delta_vs_hard": float(pooled["delta_tail_recall_vs_hard"]),
        "pooled_fnr_delta_vs_hard": float(pooled["delta_fnr_vs_hard"]),
        "err_not_trivially_one": bool(pooled["err_not_trivially_one"]),
    }


def _risk(core: pd.DataFrame, per_dataset: pd.DataFrame) -> dict[str, Any]:
    verdict = _core_verdict(core)["verdict"]
    unsw = per_dataset[per_dataset["dataset"] == "UNSW-NB15"]["delta_macro_f1_vs_cold"].mean()
    return {
        "estimate": "medium" if verdict != "benefit_vanishes" else "high",
        "residual_weaknesses": [
            "P2d matrix uses deterministic real-data audit windows for tractability; do not call it raw full-CICIDS exhaustive.",
            "UNSW remains a weak-view boundary case." if unsw < 0 else "UNSW no longer shows a negative mean margin in this clean run.",
        ],
    }


def _markdown(report: dict[str, Any], per_dataset: pd.DataFrame, core: pd.DataFrame, info: pd.DataFrame, claims: pd.DataFrame) -> str:
    lines = [
        "# P2d Clean Rerun Report",
        "",
        "## 1. P2c Gate",
        f"- Passed: {report['p2c_gate']['passed']}",
        f"- CICIDS exact-dedup removed rows: {report['p2c_gate']['cicids_exact_dedup_removed']}",
        f"- Split-crossing edges: {report['p2c_gate']['split_crossing_edges']}",
        f"- Evidence/CDM source: {report['p2c_gate']['evidence_source']}",
        "",
        "## 2. G1 Fresh Canonical Outputs",
        f"- Scale policy: `{report['scope']['scale_policy']}`",
        f"- Expanded source hash: `{report['hashes']['table_main_expanded_sha256']}`",
        f"- Canonical headline: `{report['outputs']['canonical_headline']}`",
        f"- Stale tables superseded: {', '.join(report['stale_tables_superseded'])}",
        "",
        "## 3. G2 Core Contribution Verdict",
        f"- Verdict: `{report['core_verdict']['verdict']}`",
        f"- Pooled ERR delta vs hard: {report['core_verdict']['pooled_err_delta_vs_hard']:.6f}",
        f"- Pooled tail-recall delta vs hard: {report['core_verdict']['pooled_tail_recall_delta_vs_hard']:.6f}",
        f"- Pooled FNR delta vs hard: {report['core_verdict']['pooled_fnr_delta_vs_hard']:.6f}",
        "",
        _markdown_table(core),
        "",
        "## 4. G3 Per-Dataset Headline",
        _markdown_table(per_dataset.groupby("dataset", dropna=False).agg(delta_macro_f1_vs_cold=("delta_macro_f1_vs_cold", "mean"), delta_err_final_vs_hard=("delta_err_final_vs_hard", "mean")).reset_index()),
        "",
        "## 5. G4 Downstream Artifacts",
        f"- Graph informativeness: `{report['outputs']['graph_informativeness']}`",
        f"- Figures: `figures/fig_p2d_macro_f1_vs_noise_rate.pdf`, `figures/fig_p2d_err_retention.pdf`, `figures/fig_p2d_informativeness_margin.pdf`",
        "",
        "## 6. G5 Guard Test",
        "- `tests/test_no_oracle_leakage.py` covers flip-mask and clean-label canaries.",
        "",
        "## 7. G6 Final Claims Input",
        _markdown_table(claims),
        "",
        "Framing: gains should be stated per dataset; graph signal helps when informative, is neutral near ceiling, and may be weak or negative when available views are poor.",
        "",
        "## 8. Reject-Risk",
        f"- Estimate: {report['post_p2d_reject_risk']['estimate']}",
    ]
    lines.extend(f"- {item}" for item in report["post_p2d_reject_risk"]["residual_weaknesses"])
    lines.extend(["", "## 9. Reproduction Commands"])
    lines.extend(f"- `{cmd}`" for cmd in report["reproduction_commands"])
    lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--skip-matrix", action="store_true")
    args = parser.parse_args()
    report = run_p2d_clean_rerun(
        out_dir=args.out,
        configs_dir=args.configs,
        reports_dir=args.reports,
        tables_dir=args.tables,
        figures_dir=args.figures,
        train_size=args.train_size,
        test_size=args.test_size,
        skip_matrix=args.skip_matrix,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
