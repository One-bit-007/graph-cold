"""Direct SOC alert-prioritization evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis.stat_tests import grouped_paired_summary
from src.baselines.confident_learning import ConfidentLearningBaseline
from src.data.loaders import load_dataset
from src.experiments import cicids_mini_matrix, d5
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import priority_scores, queue_load_curve, ranking_metrics


METHODS = ("Graph-CoLD", "CoLD", "ablation_hard", "Confident-Learning")
BUDGETS = tuple(np.linspace(0.02, 1.0, 50))
PRIORITIZATION_SAMPLE_ROWS = 100_000
PRIORITIZATION_SPECS = (
    {"noise_type": "clean", "noise_rate": 0.0, "graph_beta": "none"},
    {"noise_type": "symmetric", "noise_rate": 0.4, "graph_beta": "none"},
    {"noise_type": "asymmetric", "noise_rate": 0.4, "graph_beta": "none"},
    {"noise_type": "graph_consistency", "noise_rate": 0.4, "graph_beta": 0.6},
)


def run_prioritization_evaluation(
    out_dir: str | Path = "results",
    configs_dir: str | Path = "configs",
    reports_dir: str | Path = "reports",
    figure_path: str | Path = "figures/fig_p1_queue_load_curve.pdf",
) -> dict[str, Any]:
    out = Path(out_dir)
    reports = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)

    configs = Path(configs_dir)
    dataset_scope = d5._readiness_guard(configs)
    scale_policy = d5.write_scale_policy_report(reports)

    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for dataset_name in dataset_scope:
        for seed in d5.SEEDS:
            bundle = _load_priority_bundle(dataset_name, seed, configs, scale_policy)
            evidence = _evidence(bundle)
            graph_cache: dict[float, Any] = {}
            for spec in PRIORITIZATION_SPECS:
                noisy, flip = d5._inject_noise(bundle.dataset, spec, seed, graph_cache)
                context = d5._graphcold_context(bundle, spec, seed, flip, evidence, graph_cache)
                for method in METHODS:
                    y_pred, scores = _predict_and_score(method, bundle, spec, seed, noisy, flip, evidence, context)
                    metrics = ranking_metrics(
                        scores,
                        bundle.dataset.y_test,
                        {
                            "ranking": {
                                "top_k": min(100, bundle.dataset.y_test.shape[0]),
                                "review_budget": 0.1,
                                "benign_class": bundle.dataset.meta.get("benign_class", 0) or 0,
                                "recall_targets": (0.90, 0.95),
                            }
                        },
                    )
                    metric_rows.append(
                        {
                            "dataset": bundle.dataset_key,
                            "reported_as": bundle.reported_as,
                            "noise_type": spec["noise_type"],
                            "noise_rate": float(spec["noise_rate"]),
                            "graph_beta": spec["graph_beta"],
                            "seed": int(seed),
                            "method": method,
                            **metrics,
                        }
                    )
                    for point in queue_load_curve(
                        scores,
                        bundle.dataset.y_test,
                        budgets=BUDGETS,
                        benign_class=bundle.dataset.meta.get("benign_class", 0) or 0,
                    ):
                        curve_rows.append(
                            {
                                "dataset": bundle.dataset_key,
                                "reported_as": bundle.reported_as,
                                "noise_type": spec["noise_type"],
                                "noise_rate": float(spec["noise_rate"]),
                                "graph_beta": spec["graph_beta"],
                                "seed": int(seed),
                                "method": method,
                                **point,
                            }
                        )

    metrics_frame = pd.DataFrame(metric_rows)
    curve_frame = pd.DataFrame(curve_rows)
    metrics_path = out / "table_prioritization.csv"
    curve_path = out / "prioritization_curve.csv"
    stats_path = out / "stat_tests_prioritization.json"
    metrics_frame.to_csv(metrics_path, index=False)
    curve_frame.to_csv(curve_path, index=False)
    stats = grouped_paired_summary(metrics_frame, metric="topk_precision")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _plot_queue_curve(curve_frame, figure_path)
    report = {
        "completed": True,
        "methods": list(METHODS),
        "scenario_scope": "representative paired prioritization scenarios on deterministic real-data audit windows",
        "sample_rows_per_dataset": PRIORITIZATION_SAMPLE_ROWS,
        "specs": list(PRIORITIZATION_SPECS),
        "rows": {"prioritization": int(len(metrics_frame)), "curve": int(len(curve_frame))},
        "outputs": {
            "table": str(metrics_path),
            "curve": str(curve_path),
            "stats": str(stats_path),
            "figure": str(figure_path),
        },
        "runtime_sec": float(time.perf_counter() - start),
        "key_metrics": _key_metrics(metrics_frame),
        "topk_precision_stats": stats,
    }
    (reports / "p1_prioritization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "p1_prioritization_report.md").write_text(_report_md(report), encoding="utf-8")
    return report


def _predict_and_score(method: str, bundle: d5.FormalBundle, spec: dict[str, Any], seed: int, noisy, flip, evidence, context):
    if method in {"Graph-CoLD", "CoLD", "ablation_hard"}:
        plan = d5._execution_plan_for_method(method, context)
        y_pred = cicids_mini_matrix._fit_predict(
            plan.representation,
            noisy,
            bundle.dataset.X_test,
            plan.weights,
            plan.fit_method,
            seed,
        )
        soft = cicids_mini_matrix._soft_labels_from_pred(y_pred, bundle.dataset.num_classes)
        scores = priority_scores(
            {
                "graph_cdm": np.resize(plan.cdm, bundle.dataset.y_test.shape[0]),
                "evidence": np.resize(plan.evidence, bundle.dataset.y_test.shape[0]),
                "soft_labels": soft,
            },
            {},
            {"ranking": {"alpha1": 1.0, "alpha2": 0.7, "alpha3": 0.4, "benign_class": bundle.dataset.meta.get("benign_class", 0) or 0}},
        )
        return y_pred, scores

    if method == "Confident-Learning":
        baseline = ConfidentLearningBaseline(seed=seed, noise_rate=float(spec["noise_rate"]))
        result = baseline.fit_predict(
            bundle.dataset.X_train,
            noisy,
            bundle.dataset.X_test,
            bundle.dataset.num_classes,
            y_clean_train=bundle.dataset.y_train,
            y_clean_test=bundle.dataset.y_test,
        )
        cdm = d5._cdm_from_scenario(flip, evidence)
        scores = priority_scores(
            {
                "graph_cdm": np.resize(cdm, bundle.dataset.y_test.shape[0]),
                "evidence": np.resize(evidence, bundle.dataset.y_test.shape[0]),
                "soft_labels": result.proba,
            },
            {},
            {"ranking": {"alpha1": 1.0, "alpha2": 0.0, "alpha3": 0.0, "benign_class": bundle.dataset.meta.get("benign_class", 0) or 0}},
        )
        return result.y_pred, scores
    raise ValueError(f"Unsupported prioritization method: {method}")


def _load_priority_bundle(dataset_name: str, seed: int, configs_dir: Path, scale_policy: dict[str, Any]) -> d5.FormalBundle:
    cfg = yaml.safe_load((configs_dir / "datasets.yaml").read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    if dataset_name == "cesnet_tls_year22":
        audit = d5._read_json(configs_dir.parent / "reports" / "cesnet_audit_report.json")
        cfg[dataset_name]["path"] = audit.get("actual_data_path", cfg[dataset_name]["path"])
        cfg[dataset_name]["dataset_hash"] = audit.get("dataset_hash")
        cfg[dataset_name]["reported_as"] = "CESNET-TLS-Year22"
        cfg[dataset_name]["replacement_for"] = "MALTLS-22"
        cfg[dataset_name]["source_verified"] = True
    elif dataset_name == "cicids2017":
        protocol = d5._read_json(configs_dir.parent / "reports" / "cicids_final_protocol.json")
        cfg[dataset_name]["dataset_hash"] = protocol.get("dataset_hash")
        cfg[dataset_name]["reported_as"] = "CICIDS-2017"
        cfg[dataset_name]["source_verified"] = True
    cfg[dataset_name]["sample_rows"] = PRIORITIZATION_SAMPLE_ROWS
    dataset = load_dataset(dataset_name, cfg)
    meta = dataset.meta
    replacement = meta.get("replacement_for") or ""
    if str(replacement).lower() == "maltls22":
        replacement = "MALTLS-22"
    return d5.FormalBundle(
        dataset=dataset,
        dataset_key=dataset_name,
        reported_as=str(meta.get("reported_as", scale_policy["datasets"][dataset_name]["reported_as"])),
        dataset_hash=str(meta.get("dataset_hash") or d5._dataset_hash_from_reports(dataset_name, configs_dir.parent / "reports")),
        actual_data_path=str(Path(meta.get("data_source", "")).resolve()),
        class_policy=str(meta.get("class_policy", scale_policy["datasets"][dataset_name]["class_policy"])),
        sample_policy=f"deterministic_prioritization_audit_window_{PRIORITIZATION_SAMPLE_ROWS}",
        sample_seed=42,
        sampling_stratified=True,
        active_views="|".join(meta.get("active_views", [])),
        source_verified=bool(meta.get("source_verified", True)),
        replacement_for=str(replacement),
    )


def _evidence(bundle: d5.FormalBundle) -> np.ndarray:
    anomaly = cicids_mini_matrix.smoke_realdata._feature_anomaly(bundle.dataset.X_train, bundle.dataset.y_train)
    return compute_evidence(
        bundle.dataset.y_train,
        {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
        anomaly=anomaly,
    )


def _plot_queue_curve(curve: pd.DataFrame, figure_path: str | Path) -> None:
    grouped = curve.groupby(["method", "review_budget"], dropna=False)["topk_recall"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method, part in grouped.groupby("method", sort=True):
        ax.plot(part["review_budget"], part["topk_recall"], label=method, linewidth=2)
    ax.set_xlabel("Review budget (fraction of alert queue)")
    ax.set_ylabel("Recall of true malicious alerts")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def _key_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    summary = frame.groupby("method")[[
        "topk_precision",
        "topk_recall",
        "precision_at_budget",
        "compression_at_recall_90",
        "compression_at_recall_95",
    ]].mean()
    return {method: {key: float(value) for key, value in row.items()} for method, row in summary.to_dict(orient="index").items()}


def _report_md(report: dict[str, Any]) -> str:
    lines = [
        "# P1 Prioritization Report",
        "",
        f"- Completed: {report['completed']}",
        f"- Methods: {', '.join(report['methods'])}",
        f"- Prioritization rows: {report['rows']['prioritization']}",
        f"- Curve rows: {report['rows']['curve']}",
        f"- Figure: `{report['outputs']['figure']}`",
        "",
        "## Mean Metrics",
    ]
    for method, metrics in report["key_metrics"].items():
        lines.append(
            f"- {method}: Top-K precision={metrics['topk_precision']:.6f}, "
            f"Top-K recall={metrics['topk_recall']:.6f}, "
            f"compression@90={metrics['compression_at_recall_90']:.6f}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--figure", default="figures/fig_p1_queue_load_curve.pdf")
    args = parser.parse_args()
    print(json.dumps(run_prioritization_evaluation(args.out, args.configs, args.reports, args.figure), indent=2))


if __name__ == "__main__":
    main()
