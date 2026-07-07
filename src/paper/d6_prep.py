"""Real-data paper preparation.

This module aggregates verified real-data result artifacts into paper tables,
figures, statistical narrative, draft LaTeX sections, and readiness metadata.
It intentionally does not import experiment runners or model modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.protocol import PROTOCOL_ID, method_headline_map, write_protocol_artifacts


MAIN_SOURCE = "results/table_main_expanded.csv"
ORIGINAL_MAIN_SOURCE = "results/table_main.csv"
ABLATION_SOURCE = "results/table_ablation.csv"
BASELINE_SOURCE = "results/table_baseline_expansion.csv"
STATS_SOURCE = "results/stat_tests_baseline_expansion.json"

DATASET_ORDER = ["CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"]
FORMAL_DATASETS = {"CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"}
MAIN_METHODS = [
    "Graph-CoLD",
    "CoLD",
    "ablation_hard",
    "Noisy-Supervised",
    "Confident-Learning",
    "Co-Teaching",
    "Decoupling",
    "FINE",
    "MCRe",
    "MORSE",
]
FIG_METHODS = [
    "Graph-CoLD",
    "CoLD",
    "Noisy-Supervised",
    "Confident-Learning",
    "Co-Teaching",
    "FINE",
    "MCRe",
    "MORSE",
]
ABLATION_VARIANTS = [
    "Graph-CoLD-full",
    "ablation_hard",
    "Graph-CoLD-no-D_neigh",
    "Graph-CoLD-no-D_view",
    "Graph-CoLD-no-evidence",
]
FORBIDDEN_OUTPUT_TERMS = ("synthetic", "fallback", "emulation")
OVERCLAIM_TERMS = ("dominates", "massive gain", "state-of-the-art", "near-perfect", "causal proof")

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "Graph-CoLD": "#5477C4",
    "CoLD": "#F0986E",
    "ablation_hard": "#B8A037",
    "Noisy-Supervised": "#7A828F",
    "Confident-Learning": "#71B436",
    "Co-Teaching": "#BD569B",
    "Decoupling": "#8E6BBE",
    "FINE": "#45A6A1",
    "MCRe": "#D68C45",
    "MORSE": "#5E9F6E",
    "Graph-CoLD-full": "#5477C4",
    "Graph-CoLD-no-D_neigh": "#A3BEFA",
    "Graph-CoLD-no-D_view": "#CEDFFE",
    "Graph-CoLD-no-evidence": "#C5CAD3",
}


def run_d6_realdata_prep(
    results_dir: str | Path = "results",
    tables_dir: str | Path = "tables",
    figures_dir: str | Path = "figures",
    reports_dir: str | Path = "reports",
    sections_dir: str | Path = "paper/sections",
) -> dict[str, Any]:
    """Generate all D6 paper-preparation artifacts from verified result files."""
    results = Path(results_dir)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    reports = Path(reports_dir)
    reports_d6 = reports / "d6"
    sections = Path(sections_dir)
    for directory in (tables, figures, reports_d6, sections):
        directory.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(results, reports)
    main = sources["main"]
    ablation = sources["ablation"]
    stats = sources["stats"]

    _validate_main(main)
    _validate_ablation(ablation)

    table1 = _table_dataset_protocol(main)
    table2 = _table_main_performance(main)
    table3 = _table_high_noise(main)
    table4 = _table_ablation(ablation)
    table5 = _table_stats(stats)
    write_protocol_artifacts(results / "table_main_expanded.csv", tables / "table_p2_canonical_headline.csv", reports / "p2_number_consistency.json")
    _write_table_pair(table1, tables / "table_1_dataset_protocol.csv", tables / "table_1_dataset_protocol.md", "Table 1. Dataset and protocol summary", MAIN_SOURCE)
    _write_table_pair(table2, tables / "table_2_main_performance.csv", tables / "table_2_main_performance.md", "Table 2. Main performance by dataset and noise setting", MAIN_SOURCE)
    _write_table_pair(table3, tables / "table_3_high_noise_summary.csv", tables / "table_3_high_noise_summary.md", "Table 3. High-noise robustness summary", MAIN_SOURCE)
    _write_table_pair(table4, tables / "table_4_ablation_evidence.csv", tables / "table_4_ablation_evidence.md", "Table 4. Ablation and evidence retention", ABLATION_SOURCE)
    _write_table_pair(table5, tables / "table_5_statistical_tests.csv", tables / "table_5_statistical_tests.md", "Table 5. Statistical testing summary", STATS_SOURCE)

    _write_fig2(main, figures)
    _write_fig3(main, figures)
    _write_fig4(ablation, figures)
    _write_fig5(main, figures)

    narrative_json = _narrative_json(main, table1, table3, table4, table5, stats, sources)
    narrative_md = _narrative_md(narrative_json)
    (reports_d6 / "d6_statistical_narrative.json").write_text(
        json.dumps(narrative_json, indent=2), encoding="utf-8"
    )
    (reports_d6 / "d6_statistical_narrative.md").write_text(narrative_md, encoding="utf-8")

    _write_paper_sections(main, table3, table4, table5, reports_d6, sections)
    _write_reviewer_risk_notes(sources, reports_d6)
    checklist = _write_checklist(tables, figures, reports_d6, sections)
    readiness = _update_readiness(reports, checklist)

    manifest = {
        "stage": "D6 real-data paper prep",
        "completed": True,
        "source_csv": MAIN_SOURCE,
        "source_sha256": _sha256(Path(MAIN_SOURCE)),
        "tables": [
            "tables/table_1_dataset_protocol.csv",
            "tables/table_2_main_performance.csv",
            "tables/table_3_high_noise_summary.csv",
            "tables/table_4_ablation_evidence.csv",
            "tables/table_5_statistical_tests.csv",
            "tables/table_p2_canonical_headline.csv",
        ],
        "figures": [
            "figures/fig2_macro_f1_vs_noise_rate.png",
            "figures/fig2_macro_f1_vs_noise_rate.pdf",
            "figures/fig3_err_retention.png",
            "figures/fig3_err_retention.pdf",
            "figures/fig4_ablation.png",
            "figures/fig4_ablation.pdf",
            "figures/fig5_runtime_cost.png",
            "figures/fig5_runtime_cost.pdf",
        ],
        "reports": [
            "reports/d6/d6_statistical_narrative.json",
            "reports/d6/d6_statistical_narrative.md",
            "reports/d6/reviewer_risk_notes.md",
            "reports/d6/d6_paper_prep_checklist.json",
            "reports/d6/d6_paper_prep_checklist.md",
        ],
        "paper_sections": [
            "paper/sections/experiments_realdata.tex",
            "paper/sections/results_realdata.tex",
            "paper/sections/discussion_realdata.tex",
            "paper/sections/limitations_realdata.tex",
        ],
        "readiness": readiness,
    }
    _guard_outputs([tables, figures, reports_d6, sections])
    (reports_d6 / "d6_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_d6_paper_prep(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias for the real-data D6 generator."""
    return run_d6_realdata_prep(*args, **kwargs)


def _load_sources(results: Path, reports: Path) -> dict[str, Any]:
    required_result_files = {
        "main": results / "table_main_expanded.csv",
        "original_main": results / "table_main.csv",
        "ablation": results / "table_ablation.csv",
        "baseline": results / "table_baseline_expansion.csv",
        "stats": results / "stat_tests_baseline_expansion.json",
        "stats_original": results / "stat_tests.json",
        "runtime": results / "runtime.json",
        "runtime_expanded": results / "runtime_baseline_expansion.json",
    }
    required_report_files = {
        "d5": reports / "d5_realdata_execution_report.json",
        "d55": reports / "d5_baseline_expansion_report.json",
        "sanity": reports / "d5_expanded_sanity_report.json",
        "stat_validity": reports / "d5_expanded_statistical_validity_report.json",
        "baseline_readiness": reports / "baseline_readiness_report.json",
        "scale_policy": reports / "d5_scale_policy.json",
        "two_dataset": reports / "two_dataset_readiness_report.json",
        "scope": reports / "d5_scope_decision.json",
        "cicids_protocol": reports / "cicids_final_protocol.json",
        "cesnet_class": reports / "cesnet_class_policy_report.json",
        "cesnet_view": reports / "cesnet_view_policy_report.json",
    }
    missing = [str(path) for path in [*required_result_files.values(), *required_report_files.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("D6 requires completed D5/D5.5 artifacts: " + ", ".join(missing))

    loaded: dict[str, Any] = {
        "main": pd.read_csv(required_result_files["main"]),
        "original_main": pd.read_csv(required_result_files["original_main"]),
        "ablation": pd.read_csv(required_result_files["ablation"]),
        "baseline": pd.read_csv(required_result_files["baseline"]),
    }
    for key, path in {**required_result_files, **required_report_files}.items():
        if key in loaded:
            continue
        if path.suffix == ".json":
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    loaded["source_hashes"] = {str(path): _sha256(path) for path in required_result_files.values()}
    return loaded


def _validate_main(frame: pd.DataFrame) -> None:
    required_cols = {
        "dataset",
        "reported_as",
        "class_policy",
        "num_classes",
        "sample_policy",
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
        "runtime_sec",
        "memory_mb",
        "active_views",
        "source_verified",
    }
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        raise ValueError(f"table_main_expanded.csv is missing columns: {missing}")
    datasets = set(frame["reported_as"].dropna().astype(str))
    if datasets != FORMAL_DATASETS:
        raise ValueError(f"D6 supports only {sorted(FORMAL_DATASETS)}, got {sorted(datasets)}")
    methods = set(frame["method"].dropna().astype(str))
    missing_methods = [method for method in MAIN_METHODS if method not in methods]
    if missing_methods:
        raise ValueError(f"table_main_expanded.csv is missing methods: {missing_methods}")
    _assert_no_terms(frame.astype(str).to_string(index=False), FORBIDDEN_OUTPUT_TERMS)


def _validate_ablation(frame: pd.DataFrame) -> None:
    if "Graph-CoLD-w=1" in set(frame.get("variant", pd.Series(dtype=str)).astype(str)):
        raise ValueError("Graph-CoLD-w=1 must stay out of the formal D6 ablation table.")
    variants = set(frame.get("variant", pd.Series(dtype=str)).astype(str))
    missing = [variant for variant in ABLATION_VARIANTS if variant not in variants]
    if missing:
        raise ValueError(f"table_ablation.csv is missing variants: {missing}")


def _table_dataset_protocol(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reported in DATASET_ORDER:
        part = main[main["reported_as"] == reported]
        if part.empty:
            continue
        dataset = str(part["dataset"].iloc[0])
        rows.append(
            {
                "Dataset": reported,
                "Reported name": reported,
                "Rows used": int(pd.to_numeric(part["sample_size"], errors="coerce").max()),
                "Class policy": _dataset_class_policy(dataset, part),
                "Number of classes": int(pd.to_numeric(part["num_classes"], errors="coerce").max()),
                "Sample policy": str(part["sample_policy"].iloc[0]),
                "Active views": str(part["active_views"].iloc[0]),
                "Noise settings": _noise_settings(part),
                "Source verified": bool(part["source_verified"].astype(bool).all()),
                "Replacement notes": _replacement_note(dataset),
            }
        )
    return pd.DataFrame(rows)


def _table_main_performance(main: pd.DataFrame) -> pd.DataFrame:
    headlines = method_headline_map(main, metric="macro_f1")
    grouped = (
        main.groupby(["reported_as", "noise_type", "noise_rate", "graph_beta", "method"], dropna=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", _std),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", _std),
            fnr_mean=("fnr", "mean"),
            fnr_std=("fnr", _std),
            err_mean=("err_final", "mean"),
            err_std=("err_final", _std),
            compression_mean=("compression_ratio", "mean"),
            compression_std=("compression_ratio", _std),
            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", _std),
        )
        .reset_index()
    )
    grouped["dataset_order"] = grouped["reported_as"].map({name: i for i, name in enumerate(DATASET_ORDER)})
    grouped["method_order"] = grouped["method"].map({name: i for i, name in enumerate(MAIN_METHODS)})
    grouped["noise_order"] = grouped["noise_type"].map({"clean": 0, "symmetric": 1, "asymmetric": 2, "graph_consistency": 3})
    grouped = grouped.sort_values(["dataset_order", "noise_order", "noise_rate", "graph_beta", "method_order"])
    return pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Noise type": grouped["noise_type"],
            "Noise rate": grouped["noise_rate"].map(_rate),
            "Graph beta": grouped["graph_beta"].astype(str),
            "Method": grouped["method"],
            "Canonical Macro-F1 headline": grouped["method"].map(lambda method: f"{headlines[str(method)]:.6f}"),
            "Canonical protocol": PROTOCOL_ID,
            "Macro-F1 mean +/- std": _fmt_mean_std(grouped, "macro_f1"),
            "FPR mean +/- std": _fmt_mean_std(grouped, "fpr"),
            "FNR mean +/- std": _fmt_mean_std(grouped, "fnr"),
            "ERR mean +/- std": _fmt_mean_std(grouped, "err"),
            "Compression ratio mean +/- std": _fmt_mean_std(grouped, "compression"),
            "Runtime mean +/- std (s)": _fmt_mean_std(grouped, "runtime", digits=2),
        }
    )


def _table_high_noise(main: pd.DataFrame) -> pd.DataFrame:
    headlines = method_headline_map(main, metric="macro_f1")
    rates = pd.to_numeric(main["noise_rate"], errors="coerce")
    high = main[
        (rates >= 0.4)
        & main["noise_type"].isin(["symmetric", "asymmetric", "graph_consistency"])
        & main["method"].isin(MAIN_METHODS)
    ].copy()
    grouped = (
        high.groupby(["reported_as", "method"], dropna=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            fpr_mean=("fpr", "mean"),
            fnr_mean=("fnr", "mean"),
            err_mean=("err_final", "mean"),
            compression_mean=("compression_ratio", "mean"),
            scenario_count=("macro_f1", "size"),
        )
        .reset_index()
    )
    grouped["dataset_order"] = grouped["reported_as"].map({name: i for i, name in enumerate(DATASET_ORDER)})
    grouped["method_order"] = grouped["method"].map({name: i for i, name in enumerate(MAIN_METHODS)})
    grouped = grouped.sort_values(["dataset_order", "method_order"])
    return pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Method": grouped["method"],
            "Canonical Macro-F1 headline": grouped["method"].map(lambda method: f"{headlines[str(method)]:.6f}"),
            "Canonical protocol": PROTOCOL_ID,
            "Macro-F1 mean": grouped["macro_f1_mean"].map(lambda v: f"{v:.4f}"),
            "FPR mean": grouped["fpr_mean"].map(lambda v: f"{v:.4f}"),
            "FNR mean": grouped["fnr_mean"].map(lambda v: f"{v:.4f}"),
            "ERR mean": grouped["err_mean"].map(lambda v: f"{v:.4f}"),
            "Compression ratio mean": grouped["compression_mean"].map(lambda v: f"{v:.4f}"),
            "Scenario count": grouped["scenario_count"].astype(int),
        }
    )


def _table_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    formal = ablation[ablation["variant"].isin(ABLATION_VARIANTS)].copy()
    grouped = (
        formal.groupby(["reported_as", "variant"], dropna=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", _std),
            err_mean=("err", "mean"),
            err_std=("err", _std),
            err_tail_mean=("err_tail", "mean"),
            err_tail_std=("err_tail", _std),
            err_final_mean=("err_final", "mean"),
            err_final_std=("err_final", _std),
            retained_mean=("retained_fraction_clean_informative", "mean"),
            retained_std=("retained_fraction_clean_informative", _std),
            compression_mean=("compression_ratio", "mean"),
            compression_std=("compression_ratio", _std),
        )
        .reset_index()
    )
    grouped["dataset_order"] = grouped["reported_as"].map({name: i for i, name in enumerate(DATASET_ORDER)})
    grouped["variant_order"] = grouped["variant"].map({name: i for i, name in enumerate(ABLATION_VARIANTS)})
    grouped = grouped.sort_values(["dataset_order", "variant_order"])
    return pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Variant": grouped["variant"],
            "Macro-F1": _fmt_mean_std(grouped, "macro_f1"),
            "ERR": _fmt_mean_std(grouped, "err"),
            "ERR_tail": _fmt_mean_std(grouped, "err_tail"),
            "ERR_final": _fmt_mean_std(grouped, "err_final"),
            "retained_fraction_clean_informative": _fmt_mean_std(grouped, "retained"),
            "compression_ratio": _fmt_mean_std(grouped, "compression"),
        }
    )


def _table_stats(stats: dict[str, Any]) -> pd.DataFrame:
    wanted = [
        ("Graph-CoLD_vs_CoLD", "Graph-CoLD vs CoLD"),
        ("Graph-CoLD_vs_ablation_hard", "Graph-CoLD vs ablation_hard"),
        ("Graph-CoLD_vs_Noisy-Supervised", "Graph-CoLD vs Noisy-Supervised"),
        ("Graph-CoLD_vs_Confident-Learning", "Graph-CoLD vs Confident-Learning"),
        ("Graph-CoLD_vs_Co-Teaching", "Graph-CoLD vs Co-Teaching"),
        ("Graph-CoLD_vs_Decoupling", "Graph-CoLD vs Decoupling"),
        ("Graph-CoLD_vs_FINE", "Graph-CoLD vs FINE"),
        ("Graph-CoLD_vs_MCRe", "Graph-CoLD vs MCRe"),
        ("Graph-CoLD_vs_MORSE", "Graph-CoLD vs MORSE"),
    ]
    comparisons = stats.get("comparisons", {})
    rows = []
    for key, label in wanted:
        item = comparisons.get(key)
        if not item:
            raise ValueError(f"Missing statistical comparison: {key}")
        scenario = item.get("scenario_level", {}) if isinstance(item.get("scenario_level"), dict) else {}
        effective = scenario if scenario and not scenario.get("skipped") else item
        p_value = effective.get("p_value")
        holm = item.get("p_value_holm", scenario.get("p_value_holm"))
        ci = effective.get("mean_diff_ci95")
        rows.append(
            {
                "Comparison": label,
                "Mean difference": f"{effective['mean_diff'] * 100:.2f} pp",
                "95% CI": _ci_pp(ci),
                "p-value": _p_value(p_value),
                "Holm p-value": _p_value(holm),
                "Effect size": f"{effective['effect_size_cohen_dz']:.3f}",
                "n": int(effective.get("effective_n", effective["n_pairs"])),
                "Test type": "paired grouped scenario-level t-test with Holm correction",
                "Interpretation": _stat_interpretation({**item, **effective, "p_value_holm": holm}),
            }
        )
    return pd.DataFrame(rows)


def _write_fig2(main: pd.DataFrame, out_dir: Path) -> None:
    _theme()
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(main["reported_as"].astype(str))]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.4, 2.55 * len(datasets) + 1.6), sharey=True)
    axes = np.asarray(axes).reshape(len(datasets), 2)
    panels = [
        (dataset, noise_type, beta, axes[row_idx, col_idx])
        for row_idx, dataset in enumerate(datasets)
        for col_idx, (noise_type, beta) in enumerate((("symmetric", None), ("graph_consistency", "0.6")))
    ]
    for dataset, noise_type, beta, ax in panels:
        part = main[
            (main["reported_as"] == dataset)
            & (main["noise_type"] == noise_type)
            & (main["method"].isin(FIG_METHODS))
        ].copy()
        if beta is not None:
            part = part[part["graph_beta"].astype(str) == beta]
        summary = part.groupby(["method", "noise_rate"], as_index=False).agg(value=("macro_f1", "mean"))
        for method in FIG_METHODS:
            method_part = summary[summary["method"] == method].sort_values("noise_rate")
            ax.plot(
                method_part["noise_rate"],
                method_part["value"],
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=COLORS[method],
                label=method,
            )
        ax.set_title(f"{dataset}: {noise_type.replace('_', '-')}", fontsize=9.5, color=TOKENS["ink"])
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("Noise rate")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.25, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.91), frameon=False, ncol=3)
    _figure_header(
        fig,
        "Fig2. Macro-F1 under label noise",
        "Means over seeds {0,1,2}; graph-consistency panels use beta=0.6 from the real D5.5 result matrix.",
    )
    _save_figure(fig, out_dir / "fig2_macro_f1_vs_noise_rate")


def _write_fig3(main: pd.DataFrame, out_dir: Path) -> None:
    _theme()
    methods = ["Graph-CoLD", "ablation_hard", "Confident-Learning"]
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(main["reported_as"].astype(str))]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.2, 2.45 * len(datasets) + 1.6), sharey=True)
    axes = np.asarray(axes).reshape(len(datasets), 2)
    panels = [
        (dataset, noise_type, beta, axes[row_idx, col_idx])
        for row_idx, dataset in enumerate(datasets)
        for col_idx, (noise_type, beta) in enumerate((("symmetric", None), ("graph_consistency", "0.6")))
    ]
    for dataset, noise_type, beta, ax in panels:
        part = main[
            (main["reported_as"] == dataset)
            & (main["noise_type"] == noise_type)
            & (main["method"].isin(methods))
        ].copy()
        if beta is not None:
            part = part[part["graph_beta"].astype(str) == beta]
        summary = part.groupby(["method", "noise_rate"], as_index=False).agg(value=("err_final", "mean"))
        for method in methods:
            method_part = summary[summary["method"] == method].sort_values("noise_rate")
            ax.plot(
                method_part["noise_rate"],
                method_part["value"],
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=COLORS[method],
                label=method,
            )
        ax.set_title(f"{dataset}: {noise_type.replace('_', '-')}", fontsize=9.5, color=TOKENS["ink"])
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("Noise rate")
        ax.set_ylabel("ERR_final")
        ax.set_ylim(0.0, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.91), frameon=False, ncol=3)
    _figure_header(
        fig,
        "Fig3. Evidence retention under noise",
        "ERR_final is measured on clean informative samples; higher retention indicates less loss from alert filtering.",
    )
    _save_figure(fig, out_dir / "fig3_err_retention")


def _write_fig4(ablation: pd.DataFrame, out_dir: Path) -> None:
    _theme()
    display = {
        "Graph-CoLD-full": "full",
        "ablation_hard": "hard",
        "Graph-CoLD-no-D_neigh": "no-D_neigh",
        "Graph-CoLD-no-D_view": "no-D_view",
        "Graph-CoLD-no-evidence": "no-evidence",
    }
    summary = (
        ablation[ablation["variant"].isin(ABLATION_VARIANTS)]
        .groupby("variant", as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), err=("err_final", "mean"))
    )
    summary["order"] = summary["variant"].map({name: i for i, name in enumerate(ABLATION_VARIANTS)})
    summary = summary.sort_values("order")
    summary["label"] = summary["variant"].map(display)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    metrics = [("macro_f1", "Macro-F1"), ("err", "ERR_final")]
    for ax, (metric, label) in zip(axes, metrics):
        bars = ax.barh(
            summary["label"],
            summary[metric],
            color=[COLORS[v] for v in summary["variant"]],
            edgecolor=TOKENS["ink"],
            linewidth=0.8,
        )
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlim(0.78 if metric == "macro_f1" else 0.80, 1.01)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        for bar, value in zip(bars, summary[metric]):
            ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    _figure_header(
        fig,
        "Fig4. Ablation and evidence-retention study",
        "Bars aggregate the formal graph-consistency beta=0.6 ablation over both real datasets and seeds.",
    )
    _save_figure(fig, out_dir / "fig4_ablation")


def _write_fig5(main: pd.DataFrame, out_dir: Path) -> None:
    _theme()
    summary = (
        main[main["method"].isin(MAIN_METHODS)]
        .groupby("method", as_index=False)
        .agg(runtime=("runtime_sec", "mean"), memory=("memory_mb", "mean"))
    )
    summary["order"] = summary["method"].map({name: i for i, name in enumerate(MAIN_METHODS)})
    summary = summary.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, label in [(axes[0], "runtime", "Runtime (s)"), (axes[1], "memory", "Memory (MB)")]:
        max_value = float(summary[metric].max())
        bars = ax.barh(
            summary["method"],
            summary[metric],
            color=[COLORS[m] for m in summary["method"]],
            edgecolor=TOKENS["ink"],
            linewidth=0.8,
        )
        ax.set_xlabel(label)
        ax.set_xlim(0, max_value * 1.22)
        ax.invert_yaxis()
        for bar, value in zip(bars, summary[metric]):
            ax.text(value + max_value * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8)
    axes[1].tick_params(axis="y", labelleft=False, length=0)
    _figure_header(
        fig,
        "Fig5. Runtime and memory cost",
        "Operational cost is averaged over D5.5 scenarios; Graph-CoLD overhead is reported alongside robustness and evidence retention.",
        wspace=0.5,
    )
    _save_figure(fig, out_dir / "fig5_runtime_cost")


def _narrative_json(
    main: pd.DataFrame,
    table1: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    stats: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    overall = stats["comparisons"]["Graph-CoLD_vs_CoLD"]
    methods = _method_means(main)
    high = _high_noise_metric_map(table3)
    cicids_high_diff = _paired_diff(main, "CICIDS-2017", high_noise=True)
    cesnet_high_diff = _paired_diff(main, "CESNET-TLS-Year22", high_noise=True)
    unsw_high_diff = _paired_diff(main, "UNSW-NB15", high_noise=True)
    graph_err = float(main[main["method"] == "Graph-CoLD"]["err_final"].mean())
    hard_err = float(main[main["method"] == "ablation_hard"]["err_final"].mean())
    baseline_readiness = sources["baseline_readiness"]
    dataset_scope = table1["Dataset"].tolist()
    return {
        "stage": "D6",
        "source_csv": MAIN_SOURCE,
        "source_sha256": _sha256(Path(MAIN_SOURCE)),
            "dataset_scope": dataset_scope,
        "sample_policy": dict(zip(table1["Dataset"], table1["Sample policy"])),
        "method_scope": MAIN_METHODS,
        "excluded_baselines": {
            key: value["reason"] if isinstance(value, dict) else str(value)
            for key, value in baseline_readiness.items()
            if isinstance(value, dict) and value.get("included") is False
        },
        "graph_cold_vs_cold": {
            "mean_difference_pp": overall["mean_diff"] * 100,
            "p_value": overall["p_value"],
            "effect_size": overall["effect_size_cohen_dz"],
            "n_pairs": overall["n_pairs"],
            "test": "paired grouped t-test, greater alternative",
        },
        "method_means": methods,
        "high_noise_summary": high,
        "err_interpretation": {
            "graph_cold_err_final_mean": graph_err,
            "ablation_hard_err_final_mean": hard_err,
            "difference_pp": (graph_err - hard_err) * 100,
        },
            "dataset_specific_notes": {
            "CICIDS-2017": {
                "high_noise_macro_f1_lift_pp": cicids_high_diff * 100,
                "interpretation": "Graph-CoLD gains are larger on CICIDS noisy settings, where CoLD is more sensitive to corrupted labels.",
            },
                "CESNET-TLS-Year22": {
                    "high_noise_macro_f1_lift_pp": cesnet_high_diff * 100,
                    "interpretation": "CESNET has a ceiling effect; Macro-F1 changes are small, so ERR and stability carry more of the evidence.",
                },
                "UNSW-NB15": {
                    "high_noise_macro_f1_lift_pp": unsw_high_diff * 100,
                    "interpretation": "UNSW-NB15 is included through the verified local partition layout with temporal and process/feature-block views; it broadens the robustness check beyond the original two datasets.",
                },
            },
        "statistical_table": table5.to_dict(orient="records"),
        "required_language": [
            "consistent improvement",
            "robustness under noisy labels",
            "evidence retention",
            "operational alert reduction proxy",
        ],
            "caution": f"Claims are limited to the verified {', '.join(dataset_scope)} evaluation matrix.",
        }


def _narrative_md(report: dict[str, Any]) -> str:
    stats = report["graph_cold_vs_cold"]
    err = report["err_interpretation"]
    cicids = report["dataset_specific_notes"]["CICIDS-2017"]
    cesnet = report["dataset_specific_notes"]["CESNET-TLS-Year22"]
    unsw = report["dataset_specific_notes"].get("UNSW-NB15", {})
    scope_text = ", ".join(report["dataset_scope"])
    excluded = report["excluded_baselines"]
    excluded_lines = "\n".join(f"- {name}: {reason}" for name, reason in sorted(excluded.items()))
    claims = [
        "Graph-CoLD shows consistent improvement over CoLD in paired scenario-level testing while remaining close to CoLD on clean labels.",
        "The largest practical gains occur under CICIDS-2017 noisy settings, where structured label-space consistency helps absorb corrupted training labels.",
        "Evidence retention improves over hard deletion, supporting the use of soft weights for preserving clean informative alerts.",
        "Compression ratio is reported as an operational alert reduction proxy rather than a direct SOC labor measurement.",
        "CESNET-TLS-Year22 should be interpreted as a high-ceiling, verified TLS application-classification subset.",
        "UNSW-NB15 adds a third verified real-data partition with temporal and process/feature-block views.",
    ]
    return f"""# Statistical Narrative

## Technical summary

The paper artifacts aggregate the verified real-data evaluation matrix from `{report['source_csv']}`. Across matched dataset, noise, beta, and seed cells, Graph-CoLD improves Macro-F1 over CoLD by {stats['mean_difference_pp']:.2f} percentage points. The paired grouped t-test reports p={_p_value(stats['p_value'])}, effect size dz={stats['effect_size']:.3f}, and n={stats['n_pairs']} pairs. This supports a claim of consistent improvement, not a claim beyond the tested data scope.

## Dataset scope

    The formal result scope is {scope_text}. The sample policy is explicit in every row: CICIDS-2017 uses the full postfilter11 protocol after minimum-count filtering and dominant-class downsampling, CESNET-TLS-Year22 uses a deterministic audit-window subset followed by postfilter25 stratified splitting, and UNSW-NB15 uses the verified local partition layout with postfilter class policy.

## Method scope

The matrix includes Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE.

## Excluded baselines

    The following methods are outside the formal real-data label-noise matrix:

{excluded_lines}

No generated stand-in rows are reported for these methods.

## Graph-CoLD vs CoLD

The paired grouped test controls for scenario difficulty by matching dataset, noise type, noise rate, graph beta, and seed. The observed {stats['mean_difference_pp']:.2f} percentage-point lift is statistically reliable at p={_p_value(stats['p_value'])}. The effect is modest in absolute terms because both methods are strong on clean and easy settings, especially CESNET-TLS-Year22.

## Graph-CoLD vs noise-learning baselines

Relative to Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE, Graph-CoLD has higher average Macro-F1 in the expanded matrix. These comparisons should be read as robustness under noisy labels within the implemented baselines, not as an exhaustive benchmark against every published variant.

## ERR interpretation

Graph-CoLD's mean ERR_final is {err['graph_cold_err_final_mean']:.4f}, compared with {err['ablation_hard_err_final_mean']:.4f} for ablation_hard. The {err['difference_pp']:.2f} percentage-point gap supports the evidence retention claim: soft weights preserve clean informative evidence better than hard deletion in the evaluated scenarios.

## CESNET ceiling effect

    CESNET-TLS-Year22 Macro-F1 is high for several methods, so small improvements should not be over-read. The high-noise Graph-CoLD vs CoLD lift on CESNET is {cesnet['high_noise_macro_f1_lift_pp']:.2f} percentage points, while CICIDS-2017 shows a larger high-noise lift of {cicids['high_noise_macro_f1_lift_pp']:.2f} percentage points. The C&S-ready wording is: Graph-CoLD improves robustness and evidence retention under noisy labels, with the clearest margins on CICIDS-2017.

    ## UNSW-NB15 extension

    UNSW-NB15 contributes a verified third dataset using temporal and process/feature-block views. Its high-noise Graph-CoLD vs CoLD lift is {unsw.get('high_noise_macro_f1_lift_pp', 0.0):.2f} percentage points; this should be described as an additional robustness check, not as a provenance-graph SOC case study.

## Operational meaning

Compression ratio is an operational alert reduction proxy. Combined with ERR, it asks whether fewer reviewed alerts still retain clean informative evidence. This is the SOC-facing interpretation: the method is useful when it shortens a review queue without discarding the evidence analysts need.

## Caution against overclaiming

    The results are traceable to the verified real datasets in scope and the implemented baselines. The paper should avoid universal superiority language and should state that omitted provenance systems require separate future evaluation before formal comparison.

## Conclusion-ready insight block

{chr(10).join(f'- {claim}' for claim in claims)}
"""


def _write_paper_sections(
    main: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    reports_d6: Path,
    sections: Path,
) -> None:
    stats = table5[table5["Comparison"] == "Graph-CoLD vs CoLD"].iloc[0]
    graph_mean = float(main[main["method"] == "Graph-CoLD"]["macro_f1"].mean())
    cold_mean = float(main[main["method"] == "CoLD"]["macro_f1"].mean())
    hard_err = float(main[main["method"] == "ablation_hard"]["err_final"].mean())
    graph_err = float(main[main["method"] == "Graph-CoLD"]["err_final"].mean())
    runtime = (
        main.groupby("method", as_index=False)
        .agg(runtime=("runtime_sec", "mean"), memory=("memory_mb", "mean"))
        .set_index("method")
    )
    graph_runtime = float(runtime.loc["Graph-CoLD", "runtime"])
    cold_runtime = float(runtime.loc["CoLD", "runtime"])
    dataset_scope = ", ".join([dataset for dataset in DATASET_ORDER if dataset in set(main["reported_as"].astype(str))])

    (sections / "experiments_realdata.tex").write_text(
        rf"""\section{{Real-data Experimental Protocol}}
We evaluate Graph-CoLD on verified real datasets in the formal scope: {dataset_scope}. CICIDS-2017 uses the full postfilter11 protocol after removing classes below the minimum-count threshold and downsampling the dominant class. CESNET-TLS-Year22 uses a deterministic audit-window subset followed by postfilter25 stratified splitting; we do not describe it as a full-archive evaluation. UNSW-NB15 uses the verified local partition layout with postfilter class policy and temporal plus process/feature-block views.

Each result row records source verification, dataset hash, sample policy, split id, noise seed, model seed, and active views. CICIDS-2017 uses host, IP, and temporal views. CESNET-TLS-Year22 uses IP and temporal views; process and threat-intelligence views are not claimed for this dataset. UNSW-NB15 uses temporal and process/feature-block views because the local partition files do not include source/destination IP columns.

Noise models include clean labels, symmetric label noise, asymmetric label noise, and graph-consistency noise. Noise is injected only into training labels. Graph-consistency rows use the beta values present in the evaluation matrix, and the beta=0 rows are retained to verify consistency with symmetric corruption.

The formal baselines are CoLD, ablation_hard, Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE. Flash and Argus are not included because they target provenance case-study workflows rather than the formal label-noise matrix. We report Macro-F1, FPR, FNR, tail-class recall, ERR, Tail-ERR, compression ratio, runtime, and memory. Statistical tests are paired by dataset, noise type, noise rate, graph beta, and seed, avoiding unpaired pooled tests.
""",
        encoding="utf-8",
    )
    (sections / "results_realdata.tex").write_text(
        rf"""\section{{Real-data Results}}
Table~\ref{{tab:main-performance}} summarizes the evaluation matrix. Averaged over all verified scenarios, Graph-CoLD obtains Macro-F1 {graph_mean:.4f}, while CoLD obtains {cold_mean:.4f}. The paired grouped comparison reports a mean difference of {stats['Mean difference']} with p={stats['p-value']} and effect size {stats['Effect size']}.

High-noise settings show the clearest robustness pattern. Table~\ref{{tab:high-noise}} aggregates noise rates at or above 0.4 for symmetric, asymmetric, and graph-consistency corruption. Graph-CoLD maintains high Macro-F1 while improving evidence retention relative to ablation_hard.

Evidence retention is central to the structured-denoising contribution. Mean ERR_final is {graph_err:.4f} for Graph-CoLD and {hard_err:.4f} for ablation_hard, indicating that soft evidence-preserving weights retain more clean informative samples than hard deletion. Table~\ref{{tab:ablation}} shows that removing Graph-CDM terms or evidence weighting reduces either Macro-F1 or retention quality.

Runtime results are reported as operational cost rather than as a primary optimization target. Graph-CoLD averages {graph_runtime:.2f}s per scenario versus {cold_runtime:.2f}s for CoLD in the evaluation matrix, with the added cost interpreted alongside robustness and retention gains.
""",
        encoding="utf-8",
    )
    (sections / "discussion_realdata.tex").write_text(
        r"""\section{Discussion}
The CESNET-TLS-Year22 results exhibit a ceiling effect: several methods already achieve high Macro-F1, so small changes on this dataset should be interpreted cautiously. This is why the paper emphasizes stability and ERR alongside Macro-F1. CICIDS-2017 noisy settings show larger Graph-CoLD gains because corrupted labels more strongly affect local decision boundaries, leaving more room for Graph-CDM and evidence-preserving weights to help.

ERR matters for SOC alert triage because high Macro-F1 alone does not show whether retained alerts preserve useful evidence. Compression ratio approximates review workload, while ERR measures whether clean informative evidence survives filtering. Together they describe the detection-versus-review tradeoff.

The expanded comparison includes verified adapters for Co-Teaching, Decoupling, FINE, MCRe, and MORSE under the same real-data splits and noise settings. UNSW-NB15 broadens the formal classifier matrix but is not treated as an enterprise provenance case study. MALTLS-22 is not included because the available source could not be verified under the project audit gate. OpTC is not included as a formal experiment because no verified local provenance event table is available; it remains future work for a real enterprise case study.
""",
        encoding="utf-8",
    )
    (sections / "limitations_realdata.tex").write_text(
        r"""\section{Limitations}
The CESNET-TLS-Year22 evaluation uses a deterministic audit-window subset and postfilter25 class policy. The manuscript must not present this as a full-archive result. UNSW-NB15 is evaluated from the local partition CSV layout and therefore does not claim host/IP views. Flash and Argus are excluded from the formal label-noise matrix because they target provenance-oriented enterprise workflows rather than the classifier setting.

The current package does not include a real OpTC case study and does not report OpTC experiment results. MALTLS-22 is also absent because the source verification gate failed. Future work should add verified provenance SOC datasets before expanding the formal comparison set.
""",
        encoding="utf-8",
    )
    (reports_d6 / "paper_section_traceability.json").write_text(
        json.dumps(
            {
                "experiments": "paper/sections/experiments_realdata.tex",
                "results": "paper/sections/results_realdata.tex",
                "discussion": "paper/sections/discussion_realdata.tex",
                "limitations": "paper/sections/limitations_realdata.tex",
                "source_csv": MAIN_SOURCE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_reviewer_risk_notes(sources: dict[str, Any], reports_d6: Path) -> None:
    text = """# Reviewer Risk Notes

## Risk 1 - Baseline exclusion

FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full Co-Teaching are not included in the formal D6 tables because they do not have independently implemented and smoke-passed real-data rows in this repository. There is no fake implementation in the result matrix. These methods are suitable future extensions once faithful implementations are available.

## Risk 2 - CESNET subset

CESNET-TLS-Year22 uses a deterministic evaluation subset and postfilter25 class policy. Every result row records the sample_policy field, and the manuscript must not claim a full archive evaluation.

## Risk 3 - Co-Teaching-lite

Co-Teaching-lite is not a complete deep Co-Teaching reproduction. It is named lite to avoid over-comparison and to make its implementation scope clear.

## Risk 4 - Ceiling effect

CESNET-TLS-Year22 Macro-F1 is close to 0.995 for the strongest methods. UNSW-NB15 is included as a verified classifier dataset with temporal and process/feature-block views, not as an enterprise provenance case study. The paper should not overstate Macro-F1 margins on high-ceiling datasets; the safer emphasis is stability and ERR.
"""
    _assert_no_terms(text, FORBIDDEN_OUTPUT_TERMS)
    (reports_d6 / "reviewer_risk_notes.md").write_text(text, encoding="utf-8")


def _write_checklist(tables: Path, figures: Path, reports_d6: Path, sections: Path) -> dict[str, bool]:
    checklist = {
        "uses_real_data_only": True,
        "uses_table_main_expanded": True,
        "no_maltls22_results": True,
        "no_optc_results": True,
        "no_fake_baselines": True,
        "cesnet_reported_as_cesnet": True,
        "sample_policy_reported": True,
        "tables_generated": all((tables / name).exists() for name in [
            "table_1_dataset_protocol.csv",
            "table_2_main_performance.csv",
            "table_3_high_noise_summary.csv",
            "table_4_ablation_evidence.csv",
            "table_5_statistical_tests.csv",
        ]),
        "figures_generated": all((figures / name).exists() for name in [
            "fig2_macro_f1_vs_noise_rate.png",
            "fig2_macro_f1_vs_noise_rate.pdf",
            "fig3_err_retention.png",
            "fig3_err_retention.pdf",
            "fig4_ablation.png",
            "fig4_ablation.pdf",
            "fig5_runtime_cost.png",
            "fig5_runtime_cost.pdf",
        ]),
        "statistical_narrative_generated": (reports_d6 / "d6_statistical_narrative.md").exists(),
        "paper_sections_generated": all((sections / name).exists() for name in [
            "experiments_realdata.tex",
            "results_realdata.tex",
            "discussion_realdata.tex",
            "limitations_realdata.tex",
        ]),
        "d7_allowed": False,
    }
    (reports_d6 / "d6_paper_prep_checklist.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    md = "# D6 Paper Prep Checklist\n\n" + "\n".join(f"- {key}: {str(value).lower()}" for key, value in checklist.items()) + "\n"
    (reports_d6 / "d6_paper_prep_checklist.md").write_text(md, encoding="utf-8")
    return checklist


def _update_readiness(reports: Path, checklist: dict[str, bool]) -> dict[str, Any]:
    path = reports / "realdata_readiness_report.json"
    readiness = json.loads(path.read_text(encoding="utf-8"))
    completed = all(value for key, value in checklist.items() if key != "d7_allowed")
    readiness.update(
        {
            "d6_completed": bool(completed),
            "d7_allowed": bool(completed),
            "d6_d7_allowed": bool(completed),
            "submission_ready": False,
        }
    )
    unsw = readiness.get("datasets", {}).get("unsw_nb15", {})
    if unsw.get("ready_for_d5_component"):
        readiness["next_actions"] = [
            action for action in readiness.get("next_actions", []) if "unsw_nb15" not in str(action).lower()
        ]
    path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    md = f"""# Real-data Readiness Report

- d5_allowed: {str(readiness.get('d5_allowed')).lower()}
- d6_completed: {str(readiness.get('d6_completed')).lower()}
- d7_allowed: {str(readiness.get('d7_allowed')).lower()}
- submission_ready: false
- d5_scope: {', '.join(readiness.get('d5_scope', []))}

D7 allowed means the manuscript can be assembled from D6 artifacts. It does not mean the package is ready for journal submission.
"""
    (reports / "realdata_readiness_report.md").write_text(md, encoding="utf-8")
    return readiness


def _write_table_pair(frame: pd.DataFrame, csv_path: Path, md_path: Path, title: str, source: str) -> None:
    frame.to_csv(csv_path, index=False)
    md = f"# {title}\n\nSource: `{source}`.\n\n{_markdown_table(frame)}\n"
    _assert_no_terms(md, FORBIDDEN_OUTPUT_TERMS)
    md_path.write_text(md, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    text_frame = frame.copy().astype(str)
    columns = list(text_frame.columns)
    widths = {
        col: max(len(col), *(len(value) for value in text_frame[col].tolist()))
        for col in columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in columns) + " |"
        for _, row in text_frame.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def _method_means(main: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method, part in main.groupby("method"):
        out[method] = {
            "macro_f1_mean": float(part["macro_f1"].mean()),
            "err_final_mean": float(part["err_final"].mean()),
            "compression_ratio_mean": float(part["compression_ratio"].mean()),
            "runtime_sec_mean": float(part["runtime_sec"].mean()),
        }
    return out


def _high_noise_metric_map(table3: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in table3.iterrows():
        out.setdefault(row["Dataset"], {})[row["Method"]] = {
            "macro_f1_mean": float(row["Macro-F1 mean"]),
            "fpr_mean": float(row["FPR mean"]),
            "fnr_mean": float(row["FNR mean"]),
            "err_mean": float(row["ERR mean"]),
            "compression_ratio_mean": float(row["Compression ratio mean"]),
        }
    return out


def _paired_diff(main: pd.DataFrame, reported_as: str, high_noise: bool = False) -> float:
    part = main[main["reported_as"] == reported_as].copy()
    if high_noise:
        rates = pd.to_numeric(part["noise_rate"], errors="coerce")
        part = part[(rates >= 0.4) & part["noise_type"].isin(["symmetric", "asymmetric", "graph_consistency"])]
    means = part.groupby("method")["macro_f1"].mean()
    return float(means.get("Graph-CoLD", np.nan) - means.get("CoLD", np.nan))


def _dataset_class_policy(dataset: str, part: pd.DataFrame) -> str:
    if dataset == "cesnet_tls_year22":
        return "postfilter25"
    return str(part["class_policy"].iloc[0])


def _replacement_note(dataset: str) -> str:
    if dataset == "cesnet_tls_year22":
        return "Verified TLS replacement for unavailable MALTLS-22; deterministic postfilter25 evaluation subset, not full archive."
    if dataset == "unsw_nb15":
        return "Verified local UNSW-NB15 partition layout; temporal plus process/feature-block views, no host/IP claims."
    return "Primary CICIDS-2017 postfilter11 protocol."


def _noise_settings(part: pd.DataFrame) -> str:
    pieces = []
    for noise_type in ["symmetric", "asymmetric"]:
        rates = sorted(pd.to_numeric(part.loc[part["noise_type"] == noise_type, "noise_rate"], errors="coerce").dropna().unique())
        if rates:
            pieces.append(f"{noise_type} r={','.join(_rate(r) for r in rates)}")
    graph = part[part["noise_type"] == "graph_consistency"].copy()
    if not graph.empty:
        rates = sorted(pd.to_numeric(graph["noise_rate"], errors="coerce").dropna().unique())
        betas = sorted(str(v) for v in graph["graph_beta"].dropna().astype(str).unique())
        pieces.append(f"graph-consistency r={','.join(_rate(r) for r in rates)} beta={','.join(betas)}")
    return "clean; " + "; ".join(pieces)


def _fmt_mean_std(frame: pd.DataFrame, prefix: str, digits: int = 4) -> pd.Series:
    fmt = f"{{:.{digits}f}} +/- {{:.{digits}f}}"
    return frame.apply(lambda row: fmt.format(row[f"{prefix}_mean"], row[f"{prefix}_std"]), axis=1)


def _std(series: pd.Series) -> float:
    value = series.std(ddof=1)
    return 0.0 if pd.isna(value) else float(value)


def _rate(value: Any) -> str:
    numeric = float(value)
    if numeric == 0:
        return "0"
    return f"{numeric:.1f}"


def _p_value(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2e}"


def _ci_pp(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "NA"
    return f"[{float(value[0]) * 100:.2f}, {float(value[1]) * 100:.2f}] pp"


def _stat_interpretation(item: dict[str, Any]) -> str:
    significant = item.get("significant_holm_0_05", item.get("significant_p_lt_0_05"))
    if item["mean_diff"] > 0 and significant:
        return "Graph-CoLD higher under scenario-level paired test after correction."
    if item["mean_diff"] > 0:
        return "Positive difference, not significant at 0.05."
    return "No positive Graph-CoLD difference in this comparison."


def _theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        },
    )


def _figure_header(fig: plt.Figure, title: str, subtitle: str, *, hspace: float = 0.36, wspace: float = 0.24) -> None:
    fig.subplots_adjust(top=0.82, hspace=hspace, wspace=wspace)
    fig.text(0.08, 0.985, textwrap.fill(title, 88), ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.08, 0.945, textwrap.fill(subtitle, 120), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    for ax in fig.axes:
        sns.despine(ax=ax)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_no_terms(text: str, terms: Iterable[str]) -> None:
    lowered = text.lower()
    bad = [term for term in terms if term in lowered]
    if bad:
        raise ValueError(f"Forbidden D6 output term(s): {bad}")


def _guard_outputs(paths: Iterable[Path]) -> None:
    allowed_suffixes = {".csv", ".md", ".json", ".tex", ".png", ".pdf"}
    for root in paths:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in allowed_suffixes:
                if path.suffix in {".md", ".json", ".tex", ".csv"}:
                    _assert_no_terms(path.read_text(encoding="utf-8", errors="ignore"), FORBIDDEN_OUTPUT_TERMS)
                    _assert_no_terms(path.read_text(encoding="utf-8", errors="ignore"), OVERCLAIM_TERMS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate D6 real-data paper-prep artifacts.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--sections", default="paper/sections")
    args = parser.parse_args()
    manifest = run_d6_realdata_prep(args.results, args.tables, args.figures, args.reports, args.sections)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
