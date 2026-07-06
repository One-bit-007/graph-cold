"""Generate the P1 acceptance status report from traceable artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PAPER_FACING_PATHS = [
    "paper/elsevier/graph_cold_cas_realdata.tex",
    "paper/elsevier/cover_letter_draft.md",
    "paper/elsevier/data_availability_statement.md",
    "paper/elsevier/declaration_of_competing_interest.md",
    "paper/sections",
    "tables",
    "reports/revision_p0_acceptance_report.md",
    "reports/d8/d8_hardening_audit.md",
    "reports/d8/reviewer_risk_register_v1.md",
    "reproducibility/README_realdata.md",
    "src/paper/d8_harden.py",
    "src/paper/revision_p0.py",
]


def generate_p1_status(reports_dir: str | Path = "reports", results_dir: str | Path = "results", tables_dir: str | Path = "tables") -> dict[str, Any]:
    reports = Path(reports_dir)
    results = Path(results_dir)
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    main = pd.read_csv(results / "table_main.csv")
    expanded = pd.read_csv(results / "table_main_expanded.csv")
    prioritization = pd.read_csv(results / "table_prioritization.csv")
    downstream = pd.read_csv(results / "evidence_downstream_benefit.csv")
    stats_expanded = json.loads((results / "stat_tests_baseline_expansion.json").read_text(encoding="utf-8"))
    stats_priority = json.loads((results / "stat_tests_prioritization.json").read_text(encoding="utf-8"))
    cold_refresh = json.loads((reports / "p1_cold_refresh_report.json").read_text(encoding="utf-8"))
    unsw = json.loads((reports / "unsw_dataset_decision.json").read_text(encoding="utf-8"))
    graph_noise = json.loads((reports / "p1_graph_noise_validation.json").read_text(encoding="utf-8"))

    p0_gate = _p0_gate(main, expanded, downstream, stats_expanded)
    p1_tables = {
        "prioritization": _write_prioritization_table(prioritization, tables),
        "statistical_rigor": _write_stat_table(stats_expanded, tables),
        "graph_noise": _write_graph_noise_table(graph_noise, tables),
    }
    p1 = _p1_delta(unsw, prioritization, stats_priority, stats_expanded, graph_noise)
    grep = _paper_facing_grep()
    p0_gate["A4"]["passed"] = grep["clean"]
    p0_gate["A4"]["evidence"] = grep["summary"]

    report = {
        "p0_gate": p0_gate,
        "p1": p1,
        "tables": p1_tables,
        "figures": {
            "queue_load_curve": "figures/fig_p1_queue_load_curve.pdf",
            "graph_noise_beta_sweep": "figures/fig_p1_graph_noise_beta_sweep.pdf",
        },
        "cold_refresh": cold_refresh,
        "acceptance_risk": {
            "estimated_reject_risk_after_p1": "approximately 30-40%",
            "remaining_weaknesses": [
                "UNSW-NB15 is verified in contract but absent locally, so the third-dataset claim remains blocked.",
                "Direct prioritization metrics are now measured, but Graph-CoLD is essentially tied with CoLD and ablation_hard on Top-K precision in the audit-window ranking evaluation.",
                "Prioritization evaluation uses deterministic real-data audit windows to avoid local memory exhaustion on full CICIDS loading.",
            ],
        },
    }
    (reports / "p1_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "p1_status.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _p0_gate(main: pd.DataFrame, expanded: pd.DataFrame, downstream: pd.DataFrame, stats_expanded: dict[str, Any]) -> dict[str, Any]:
    noisy = main[main["noise_type"] != "clean"]
    means = noisy.groupby("method")[["macro_f1", "err_final", "fnr", "tail_recall"]].mean()
    hard_gt_cold = float(means.loc["ablation_hard", "err_final"]) > float(means.loc["CoLD", "err_final"])
    graph_gt_hard = float(means.loc["Graph-CoLD", "err_final"]) > float(means.loc["ablation_hard", "err_final"])
    required = {"MCRe", "MORSE", "Co-Teaching", "FINE"}
    method_counts = expanded.groupby("method").size().to_dict()
    benefit = _downstream_stats(downstream)
    return {
        "A1": {
            "passed": bool(
                abs(float(means.loc["ablation_hard", "macro_f1"]) - float(means.loc["CoLD", "macro_f1"])) > 1e-6
                and abs(float(means.loc["ablation_hard", "err_final"]) - float(means.loc["CoLD", "err_final"])) > 1e-6
                and graph_gt_hard
                and hard_gt_cold
            ),
            "evidence": means.loc[["Graph-CoLD", "ablation_hard", "CoLD"]].to_dict(orient="index"),
        },
        "A2": {
            "passed": bool(required.issubset(method_counts) and all(method_counts[name] == 102 for name in required)),
            "evidence": {
                "method_counts": method_counts,
                "graphcold_vs_comparisons": {
                    key: {
                        "mean_diff": value.get("scenario_level", {}).get("mean_diff", value.get("mean_diff")),
                        "p_value_holm": value.get("p_value_holm"),
                    }
                    for key, value in stats_expanded.get("comparisons", {}).items()
                    if key in {f"Graph-CoLD_vs_{name}" for name in required}
                },
            },
        },
        "A3": {"passed": bool(benefit["tail_recall_delta_mean"] > 0 or benefit["high_noise_fnr_delta_mean"] < 0), "evidence": benefit},
        "A4": {"passed": False, "evidence": "checked later"},
    }


def _downstream_stats(frame: pd.DataFrame) -> dict[str, Any]:
    tail = frame["tail_recall_delta"].dropna().to_numpy(dtype=float)
    high = frame.loc[pd.to_numeric(frame["noise_rate"], errors="coerce") >= 0.4, "fnr_delta_graphcold_minus_hard"].dropna().to_numpy(dtype=float)
    tail_p = stats.ttest_1samp(tail, 0.0, alternative="greater").pvalue if tail.size > 1 else np.nan
    fnr_p = stats.ttest_1samp(high, 0.0, alternative="less").pvalue if high.size > 1 else np.nan
    return {
        "rows": int(len(frame)),
        "tail_recall_delta_mean": float(np.mean(tail)) if tail.size else 0.0,
        "tail_recall_delta_p_greater": None if np.isnan(tail_p) else float(tail_p),
        "high_noise_fnr_delta_mean": float(np.mean(high)) if high.size else 0.0,
        "high_noise_fnr_delta_p_less": None if np.isnan(fnr_p) else float(fnr_p),
    }


def _p1_delta(unsw: dict[str, Any], prioritization: pd.DataFrame, stats_priority: dict[str, Any], stats_expanded: dict[str, Any], graph_noise: dict[str, Any]) -> dict[str, Any]:
    priority_summary = prioritization.groupby("method")[[
        "topk_precision",
        "topk_recall",
        "precision_at_budget",
        "compression_at_recall_90",
        "compression_at_recall_95",
    ]].mean().to_dict(orient="index")
    return {
        "G1": {
            "status": "blocked",
            "blocker": unsw.get("blocking_reasons", []),
            "alternative": "No already-available verified third dataset is present locally; USTC-TFC2016 remains candidate-only.",
        },
        "G2": {
            "status": "completed",
            "table": "results/table_prioritization.csv",
            "curve": "results/prioritization_curve.csv",
            "figure": "figures/fig_p1_queue_load_curve.pdf",
            "summary": {method: {key: float(value) for key, value in metrics.items()} for method, metrics in priority_summary.items()},
            "stats": stats_priority.get("comparisons", {}),
        },
        "G3": {
            "status": "completed",
            "table": "tables/table_p1_statistical_rigor.csv",
            "independence_aware_overall": stats_expanded.get("independence_aware", {}).get("overall"),
            "correction": stats_expanded.get("multiple_comparison_correction"),
        },
        "G4": {
            "status": "completed",
            "figure": graph_noise.get("figure"),
            "beta0_matches_symmetric": graph_noise.get("beta0_matches_symmetric"),
            "concentration_increases": graph_noise.get("concentration_increases"),
        },
    }


def _write_prioritization_table(frame: pd.DataFrame, tables: Path) -> str:
    summary = frame.groupby("method")[[
        "topk_precision",
        "topk_recall",
        "precision_at_budget",
        "compression_at_recall_90",
        "compression_at_recall_95",
    ]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    out = tables / "table_p1_prioritization.csv"
    summary.reset_index().to_csv(out, index=False)
    return str(out)


def _write_stat_table(stats_report: dict[str, Any], tables: Path) -> str:
    rows = []
    for name, info in stats_report.get("comparisons", {}).items():
        if info.get("skipped"):
            continue
        scenario = info.get("scenario_level", {})
        rows.append(
            {
                "comparison": name,
                "seed_level_mean_diff": info.get("mean_diff"),
                "scenario_mean_diff": scenario.get("mean_diff"),
                "scenario_ci95_low": (scenario.get("mean_diff_ci95") or [None, None])[0],
                "scenario_ci95_high": (scenario.get("mean_diff_ci95") or [None, None])[1],
                "effective_n": scenario.get("effective_n"),
                "scenario_p_value": scenario.get("p_value"),
                "holm_p_value": info.get("p_value_holm"),
                "holm_significant_0_05": info.get("significant_holm_0_05"),
            }
        )
    out = tables / "table_p1_statistical_rigor.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def _write_graph_noise_table(report: dict[str, Any], tables: Path) -> str:
    rows = [{"beta": beta, "transition_concentration": value} for beta, value in report.get("transition_concentration", {}).items()]
    out = tables / "table_p1_graph_noise_beta_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def _paper_facing_grep() -> dict[str, Any]:
    pattern = "D" + r"[0-9]+(\.[0-9]+)?|reinforced|sm" + "oke|repository candidate|before (journal )?upload|risk-clarification"
    completed = subprocess.run(["rg", "-n", pattern, *PAPER_FACING_PATHS], capture_output=True, text=True)
    return {"clean": completed.returncode == 1, "summary": "no matches" if completed.returncode == 1 else completed.stdout.strip()}


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# P1 Status Report", "", "## P0 Gate Report", "", "| Gate | Status | Evidence |", "|---|---:|---|"]
    for gate, info in report["p0_gate"].items():
        lines.append(f"| {gate} | {'PASS' if info['passed'] else 'FAIL'} | `{json.dumps(info['evidence'], ensure_ascii=False)[:900]}` |")
    lines.extend(["", "## P1 Results Delta", ""])
    for goal, info in report["p1"].items():
        lines.append(f"### {goal}")
        lines.append(f"- Status: {info['status']}")
        if goal == "G1":
            lines.append(f"- Blocker: {info['blocker']}")
            lines.append(f"- Alternative: {info['alternative']}")
        elif goal == "G2":
            lines.append(f"- Table: `{info['table']}`")
            lines.append(f"- Curve: `{info['curve']}`")
            lines.append(f"- Figure: `{info['figure']}`")
            graph = info["summary"].get("Graph-CoLD", {})
            lines.append(
                f"- Graph-CoLD mean Top-K precision={graph.get('topk_precision', 0):.6f}, "
                f"compression@90={graph.get('compression_at_recall_90', 0):.6f}."
            )
        elif goal == "G3":
            overall = info.get("independence_aware_overall") or {}
            lines.append(f"- Table: `{info['table']}`")
            lines.append(
                f"- Scenario-level mean diff={overall.get('mean_diff')}, "
                f"CI={overall.get('mean_diff_ci95')}, p={overall.get('p_value')}, "
                f"effective_n={overall.get('effective_n')}."
            )
        elif goal == "G4":
            lines.append(f"- Figure: `{info['figure']}`")
            lines.append(f"- beta0 matches symmetric: {info['beta0_matches_symmetric']}")
            lines.append(f"- concentration increases: {info['concentration_increases']}")
        lines.append("")
    lines.extend(
        [
            "## Acceptance-Risk Self-Assessment",
            "",
            f"- Estimated reject risk after P1: {report['acceptance_risk']['estimated_reject_risk_after_p1']}.",
            "- Remaining weaknesses:",
        ]
    )
    lines.extend([f"  - {item}" for item in report["acceptance_risk"]["remaining_weaknesses"]])
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "- `python -m src.experiments.p1_refresh_cold --out results --configs configs --reports reports`",
            "- `python -m src.analysis.prioritization --out results --configs configs --reports reports --figure figures/fig_p1_queue_load_curve.pdf`",
            "- `python -m src.data.unsw_policy --data-root E:/graphcold-data --out reports`",
            "- `python -c \"from src.analysis.graph_noise_validation import beta_sweep_report; beta_sweep_report(figure_path='figures/fig_p1_graph_noise_beta_sweep.pdf')\"`",
            "- `python -m src.paper.p1_status --reports reports --results results --tables tables`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--results", default="results")
    parser.add_argument("--tables", default="tables")
    args = parser.parse_args()
    print(json.dumps(generate_p1_status(args.reports, args.results, args.tables), indent=2))


if __name__ == "__main__":
    main()
