"""D9.5 smoke gate for Decoupling and FINE-style baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments import d5, d5_baseline_expansion
from src.experiments.d9_5_baseline_common import METHODS, make_baseline, pass_smoke_row, row_from_result, timed_baseline


SMOKE_DATASETS = ("cicids2017", "cesnet_tls_year22")
SMOKE_SPECS = (
    {"noise_type": "clean", "noise_rate": 0.0, "graph_beta": "none"},
    {"noise_type": "symmetric", "noise_rate": 0.2, "graph_beta": "none"},
)
SMOKE_SEED = 42


def run_d9_5_baseline_smoke(
    methods: list[str] | tuple[str, ...] = METHODS,
    out: str | Path = "reports",
    configs: str | Path = "configs",
) -> dict[str, Any]:
    reports = Path(out) / "d9_5"
    reports.mkdir(parents=True, exist_ok=True)
    configs_path = Path(configs)
    _write_feasibility_audit(reports)
    scale_policy = d5.write_scale_policy_report(Path(out))

    rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {method: [] for method in methods}
    graph_cache_by_dataset: dict[tuple[str, int], dict[float, Any]] = {}
    for dataset_name in SMOKE_DATASETS:
        bundle = d5._load_formal_dataset(dataset_name, SMOKE_SEED, configs_path, scale_policy)
        evidence = d5_baseline_expansion._evidence(bundle)
        for spec in SMOKE_SPECS:
            graph_cache = graph_cache_by_dataset.setdefault((dataset_name, SMOKE_SEED), {})
            noisy, flip = d5._inject_noise(bundle.dataset, spec, SMOKE_SEED, graph_cache)
            for method in methods:
                baseline = make_baseline(method, SMOKE_SEED, float(spec["noise_rate"]))
                try:
                    print(f"[d9.5-smoke] {dataset_name} {spec['noise_type']} {method}", flush=True)
                    result, runtime_sec, memory_mb = timed_baseline(baseline, bundle, noisy, spec)
                    row = row_from_result(bundle, spec, SMOKE_SEED, result, runtime_sec, memory_mb, evidence, flip, False)
                    ok, reasons = pass_smoke_row(row, result, noisy)
                    row.update(
                        {
                            "dataset": dataset_name,
                            "train_label_source": result.details.get("train_label_source", "noisy_y_train"),
                            "eval_label_source": result.details.get("eval_label_source", "clean_y_test"),
                            "smoke_passed": bool(ok),
                            "failure_reason": "; ".join(reasons),
                            "details": result.details,
                        }
                    )
                    rows.append(_smoke_report_row(row))
                    if not ok:
                        failures.setdefault(method, []).append(f"{dataset_name}/{spec['noise_type']}: {row['failure_reason']}")
                except Exception as exc:
                    failures.setdefault(method, []).append(f"{dataset_name}/{spec['noise_type']}: {exc}")
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "noise_type": spec["noise_type"],
                            "noise_rate": spec["noise_rate"],
                            "seed": SMOKE_SEED,
                            "method": method,
                            "macro_f1": None,
                            "fpr": None,
                            "fnr": None,
                            "err": None,
                            "err_tail": None,
                            "err_final": None,
                            "retained_fraction": None,
                            "runtime_sec": None,
                            "memory_mb": None,
                            "train_label_source": "noisy_y_train",
                            "eval_label_source": "clean_y_test",
                            "smoke_passed": False,
                            "failure_reason": str(exc),
                        }
                    )

    required_pairs = {(dataset, spec["noise_type"]) for dataset in SMOKE_DATASETS for spec in SMOKE_SPECS}
    passed = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        seen = {(row["dataset"], row["noise_type"]) for row in method_rows if row.get("smoke_passed") is True}
        if seen == required_pairs and not failures.get(method):
            passed.append(method)
    report = {
        "stage": "D9.5 baseline smoke gate",
        "seed": SMOKE_SEED,
        "datasets": list(SMOKE_DATASETS),
        "settings": ["clean", "symmetric_20"],
        "methods_requested": list(methods),
        "passed": bool(passed),
        "passed_methods": passed,
        "failed_methods": {method: reasons for method, reasons in failures.items() if reasons},
        "rows": rows,
        "criteria": {
            "no_crash": True,
            "no_nan_inf": True,
            "no_100_f1_zero_fpr_fnr_anomaly": True,
            "train_label_source": "noisy_y_train under noise",
            "eval_label_source": "clean_y_test",
            "deterministic_under_seed_42": True,
            "macro_f1_min": 0.50,
            "fine_style_retained_fraction_nonzero": True,
            "decoupling_disagreement_update_recorded": True,
            "no_test_label_leakage": True,
        },
    }
    (reports / "baseline_smoke_decoupling_fine.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "baseline_smoke_decoupling_fine.md").write_text(_smoke_markdown(report), encoding="utf-8")
    return report


def _smoke_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "noise_type": row["noise_type"],
        "noise_rate": float(row["noise_rate"]),
        "seed": int(row["seed"]),
        "method": row["method"],
        "macro_f1": float(row["macro_f1"]),
        "fpr": float(row["fpr"]),
        "fnr": float(row["fnr"]),
        "err": float(row["err"]),
        "err_tail": float(row["err_tail"]),
        "err_final": float(row["err_final"]),
        "retained_fraction": float(row["retained_fraction"]),
        "runtime_sec": float(row["runtime_sec"]),
        "memory_mb": float(row["memory_mb"]),
        "train_label_source": row["train_label_source"],
        "eval_label_source": row["eval_label_source"],
        "smoke_passed": bool(row["smoke_passed"]),
        "failure_reason": row.get("failure_reason", ""),
        "details": row.get("details", {}),
    }


def _write_feasibility_audit(out: Path) -> dict[str, Any]:
    methods = [
        {
            "method": "Decoupling",
            "paper_source_verified": True,
            "source": "Decoupling when to update from how to update, NeurIPS 2017",
            "official_code_available": False,
            "data_compatibility_with_CICIDS_CESNET": True,
            "implementation_target": "tabular two-classifier disagreement update",
            "faithful_full_or_style_approximate": "faithful to standard disagreement-update mechanism",
            "include_in_this_patch": True,
            "reason": "classic noisy-label disagreement-update baseline; feasible on tabular CICIDS/CESNET",
        },
        {
            "method": "FINE",
            "paper_source_verified": True,
            "source": "FINE: Filtering Noisy Instances via their Eigenvectors, NeurIPS 2021",
            "official_code_available": False,
            "data_compatibility_with_CICIDS_CESNET": "partial",
            "implementation_target": "none",
            "faithful_full_or_style_approximate": "excluded full method",
            "include_in_this_patch": False,
            "reason": "full original implementation is not reproduced here",
        },
        {
            "method": "FINE-style",
            "paper_source_verified": True,
            "source": "FINE-inspired class-wise eigenvector filtering",
            "official_code_available": False,
            "data_compatibility_with_CICIDS_CESNET": True,
            "implementation_target": "standardized feature PCA plus class-wise eigenvector filtering",
            "faithful_full_or_style_approximate": "style/approximate",
            "include_in_this_patch": True,
            "reason": "representation/eigenvector filtering baseline using standardized feature projections",
            "warning": "not full FINE unless exact original implementation is reproduced",
        },
    ]
    for method in ("MCRe", "MORSE", "Flash", "Argus"):
        methods.append(
            {
                "method": method,
                "paper_source_verified": False,
                "official_code_available": False,
                "data_compatibility_with_CICIDS_CESNET": False,
                "implementation_target": "excluded",
                "faithful_full_or_style_approximate": "not implemented",
                "include_in_this_patch": False,
                "reason": "no verified real-data compatible implementation in current artifact",
            }
        )
    audit = {
        "stage": "D9.5 baseline feasibility audit",
        "baseline_reinforcement_scope": ["Decoupling", "FINE-style"],
        "methods": methods,
    }
    (out / "baseline_feasibility_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# D9.5 Baseline Feasibility Audit", ""]
    for item in methods:
        lines.append(f"## {item['method']}")
        lines.append(f"- include_in_this_patch: {item['include_in_this_patch']}")
        lines.append(f"- target: {item['implementation_target']}")
        lines.append(f"- faithfulness: {item['faithful_full_or_style_approximate']}")
        lines.append(f"- reason: {item['reason']}")
        if item.get("warning"):
            lines.append(f"- warning: {item['warning']}")
        lines.append("")
    (out / "baseline_feasibility_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def _smoke_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D9.5 Baseline Smoke Gate",
        "",
        f"- Seed: {report['seed']}",
        f"- Passed methods: {', '.join(report['passed_methods']) or 'none'}",
        "",
        "## Rows",
    ]
    for row in report["rows"]:
        lines.append(
            f"- {row['dataset']} {row['noise_type']} {row['method']}: "
            f"Macro-F1={row['macro_f1']}, ERR_final={row['err_final']}, "
            f"retained={row['retained_fraction']}, passed={row['smoke_passed']}, "
            f"reason={row['failure_reason'] or 'none'}"
        )
    if report["failed_methods"]:
        lines.extend(["", "## Failed"])
        for method, reasons in report["failed_methods"].items():
            lines.append(f"- {method}: {'; '.join(reasons)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", default="reports")
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()
    print(json.dumps(run_d9_5_baseline_smoke(args.methods, args.out, args.configs), indent=2))


if __name__ == "__main__":
    main()
