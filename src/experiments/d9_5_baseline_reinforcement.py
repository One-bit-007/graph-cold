"""D9.5 baseline reinforcement matrix and manuscript patch generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.result_sanity import check_results
from src.analysis.stat_tests import grouped_paired_summary
from src.experiments import d5, d5_baseline_expansion
from src.experiments.d9_5_baseline_common import (
    BASELINE_SOURCE,
    EXTRA_FIELDNAMES,
    FAITHFULNESS,
    IMPLEMENTATION_NOTES,
    METHODS,
    REINFORCED_FIELDNAMES,
    assert_original_rows_unchanged,
    make_baseline,
    original_expanded_with_extra,
    row_from_result,
    timed_baseline,
)
from src.experiments.d9_5_baseline_smoke import run_d9_5_baseline_smoke


RESULT_MAIN = Path("results/table_main_expanded.csv")
RESULT_REINFORCEMENT = Path("results/table_baseline_reinforcement.csv")
RESULT_REINFORCED = Path("results/table_main_reinforced.csv")
STATS_REINFORCED = Path("results/stat_tests_reinforced.json")
RUNTIME_REINFORCED = Path("results/runtime_reinforced.json")
REPORT_DIR = Path("reports/d9_5")
PACKAGE_DIR = Path("submission/cas_candidate_d9_5")
FORMAL_NEW_METHODS = ("Decoupling", "FINE-style")


def run_d9_5_baseline_reinforcement(
    methods: list[str] | tuple[str, ...] = METHODS,
    out: str | Path = "results",
    reports: str | Path = "reports",
    configs: str | Path = "configs",
) -> dict[str, Any]:
    out_dir = Path(out)
    reports_root = Path(reports)
    reports_d95 = reports_root / "d9_5"
    configs_path = Path(configs)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_d95.mkdir(parents=True, exist_ok=True)
    before_hashes = _locked_hashes()

    smoke = _load_or_run_smoke(methods, reports_root, configs_path)
    passed_methods = [method for method in methods if method in smoke.get("passed_methods", [])]
    baseline_rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    if passed_methods:
        existing = out_dir / "table_baseline_reinforcement.csv"
        if existing.exists():
            existing_frame = pd.read_csv(existing, keep_default_na=False)
            if _reinforcement_complete(existing_frame, passed_methods):
                baseline_rows = existing_frame.reindex(columns=REINFORCED_FIELDNAMES).to_dict(orient="records")
                runtime_records = _runtime_records(existing_frame)
        if not baseline_rows:
            baseline_rows, runtime_records = _run_matrix(passed_methods, configs_path, reports_root)

    baseline_frame = pd.DataFrame(baseline_rows, columns=REINFORCED_FIELDNAMES)
    baseline_frame.to_csv(out_dir / "table_baseline_reinforcement.csv", index=False)

    original = original_expanded_with_extra(out_dir / "table_main_expanded.csv")
    reinforced = pd.concat([original, baseline_frame], ignore_index=True).reindex(columns=REINFORCED_FIELDNAMES)
    reinforced.to_csv(out_dir / "table_main_reinforced.csv", index=False)

    runtime = _runtime_json(runtime_records, smoke, passed_methods)
    (out_dir / "runtime_reinforced.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    stats = grouped_paired_summary(reinforced, metric="macro_f1")
    (out_dir / "stat_tests_reinforced.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sanity = _reinforced_sanity(reinforced, passed_methods)
    (reports_d95 / "reinforced_sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    (reports_d95 / "reinforced_sanity_report.md").write_text(_sanity_markdown(sanity), encoding="utf-8")

    stat_validity = _stat_validity(stats)
    (reports_d95 / "reinforced_statistical_validity_report.json").write_text(json.dumps(stat_validity, indent=2), encoding="utf-8")
    (reports_d95 / "reinforced_statistical_validity_report.md").write_text(_stat_markdown(stat_validity), encoding="utf-8")

    status = "strong" if set(passed_methods) == set(FORMAL_NEW_METHODS) else "partial" if passed_methods else "failed"
    report = _reinforcement_report(reinforced, baseline_frame, smoke, sanity, stats, status, before_hashes)
    (reports_d95 / "baseline_reinforcement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports_d95 / "baseline_reinforcement_report.md").write_text(_report_markdown(report), encoding="utf-8")

    manuscript = {"manuscript_reinforced": False, "pdf_compiles": False}
    if sanity["passed"] and passed_methods:
        manuscript = _write_manuscript_patch(reinforced, baseline_frame, stats, passed_methods)

    final_audit = _final_audit(
        passed_methods=passed_methods,
        reinforcement_matrix_completed=bool(not baseline_frame.empty),
        sanity=sanity,
        manuscript=manuscript,
        before_hashes=before_hashes,
    )
    (reports_d95 / "d9_5_final_audit.json").write_text(json.dumps(final_audit, indent=2), encoding="utf-8")
    (reports_d95 / "d9_5_final_audit.md").write_text(_final_audit_markdown(final_audit), encoding="utf-8")
    return final_audit


def _load_or_run_smoke(methods: list[str] | tuple[str, ...], reports: Path, configs: Path) -> dict[str, Any]:
    path = reports / "d9_5/baseline_smoke_decoupling_fine.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if set(methods).issubset(set(report.get("methods_requested", methods))):
            return report
    return run_d9_5_baseline_smoke(methods, reports, configs)


def _run_matrix(methods: list[str], configs: Path, reports: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scale_policy = d5.write_scale_policy_report(reports)
    rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    for dataset_name in d5.FORMAL_DATASETS:
        for seed in d5.SEEDS:
            bundle = d5._load_formal_dataset(dataset_name, seed, configs, scale_policy)
            evidence = d5_baseline_expansion._evidence(bundle)
            graph_cache: dict[float, Any] = {}
            for spec in d5._noise_specs():
                noisy, flip = d5._inject_noise(bundle.dataset, spec, seed, graph_cache)
                for method in methods:
                    print(
                        f"[d9.5-matrix] {dataset_name} seed={seed} "
                        f"{spec['noise_type']} rate={spec['noise_rate']} beta={spec['graph_beta']} {method}",
                        flush=True,
                    )
                    baseline = make_baseline(method, seed, float(spec["noise_rate"]))
                    result, runtime_sec, memory_mb = timed_baseline(baseline, bundle, noisy, spec)
                    row = row_from_result(bundle, spec, seed, result, runtime_sec, memory_mb, evidence, flip, True)
                    rows.append(row)
                    runtime_records.append(
                        {
                            "dataset": row["dataset"],
                            "reported_as": row["reported_as"],
                            "noise_type": row["noise_type"],
                            "noise_rate": row["noise_rate"],
                            "graph_beta": row["graph_beta"],
                            "seed": row["seed"],
                            "method": row["method"],
                            "runtime_sec": row["runtime_sec"],
                            "memory_mb": row["memory_mb"],
                        }
                    )
    return rows, runtime_records


def _reinforcement_complete(frame: pd.DataFrame, methods: list[str]) -> bool:
    if frame.empty or not set(REINFORCED_FIELDNAMES).issubset(frame.columns):
        return False
    expected = len(d5.FORMAL_DATASETS) * len(d5.SEEDS) * len(d5._noise_specs()) * len(methods)
    if len(frame) != expected:
        return False
    if set(frame["method"].astype(str).unique()) != set(methods):
        return False
    if "smoke_passed" in frame and not frame["smoke_passed"].astype(str).str.lower().isin({"true", "1"}).all():
        return False
    numeric = frame.select_dtypes(include=[np.number])
    return bool(not numeric.isna().any().any() and np.isfinite(numeric.to_numpy(dtype=float)).all())


def _reinforced_sanity(frame: pd.DataFrame, passed_methods: list[str]) -> dict[str, Any]:
    report = check_results(frame)
    checks = report.setdefault("checks", {})
    checks["original_d5_expanded_rows_unchanged"] = assert_original_rows_unchanged(RESULT_MAIN, frame)
    checks["decoupling_rows_smoke_passed"] = _method_smoke_rows(frame, "Decoupling") if "Decoupling" in passed_methods else True
    checks["fine_style_rows_smoke_passed"] = _method_smoke_rows(frame, "FINE-style") if "FINE-style" in passed_methods else True
    checks["fine_not_misnamed"] = "FINE" not in set(frame["method"].astype(str))
    checks["co_teaching_lite_still_lite"] = "Co-Teaching-lite" in set(frame["method"].astype(str)) and "Co-Teaching" not in set(frame["method"].astype(str))
    checks["reinforcement_methods_only"] = set(frame[frame["baseline_source"].astype(str).str.len() > 0]["method"].astype(str)).issubset(set(FORMAL_NEW_METHODS))
    report["passed"] = bool(all(checks.values()))
    report["blocking_reasons"] = [name for name, ok in checks.items() if not ok]
    return report


def _method_smoke_rows(frame: pd.DataFrame, method: str) -> bool:
    part = frame[frame["method"] == method]
    if part.empty:
        return False
    return bool(part["smoke_passed"].astype(str).str.lower().isin({"true", "1"}).all())


def _stat_validity(stats: dict[str, Any]) -> dict[str, Any]:
    wanted = [
        "Graph-CoLD_vs_CoLD",
        "Graph-CoLD_vs_ablation_hard",
        "Graph-CoLD_vs_Noisy-Supervised",
        "Graph-CoLD_vs_Confident-Learning",
        "Graph-CoLD_vs_Co-Teaching-lite",
        "Graph-CoLD_vs_Decoupling",
        "Graph-CoLD_vs_FINE-style",
    ]
    comparisons = stats.get("comparisons", {})
    return {
        "stage": "D9.5 reinforced statistical validity",
        "paired_grouped_testing": True,
        "naive_pooled_test_used": False,
        "comparisons_present": {key: key in comparisons and not comparisons[key].get("skipped", False) for key in wanted},
        "comparisons": {key: comparisons.get(key, {"missing": True}) for key in wanted},
        "warnings": _stat_warnings(comparisons),
    }


def _stat_warnings(comparisons: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key, item in comparisons.items():
        if isinstance(item, dict) and item.get("extreme_p_value_warning"):
            warnings.append(f"{key}: extreme p-value")
        if isinstance(item, dict) and item.get("skipped"):
            warnings.append(f"{key}: skipped {item.get('reason')}")
    return warnings


def _write_manuscript_patch(
    reinforced: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    stats: dict[str, Any],
    passed_methods: list[str],
) -> dict[str, Any]:
    Path("tables").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("paper/elsevier/tables").mkdir(parents=True, exist_ok=True)
    Path("paper/elsevier/figures").mkdir(parents=True, exist_ok=True)

    table2 = _table_main_performance(reinforced)
    table3 = _table_high_noise(reinforced)
    table5 = _table_stats(stats, passed_methods=passed_methods)
    table2.to_csv("tables/table_2_main_performance_reinforced.csv", index=False)
    table3.to_csv("tables/table_3_high_noise_summary_reinforced.csv", index=False)
    table5.to_csv("tables/table_5_statistical_tests_reinforced.csv", index=False)
    _write_table_tex(Path("paper/elsevier/tables/table_2_main_summary_reinforced.tex"), _method_table(reinforced))
    _write_table_tex(Path("paper/elsevier/tables/table_3_high_noise_reinforced.tex"), table3[["Dataset", "Method", "Macro-F1 mean", "ERR mean", "Compression ratio mean", "Scenario count"]].rename(columns={"Macro-F1 mean": "Macro-F1", "ERR mean": "ERR", "Compression ratio mean": "Compression", "Scenario count": "Scenarios"}))
    _write_table_tex(Path("paper/elsevier/tables/table_5_statistical_tests_reinforced.tex"), table5[["Comparison", "Mean difference", "p-value", "Effect size", "n"]])
    _write_fig2(reinforced)
    _write_fig5(reinforced)
    for stem in ("fig2_macro_f1_vs_noise_rate_reinforced", "fig5_runtime_cost_reinforced"):
        for suffix in (".pdf", ".png"):
            shutil.copy2(Path("figures") / f"{stem}{suffix}", Path("paper/elsevier/figures") / f"{stem}{suffix}")

    _ensure_references(passed_methods)
    tex_path = _write_reinforced_tex(reinforced, stats, passed_methods)
    pdf_ok = _compile_reinforced_pdf(tex_path)
    package_ok = False
    if pdf_ok:
        _write_candidate_package(passed_methods)
        package_ok = True
    patch = {
        "stage": "D9.5 manuscript reinforcement patch",
        "passed_methods": passed_methods,
        "tables": [
            "tables/table_2_main_performance_reinforced.csv",
            "tables/table_3_high_noise_summary_reinforced.csv",
            "tables/table_5_statistical_tests_reinforced.csv",
        ],
        "figures": [
            "figures/fig2_macro_f1_vs_noise_rate_reinforced.pdf",
            "figures/fig2_macro_f1_vs_noise_rate_reinforced.png",
            "figures/fig5_runtime_cost_reinforced.pdf",
            "figures/fig5_runtime_cost_reinforced.png",
        ],
        "manuscript_reinforced": tex_path.exists(),
        "pdf_compiles": bool(pdf_ok),
        "candidate_package": str(PACKAGE_DIR),
        "candidate_package_ready": bool(package_ok),
        "fine_style_naming_correct": (
            "FINE-style" not in passed_methods
            or ("FINE-style" in tex_path.read_text(encoding="utf-8") and "full FINE" in tex_path.read_text(encoding="utf-8"))
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "manuscript_reinforcement_patch.md").write_text(_patch_markdown(patch, reinforced, stats), encoding="utf-8")
    (REPORT_DIR / "reviewer_risk_update_baselines.md").write_text(_risk_update(passed_methods), encoding="utf-8")
    return patch


def _write_reinforced_tex(reinforced: pd.DataFrame, stats: dict[str, Any], passed_methods: list[str]) -> Path:
    base_path = Path("paper/elsevier/graph_cold_cas_realdata.tex")
    out_path = Path("paper/elsevier/graph_cold_cas_realdata_reinforced.tex")
    text = base_path.read_text(encoding="utf-8")
    text = text.replace("results/table\\_main\\_expanded.csv", "results/table\\_main\\_reinforced.csv")
    text = text.replace("D5.5 matrix", "D9.5 reinforced matrix")
    text = text.replace("D5.5 result matrix", "D9.5 reinforced result matrix")
    text = text.replace("tables/table_2_main_summary.tex", "tables/table_2_main_summary_reinforced.tex")
    text = text.replace("tables/table_3_high_noise.tex", "tables/table_3_high_noise_reinforced.tex")
    text = text.replace("tables/table_5_statistical_tests.tex", "tables/table_5_statistical_tests_reinforced.tex")
    text = text.replace("figures/fig2_macro_f1_vs_noise_rate.pdf", "figures/fig2_macro_f1_vs_noise_rate_reinforced.pdf")
    text = text.replace("figures/fig5_runtime_cost.pdf", "figures/fig5_runtime_cost_reinforced.pdf")
    if "Decoupling" in passed_methods:
        text = text.replace(
            "Broader noisy\nlabel learning includes loss-correction methods",
            "Broader noisy\nlabel learning includes disagreement-update Decoupling \\cite{malach2017decoupling}, loss-correction methods",
        )
    if "FINE-style" in passed_methods:
        text = text.replace(
            "loss-correction methods",
            "FINE eigenvector filtering \\cite{kim2021fine}, loss-correction methods",
            1,
        )
    if "Decoupling baseline" not in text and "Decoupling" in passed_methods:
        baseline_sentence = (
            "Formal methods include Graph-CoLD, the aligned CoLD baseline, a hard-deletion\n"
            "ablation, noisy supervised learning, confidence learning, Co-Teaching-lite,\n"
            "and Decoupling. The Decoupling baseline is implemented as a tabular\n"
            "disagreement-update method under the same noisy-label protocol."
        )
        if "FINE-style" in passed_methods:
            baseline_sentence = (
                "Formal methods include Graph-CoLD, the aligned CoLD baseline, a hard-deletion\n"
                "ablation, noisy supervised learning, confidence learning, Co-Teaching-lite,\n"
                "Decoupling, and FINE-style. The Decoupling baseline is implemented as a\n"
                "tabular disagreement-update method under the same noisy-label protocol.\n"
                "FINE-style is inspired by FINE's representation/eigenvector filtering\n"
                "mechanism and is not claimed as a full FINE reproduction."
            )
        text = text.replace(
            "Formal methods include Graph-CoLD, the aligned CoLD baseline, a hard-deletion\nablation, noisy supervised learning, confidence learning, and Co-Teaching-lite.",
            baseline_sentence,
        )
    excluded = ["full FINE", "MCRe", "MORSE", "Flash", "Argus", "full Co-Teaching"]
    if "Decoupling" not in passed_methods:
        excluded.append("Decoupling")
    text = text.replace(
        "Methods\nexcluded from formal comparison are FINE, MCRe, MORSE, Flash, Argus, Decoupling, full Co-Teaching; each is omitted because it lacks an independently\nsmoke-passed real-data implementation in this repository.",
        "Methods excluded from formal comparison after D9.5 are "
        + ", ".join(excluded)
        + "; each is omitted because it lacks an independently smoke-passed real-data implementation in this repository.",
    )
    dec = stats["comparisons"].get("Graph-CoLD_vs_Decoupling", {})
    fine = stats["comparisons"].get("Graph-CoLD_vs_FINE-style", {})
    pieces = []
    if "Decoupling" in passed_methods:
        pieces.append(
            f"Graph-CoLD versus Decoupling has mean difference {_pp(dec)} "
            f"(p={_p(dec)}, dz={_dz(dec)}, n={dec.get('n_pairs', 'NA')})."
        )
    if "FINE-style" in passed_methods:
        pieces.append(
            f"Graph-CoLD versus FINE-style has mean difference {_pp(fine)} "
            f"(p={_p(fine)}, dz={_dz(fine)}, n={fine.get('n_pairs', 'NA')})."
        )
    insert = (
        "\nThe D9.5 reinforcement adds "
        + ", ".join(passed_methods)
        + " as smoke-passed real-data baseline"
        + ("s. " if len(passed_methods) != 1 else ". ")
        + " ".join(pieces)
        + " These rows strengthen baseline coverage without claiming state-of-the-art breadth.\n"
    )
    text = text.replace("\\subsection{RQ2: What happens under high noise?}", insert + "\n\\subsection{RQ2: What happens under high noise?}")
    text = text.replace(
        "Co-Teaching-lite is a lightweight approximation, and excluded\nbaselines need faithful implementations before broader comparison.",
        "Co-Teaching-lite is a lightweight approximation. Excluded baselines need\n"
        "faithful implementations before broader comparison.",
    )
    text = text.replace("\\bibliography{references}", "\\bibliography{references}")
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _ensure_references(passed_methods: list[str]) -> None:
    path = Path("paper/elsevier/references.bib")
    text = path.read_text(encoding="utf-8")
    additions = []
    if "Decoupling" in passed_methods and "malach2017decoupling" not in text:
        additions.append(
            r"""@inproceedings{malach2017decoupling,
  title={Decoupling "when to update" from "how to update"},
  author={Malach, Eran and Shalev-Shwartz, Shai},
  booktitle={Advances in Neural Information Processing Systems},
  pages={960--970},
  year={2017},
  url={https://papers.neurips.cc/paper/6697-decoupling-when-to-update-from-how-to-update}
}
"""
        )
    if "FINE-style" in passed_methods and "kim2021fine" not in text:
        additions.append(
            r"""@inproceedings{kim2021fine,
  title={{FINE} Samples for Learning with Noisy Labels},
  author={Kim, Taehyeon and Ko, Jongwoo and Cho, Sangwook and Choi, Jinhwan and Yun, Se-Young},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={24137--24149},
  year={2021},
  url={https://proceedings.neurips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html}
}
"""
        )
    if additions:
        path.write_text(text.rstrip() + "\n\n" + "\n".join(additions), encoding="utf-8")


def _compile_reinforced_pdf(tex_path: Path) -> bool:
    root = tex_path.parent
    stem = tex_path.stem
    commands = [
        ["pdflatex", "--disable-installer", "-halt-on-error", "-interaction=nonstopmode", tex_path.name],
        ["bibtex", stem],
        ["pdflatex", "--disable-installer", "-halt-on-error", "-interaction=nonstopmode", tex_path.name],
        ["pdflatex", "--disable-installer", "-halt-on-error", "-interaction=nonstopmode", tex_path.name],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for suffix in (".aux", ".bbl", ".blg", ".log", ".out"):
        temp = root / f"{stem}{suffix}"
        if temp.exists():
            temp.unlink()
    return (root / f"{stem}.pdf").exists()


def _write_candidate_package(passed_methods: list[str]) -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    author_tex = Path("paper/elsevier/graph_cold_cas_realdata_reinforced.tex")
    author_pdf = Path("paper/elsevier/graph_cold_cas_realdata_reinforced.pdf")
    review_tex = PACKAGE_DIR / "manuscript_anonymous_review_reinforced.tex"
    review_pdf = PACKAGE_DIR / "manuscript_anonymous_review_reinforced.pdf"
    shutil.copy2(author_tex, PACKAGE_DIR / "manuscript_author_version_reinforced.tex")
    shutil.copy2(author_pdf, PACKAGE_DIR / "manuscript_author_version_reinforced.pdf")
    review_text = author_tex.read_text(encoding="utf-8")
    review_text = review_text.replace(r"\author{Graph-CoLD Project Team}", r"\author{Anonymous Authors}")
    review_text = review_text.replace(r"\address{Computers \& Security submission candidate v1.0}", r"\address{Manuscript under anonymous review}")
    review_text = re.sub(
        r"\\section\*\{Acknowledgements\}.*?\\section\*\{Declaration of competing interest\}",
        lambda _: "\\section*{Acknowledgements}\nAcknowledgements are omitted for anonymous review.\n\n\\section*{Declaration of competing interest}",
        review_text,
        flags=re.S,
    )
    review_tex.write_text(review_text, encoding="utf-8")
    temp_review_tex = author_tex.parent / "graph_cold_cas_realdata_reinforced_anonymous.tex"
    temp_review_pdf = temp_review_tex.with_suffix(".pdf")
    temp_review_tex.write_text(review_text, encoding="utf-8")
    try:
        _compile_reinforced_pdf(temp_review_tex)
        shutil.copy2(temp_review_pdf, review_pdf)
    finally:
        if temp_review_tex.exists():
            temp_review_tex.unlink()
        if temp_review_pdf.exists():
            temp_review_pdf.unlink()
    summary = f"""# D9.5 Baseline Reinforcement Summary

- submission_ready: false
- human_confirmation_required: true
- included new methods: {', '.join(passed_methods) or 'none'}
- FINE-style naming: explicitly not full FINE
- Existing D5/D5.5 result files were not overwritten.
"""
    (PACKAGE_DIR / "baseline_reinforcement_summary.md").write_text(summary, encoding="utf-8")
    manifest = {
        "stage": "D9.5 baseline reinforcement package",
        "submission_ready": False,
        "human_confirmation_required": True,
        "included_new_methods": passed_methods,
        "files": _package_files(PACKAGE_DIR),
    }
    (PACKAGE_DIR / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _package_files(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root)).replace("\\", "/")] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return out


def _table_main_performance(main: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Noise type": grouped["noise_type"],
            "Noise rate": grouped["noise_rate"].map(lambda v: f"{float(v):.1f}"),
            "Graph beta": grouped["graph_beta"].astype(str),
            "Method": grouped["method"],
            "Macro-F1 mean +/- std": _fmt(grouped, "macro_f1"),
            "FPR mean +/- std": _fmt(grouped, "fpr"),
            "FNR mean +/- std": _fmt(grouped, "fnr"),
            "ERR mean +/- std": _fmt(grouped, "err"),
            "Compression ratio mean +/- std": _fmt(grouped, "compression"),
            "Runtime mean +/- std (s)": _fmt(grouped, "runtime", digits=2),
        }
    )


def _method_table(main: pd.DataFrame) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(["Graph-CoLD", "CoLD", "ablation_hard", "Noisy-Supervised", "Confident-Learning", "Co-Teaching-lite", "Decoupling", "FINE-style"])}
    grouped = (
        main.groupby(["reported_as", "method"], dropna=False)
        .agg(
            macro=("macro_f1", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            err=("err_final", "mean"),
            compression=("compression_ratio", "mean"),
            runtime=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    grouped["dataset_order"] = grouped["reported_as"].map({"CICIDS-2017": 0, "CESNET-TLS-Year22": 1})
    grouped["method_order"] = grouped["method"].map(order)
    grouped = grouped.sort_values(["dataset_order", "method_order"])
    return pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Method": grouped["method"],
            "Macro-F1": grouped["macro"].map(lambda v: f"{v:.4f}"),
            "FPR": grouped["fpr"].map(lambda v: f"{v:.4f}"),
            "FNR": grouped["fnr"].map(lambda v: f"{v:.4f}"),
            "ERR": grouped["err"].map(lambda v: f"{v:.4f}"),
            "Compression": grouped["compression"].map(lambda v: f"{v:.4f}"),
            "Runtime (s)": grouped["runtime"].map(lambda v: f"{v:.2f}"),
        }
    )


def _table_high_noise(main: pd.DataFrame) -> pd.DataFrame:
    rates = pd.to_numeric(main["noise_rate"], errors="coerce")
    high = main[(rates >= 0.4) & main["noise_type"].isin(["symmetric", "asymmetric", "graph_consistency"])]
    grouped = (
        high.groupby(["reported_as", "method"], dropna=False)
        .agg(
            macro=("macro_f1", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            err=("err_final", "mean"),
            compression=("compression_ratio", "mean"),
            count=("macro_f1", "size"),
        )
        .reset_index()
    )
    return pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Method": grouped["method"],
            "Macro-F1 mean": grouped["macro"].map(lambda v: f"{v:.4f}"),
            "FPR mean": grouped["fpr"].map(lambda v: f"{v:.4f}"),
            "FNR mean": grouped["fnr"].map(lambda v: f"{v:.4f}"),
            "ERR mean": grouped["err"].map(lambda v: f"{v:.4f}"),
            "Compression ratio mean": grouped["compression"].map(lambda v: f"{v:.4f}"),
            "Scenario count": grouped["count"].astype(int),
        }
    )


def _table_stats(stats: dict[str, Any], passed_methods: list[str] | None = None) -> pd.DataFrame:
    wanted = [
        ("Graph-CoLD_vs_CoLD", "Graph-CoLD vs CoLD"),
        ("Graph-CoLD_vs_ablation_hard", "Graph-CoLD vs ablation_hard"),
        ("Graph-CoLD_vs_Noisy-Supervised", "Graph-CoLD vs Noisy-Supervised"),
        ("Graph-CoLD_vs_Confident-Learning", "Graph-CoLD vs Confident-Learning"),
        ("Graph-CoLD_vs_Co-Teaching-lite", "Graph-CoLD vs Co-Teaching-lite"),
    ]
    passed = set(passed_methods or FORMAL_NEW_METHODS)
    if "Decoupling" in passed:
        wanted.append(("Graph-CoLD_vs_Decoupling", "Graph-CoLD vs Decoupling"))
    if "FINE-style" in passed:
        wanted.append(("Graph-CoLD_vs_FINE-style", "Graph-CoLD vs FINE-style"))
    rows = []
    comparisons = stats.get("comparisons", {})
    for key, label in wanted:
        item = comparisons.get(key, {})
        rows.append(
            {
                "Comparison": label,
                "Mean difference": _pp(item),
                "p-value": _p(item),
                "Effect size": _dz(item),
                "n": item.get("n_pairs", "NA"),
                "Test type": "paired grouped t-test, greater alternative",
            }
        )
    return pd.DataFrame(rows)


def _write_fig2(main: pd.DataFrame) -> None:
    methods = ["Graph-CoLD", "CoLD", "Decoupling", "FINE-style", "Confident-Learning", "Co-Teaching-lite"]
    colors = {"Graph-CoLD": "#5477C4", "CoLD": "#F0986E", "Decoupling": "#5AAE8A", "FINE-style": "#9467BD", "Confident-Learning": "#71B436", "Co-Teaching-lite": "#BD569B"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharey=True)
    panels = [("CICIDS-2017", "symmetric", "none", axes[0, 0]), ("CICIDS-2017", "graph_consistency", "0.6", axes[0, 1]), ("CESNET-TLS-Year22", "symmetric", "none", axes[1, 0]), ("CESNET-TLS-Year22", "graph_consistency", "0.6", axes[1, 1])]
    for dataset, noise, beta, ax in panels:
        part = main[(main["reported_as"] == dataset) & (main["noise_type"] == noise) & (main["method"].isin(methods))].copy()
        if beta != "none":
            part = part[part["graph_beta"].astype(str) == beta]
        summary = part.groupby(["method", "noise_rate"], as_index=False).agg(macro=("macro_f1", "mean"))
        for method in methods:
            p = summary[summary["method"] == method].sort_values("noise_rate")
            if p.empty:
                continue
            ax.plot(p["noise_rate"], p["macro"], marker="o", linewidth=1.25, markersize=3.8, label=method, color=colors[method])
        ax.set_title(f"{dataset}: {noise.replace('_', '-')}", fontsize=9)
        ax.set_xlabel("Noise rate")
        ax.set_ylabel("Macro-F1")
        ax.grid(True, color="#E6E8F0", linewidth=0.7)
        ax.set_ylim(0.2, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Fig2 reinforced. Macro-F1 under label noise", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("figures/fig2_macro_f1_vs_noise_rate_reinforced.png", dpi=220, bbox_inches="tight")
    fig.savefig("figures/fig2_macro_f1_vs_noise_rate_reinforced.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_fig5(main: pd.DataFrame) -> None:
    methods = ["Graph-CoLD", "CoLD", "Noisy-Supervised", "Confident-Learning", "Co-Teaching-lite", "Decoupling", "FINE-style"]
    summary = main[main["method"].isin(methods)].groupby("method", as_index=False).agg(runtime=("runtime_sec", "mean"), memory=("memory_mb", "mean"))
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(methods)})
    summary = summary.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    axes[0].bar(summary["method"], summary["runtime"], color="#5477C4")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Runtime")
    axes[1].bar(summary["method"], summary["memory"], color="#F0986E")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_title("Memory")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", color="#E6E8F0", linewidth=0.7)
    fig.suptitle("Fig5 reinforced. Runtime and memory cost by method", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("figures/fig5_runtime_cost_reinforced.png", dpi=220, bbox_inches="tight")
    fig.savefig("figures/fig5_runtime_cost_reinforced.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_table_tex(path: Path, frame: pd.DataFrame) -> None:
    align = "l" * len(frame.columns)
    lines = [r"\small", f"\\begin{{tabular}}{{{align}}}", r"\hline"]
    lines.append(" & ".join(_tex_escape(str(col)) for col in frame.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in frame.iterrows():
        lines.append(" & ".join(_tex_escape(str(value)) for value in row.tolist()) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _runtime_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cols = ["dataset", "reported_as", "noise_type", "noise_rate", "graph_beta", "seed", "method", "runtime_sec", "memory_mb"]
    return frame[[col for col in cols if col in frame.columns]].to_dict(orient="records")


def _runtime_json(records: list[dict[str, Any]], smoke: dict[str, Any], passed_methods: list[str]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    summary = {}
    if not frame.empty:
        grouped = frame.groupby("method")[["runtime_sec", "memory_mb"]].agg(["mean", "std", "max"])
        for method, row in grouped.iterrows():
            summary[str(method)] = {f"{metric}_{stat}": float(value) if pd.notna(value) else 0.0 for (metric, stat), value in row.items()}
    return {"records": records, "summary": summary, "smoke_passed_methods": passed_methods, "smoke_report": smoke}


def _reinforcement_report(
    reinforced: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    smoke: dict[str, Any],
    sanity: dict[str, Any],
    stats: dict[str, Any],
    status: str,
    before_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": "D9.5 baseline reinforcement",
        "baseline_reinforcement_status": status,
        "rows": {"table_baseline_reinforcement": int(len(baseline_frame)), "table_main_reinforced": int(len(reinforced))},
        "datasets": sorted(reinforced["dataset"].astype(str).unique().tolist()),
        "methods": sorted(reinforced["method"].astype(str).unique().tolist()),
        "included_new_methods": sorted(baseline_frame["method"].astype(str).unique().tolist()) if not baseline_frame.empty else [],
        "smoke": smoke,
        "sanity": sanity,
        "statistics": _stat_validity(stats),
        "original_hashes_before": before_hashes,
        "original_hashes_after": _locked_hashes(),
        "original_results_unchanged": before_hashes == _locked_hashes(),
    }


def _final_audit(
    passed_methods: list[str],
    reinforcement_matrix_completed: bool,
    sanity: dict[str, Any],
    manuscript: dict[str, Any],
    before_hashes: dict[str, str],
) -> dict[str, Any]:
    after_hashes = _locked_hashes()
    return {
        "decoupling_implemented": True,
        "fine_style_implemented": True,
        "decoupling_smoke_passed": "Decoupling" in passed_methods,
        "fine_style_smoke_passed": "FINE-style" in passed_methods,
        "reinforcement_matrix_completed": bool(reinforcement_matrix_completed),
        "table_main_reinforced_exists": RESULT_REINFORCED.exists(),
        "graphcold_existing_rows_unchanged": bool(sanity["checks"].get("original_d5_expanded_rows_unchanged", False)),
        "manuscript_reinforced": bool(manuscript.get("manuscript_reinforced", False)),
        "pdf_compiles": bool(manuscript.get("pdf_compiles", False)),
        "no_fake_baselines": bool(sanity["checks"].get("no_fake_baseline_rows", False)),
        "fine_not_misnamed": bool(sanity["checks"].get("fine_not_misnamed", False)),
        "submission_ready": False,
        "human_confirmation_required": True,
        "results_table_main_expanded_unchanged": before_hashes == after_hashes,
        "blocking_items": [
            "Human author/funding/COI confirmation still required.",
            "submission_ready intentionally remains false.",
        ],
    }


def _locked_hashes() -> dict[str, str]:
    paths = [
        Path("results/table_main_expanded.csv"),
        Path("results/table_baseline_expansion.csv"),
        Path("results/stat_tests_baseline_expansion.json"),
    ]
    return {str(path).replace("\\", "/"): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(frame: pd.DataFrame, prefix: str, digits: int = 4) -> pd.Series:
    fmt = f"{{:.{digits}f}} +/- {{:.{digits}f}}"
    return frame.apply(lambda row: fmt.format(row[f"{prefix}_mean"], row[f"{prefix}_std"]), axis=1)


def _std(series: pd.Series) -> float:
    value = series.std(ddof=1)
    return 0.0 if pd.isna(value) else float(value)


def _pp(item: dict[str, Any]) -> str:
    if not item or item.get("skipped"):
        return "NA"
    return f"{float(item.get('mean_diff', 0.0)) * 100.0:.2f} pp"


def _p(item: dict[str, Any]) -> str:
    if not item or item.get("skipped") or item.get("p_value") is None:
        return "NA"
    return f"{float(item['p_value']):.2e}"


def _dz(item: dict[str, Any]) -> str:
    if not item or item.get("skipped"):
        return "NA"
    value = item.get("effect_size_cohen_dz")
    if value is None:
        return "NA"
    return "inf" if value == float("inf") else f"{float(value):.3f}"


def _tex_escape(text: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _sanity_markdown(report: dict[str, Any]) -> str:
    lines = ["# D9.5 Reinforced Sanity Report", "", f"- Passed: {report['passed']}", "", "## Checks"]
    lines.extend([f"- {key}: {value}" for key, value in report["checks"].items()])
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {reason}" for reason in report["blocking_reasons"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _stat_markdown(report: dict[str, Any]) -> str:
    lines = ["# D9.5 Reinforced Statistical Validity Report", "", f"- Paired grouped testing: {report['paired_grouped_testing']}", "", "## Comparisons"]
    for key, present in report["comparisons_present"].items():
        item = report["comparisons"].get(key, {})
        lines.append(f"- {key}: present={present}, diff={_pp(item)}, p={_p(item)}, dz={_dz(item)}")
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {w}" for w in report["warnings"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D9.5 Baseline Reinforcement Report",
        "",
        f"- Status: {report['baseline_reinforcement_status']}",
        f"- table_baseline_reinforcement rows: {report['rows']['table_baseline_reinforcement']}",
        f"- table_main_reinforced rows: {report['rows']['table_main_reinforced']}",
        f"- Original results unchanged: {report['original_results_unchanged']}",
        f"- Included new methods: {', '.join(report['included_new_methods']) or 'none'}",
        "",
        "## Statistics",
    ]
    for key, item in report["statistics"]["comparisons"].items():
        lines.append(f"- {key}: diff={_pp(item)}, p={_p(item)}, dz={_dz(item)}")
    lines.append("")
    return "\n".join(lines)


def _patch_markdown(patch: dict[str, Any], reinforced: pd.DataFrame, stats: dict[str, Any]) -> str:
    lines = [
        "# D9.5 Manuscript Reinforcement Patch",
        "",
        f"- Reinforced manuscript: {patch['manuscript_reinforced']}",
        f"- PDF compiles: {patch['pdf_compiles']}",
        f"- Candidate package ready: {patch['candidate_package_ready']}",
        "",
        "## Sections Patched",
        "- Related Work: Decoupling and FINE/eigenvector filtering.",
        "- Experimental Setup: Decoupling and FINE-style baseline scope.",
        "- Results: Graph-CoLD vs Decoupling and Graph-CoLD vs FINE-style.",
        "- Limitations: FINE-style is not full FINE.",
        "",
    ]
    return "\n".join(lines)


def _risk_update(passed_methods: list[str]) -> str:
    return f"""# Reviewer Risk Update - D9.5 Baselines

- Decoupling included: {"Decoupling" in passed_methods}
- FINE-style included: {"FINE-style" in passed_methods}
- Full FINE remains excluded; manuscript uses the FINE-style name.
- This strengthens baseline breadth but still should not be framed as state-of-the-art coverage.
- submission_ready remains false pending human confirmation.
"""


def _final_audit_markdown(audit: dict[str, Any]) -> str:
    return "# D9.5 Final Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()
    print(json.dumps(run_d9_5_baseline_reinforcement(args.methods, args.out, args.reports, args.configs), indent=2))


if __name__ == "__main__":
    main()
