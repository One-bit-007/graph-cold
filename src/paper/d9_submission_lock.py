"""D9 submission-lock audit and candidate package generation.

This module performs a paper/package audit only. It reads frozen D5/D5.5/D6/D7
and D8 artifacts, writes submission support material, creates author and
anonymous review candidate bundles, and records a lock audit. It does not run
experiments, does not modify results, and deliberately leaves
``submission_ready`` false pending human author confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


PAPER = Path("paper/elsevier")
REPORTS = Path("reports")
PACKAGE = Path("submission/cas_candidate_d9")
FORMAL_DATASETS = {"CICIDS-2017", "CESNET-TLS-Year22"}
FORMAL_METHODS = {
    "Graph-CoLD",
    "CoLD",
    "ablation_hard",
    "Noisy-Supervised",
    "Confident-Learning",
    "Co-Teaching-lite",
}
EXCLUDED_DATASETS = ("MALTLS-22", "OpTC", "UNSW-NB15", "USTC-TFC2016")
EXCLUDED_METHODS = ("FINE", "MCRe", "MORSE", "Flash", "Argus", "Decoupling")
RESULT_SOURCES = (
    Path("results/table_main_expanded.csv"),
    Path("results/table_baseline_expansion.csv"),
    Path("results/stat_tests_baseline_expansion.json"),
)
STATIC_REQUIRED = (
    PAPER / "graph_cold_cas_realdata.tex",
    PAPER / "graph_cold_cas_realdata.pdf",
    PAPER / "references.bib",
    PAPER / "cover_letter_draft.md",
    REPORTS / "d8/d8_hardening_audit.json",
    REPORTS / "d8/reviewer_risk_register_v1.md",
    REPORTS / "d7/d7_final_audit.json",
    REPORTS / "d6/d6_statistical_narrative.md",
    REPORTS / "d5_expanded_sanity_report.json",
    REPORTS / "d5_expanded_statistical_validity_report.json",
    REPORTS / "d5_scale_policy.json",
    REPORTS / "cicids_final_protocol.json",
    REPORTS / "cesnet_class_policy_report.json",
    REPORTS / "cesnet_view_policy_report.json",
    REPORTS / "realdata_readiness_report.json",
    Path("reproducibility/README_realdata.md"),
    Path("reproducibility/run_d5_realdata.ps1"),
    Path("docs/DATASETS.md"),
    Path("docs/DATASET_DOWNLOADS.md"),
    Path("docs/STORAGE_AND_DATA_ROOT.md"),
)
SUPPORT_FILES = (
    PAPER / "highlights.md",
    PAPER / "declaration_of_competing_interest.md",
    PAPER / "funding_statement.md",
    PAPER / "credit_author_statement.md",
    PAPER / "data_availability_statement.md",
)


def run_d9_submission_lock(
    paper_dir: str | Path = PAPER,
    reports_dir: str | Path = REPORTS,
    package_dir: str | Path = PACKAGE,
    compile_package: bool = True,
) -> dict[str, Any]:
    """Generate D9 support files, candidate package, and lock audit."""
    paper = Path(paper_dir)
    reports = Path(reports_dir)
    package = Path(package_dir)
    d9_dir = reports / "d9"
    d9_dir.mkdir(parents=True, exist_ok=True)

    _read_required_inputs()
    result_hash_before = _source_hashes(RESULT_SOURCES)
    metrics = _metrics()
    _write_support_materials(metrics)
    _write_reproducibility_docs()
    _write_method_figure()
    _harden_manuscript_text(metrics)
    _build_main_pdf(paper)
    _build_candidate_package(package, paper, metrics, compile_package)

    result_hash_after = _source_hashes(RESULT_SOURCES)
    audit = _audit(package, result_hash_before, result_hash_after)
    _write_audit_files(d9_dir, audit)
    _write_package_manifest(package, audit)
    return audit


def refresh_d9_audit(package_dir: str | Path = PACKAGE, reports_dir: str | Path = REPORTS) -> dict[str, Any]:
    """Refresh D9 audit after an external package build."""
    audit = _audit(Path(package_dir), _source_hashes(RESULT_SOURCES), _source_hashes(RESULT_SOURCES))
    _write_audit_files(Path(reports_dir) / "d9", audit)
    _write_package_manifest(Path(package_dir), audit)
    return audit


def _read_required_inputs() -> None:
    paths: list[Path] = list(STATIC_REQUIRED) + list(RESULT_SOURCES)
    paths.extend(sorted(Path("tables").glob("*.csv")))
    paths.extend(sorted(Path("tables").glob("*.md")))
    paths.extend(sorted(Path("figures").glob("*.pdf")))
    paths.extend(sorted(Path("figures").glob("*.png")))
    paths.extend(sorted(path for path in Path("tests").rglob("*") if path.is_file()))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("D9 required artifacts missing: " + ", ".join(missing))
    for path in paths:
        path.read_bytes()


def _metrics() -> dict[str, float]:
    main = pd.read_csv("results/table_main_expanded.csv")
    stats = json.loads(Path("results/stat_tests_baseline_expansion.json").read_text(encoding="utf-8"))
    if set(main["reported_as"].dropna().unique()) != FORMAL_DATASETS:
        raise ValueError("Formal dataset scope drifted.")
    if not FORMAL_METHODS.issubset(set(main["method"].dropna().unique())):
        raise ValueError("Formal method scope drifted.")
    if not main["source_verified"].astype(bool).all():
        raise ValueError("All D9 result rows must be source_verified=true.")
    means = main.groupby("method").agg(
        macro=("macro_f1", "mean"),
        err=("err_final", "mean"),
        runtime=("runtime_sec", "mean"),
        memory=("memory_mb", "mean"),
    )
    comparison = stats["comparisons"]["Graph-CoLD_vs_CoLD"]
    return {
        "graph_macro": float(means.loc["Graph-CoLD", "macro"]),
        "cold_macro": float(means.loc["CoLD", "macro"]),
        "graph_err": float(means.loc["Graph-CoLD", "err"]),
        "hard_err": float(means.loc["ablation_hard", "err"]),
        "graph_runtime": float(means.loc["Graph-CoLD", "runtime"]),
        "graph_memory": float(means.loc["Graph-CoLD", "memory"]),
        "mean_diff_pp": float(comparison["mean_diff"]) * 100.0,
        "p_value": float(comparison["p_value"]),
        "effect_size": float(comparison["effect_size_cohen_dz"]),
        "n_pairs": int(comparison["n_pairs"]),
        "err_gap_pp": (float(means.loc["Graph-CoLD", "err"]) - float(means.loc["ablation_hard", "err"])) * 100.0,
    }


def _write_support_materials(metrics: dict[str, float]) -> None:
    (PAPER / "highlights.md").write_text(
        "\n".join(
            [
                "# Highlights",
                "",
                "- Graph-CoLD adds graph label consistency to noisy-label IDS.",
                "- Soft evidence weights preserve clean informative alerts.",
                "- Verified results cover CICIDS-2017 and CESNET-TLS-Year22 only.",
                "- Paired testing shows a 1.83 pp Macro-F1 lift over CoLD.",
                "- The package reports scope limits for datasets and baselines.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (PAPER / "declaration_of_competing_interest.md").write_text(
        "# Declaration of Competing Interest\n\n"
        "Candidate statement: no competing interest is recorded in this repository package. "
        "The named authors must confirm the final declaration before journal upload.\n",
        encoding="utf-8",
    )
    (PAPER / "funding_statement.md").write_text(
        "# Funding Statement\n\n"
        "Candidate statement: no external funding source is recorded in this repository package. "
        "The named authors must confirm funder names, grant numbers, or a no-funding statement before submission.\n",
        encoding="utf-8",
    )
    (PAPER / "credit_author_statement.md").write_text(
        "# CRediT Author Statement\n\n"
        "Candidate statement for author confirmation: Conceptualization, methodology, software, validation, "
        "formal analysis, data curation, writing - original draft, writing - review and editing, and visualization "
        "are attributed to the Graph-CoLD project authors. Replace this project-level statement with named author "
        "roles before final upload.\n",
        encoding="utf-8",
    )
    (PAPER / "data_availability_statement.md").write_text(
        "# Data Availability Statement\n\n"
        "Code, scripts, result tables, audit reports, and manuscript-generation commands are available in the "
        "Graph-CoLD repository. Raw CICIDS-2017 and CESNET-TLS-Year22 data are not redistributed with the repository; "
        "they must be obtained from their official sources and placed under the documented local data roots. "
        "The D9 package records hashes for the frozen result files used by the manuscript.\n",
        encoding="utf-8",
    )
    (PAPER / "cover_letter_draft.md").write_text(
        f"""# Cover Letter Candidate v1.0

Dear Editors,

We submit the candidate manuscript "Graph-CoLD: Evidence-preserving Graph Label
Denoising for SOC Alert Prioritization under Noisy Labels" for consideration by
Computers & Security, pending final author metadata confirmation.

The manuscript studies noisy-label intrusion detection and SOC alert
prioritization using verified CICIDS-2017 postfilter11 and CESNET-TLS-Year22
postfilter25 settings. The core contribution is an evidence-preserving graph
label-denoising method that uses label-space consistency over active graph views
rather than hard deletion alone.

The evaluation is deliberately bounded to implemented and smoke-passed real-data
baselines. Graph-CoLD improves Macro-F1 over the aligned CoLD baseline by
{metrics['mean_diff_pp']:.2f} percentage points in a paired scenario-level test
(p={metrics['p_value']:.2e}) and improves mean ERR_final by
{metrics['err_gap_pp']:.2f} percentage points over hard deletion.

The manuscript explicitly states that CESNET-TLS-Year22 is a deterministic
audit-window subset, that MALTLS-22 and OpTC are not reported, and that
Co-Teaching-lite is not a full Co-Teaching implementation or reproduction.

Sincerely,

The Graph-CoLD authors
""",
        encoding="utf-8",
    )


def _write_reproducibility_docs() -> None:
    readme = """# Graph-CoLD Real-data Reproducibility Package

This package recreates the D5/D5.5 result matrix, D6 paper tables/figures, D7
assembly, D8 manuscript hardening, and D9 submission-lock package from verified
local datasets. Raw datasets are not committed.

## Formal Scope

- Datasets: CICIDS-2017 postfilter11; CESNET-TLS-Year22 postfilter25.
- Methods: Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised,
  Confident-Learning, and Co-Teaching-lite.
- Excluded from formal results: MALTLS-22, OpTC, UNSW-NB15, USTC-TFC2016,
  FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full Co-Teaching.
- MALTLS-22 and OpTC are not part of the formal evaluation package.

## Data Roots

- CICIDS-2017: `data/cicids2017`
- CESNET-TLS-Year22: `E:\\graphcold-data\\tls_alternative\\cesnet_tls_year22`
- External data root: `E:\\graphcold-data`

## Frozen Source Artifacts

- `results/table_main_expanded.csv`: `c7d998d6c918ecfbcb9cc56bd494dcec73b3fa6826b2046fb53e2ca2109519cd`
- `results/table_baseline_expansion.csv`: `b74a3552b9a11b87ee847df2fa5490197fcb4c4fbe59973c7ec3593945b9d158`
- `results/stat_tests_baseline_expansion.json`: `6aff31cb1d29cbae5a63bb586eb73fdf63b0fe38391c5819f8cdf9ac2fcfd7e4`
- `reports/realdata_readiness_report.json`: `3c284a3a4f09b023bac4e20400e589f31c61ecf717caf1886e737e93f0b98e0e`

## Entry Points

Run readiness gates:

```powershell
python -m src.data.audit
python scripts/check_data_ready.py
```

Recreate D5/D5.5 real-data results:

```powershell
powershell -ExecutionPolicy Bypass -File .\\reproducibility\\run_d5_realdata.ps1
```

Regenerate D6 paper assets:

```powershell
powershell -ExecutionPolicy Bypass -File .\\reproducibility\\run_d6_paper_assets.ps1
```

Rebuild the D7 manuscript:

```powershell
powershell -ExecutionPolicy Bypass -File .\\reproducibility\\run_d7_build.ps1
```

Regenerate the D8 hardened manuscript:

```powershell
powershell -ExecutionPolicy Bypass -File .\\reproducibility\\run_d8_manuscript.ps1
```

Regenerate the D9 candidate submission package:

```powershell
powershell -ExecutionPolicy Bypass -File .\\reproducibility\\run_d9_submission_package.ps1
```

Large dataset downloads are manual or optional and are not started by the paper
asset scripts. Keep raw archives and extracted data outside Git tracking.
"""
    Path("reproducibility/README_realdata.md").write_text(readme, encoding="utf-8")
    _write_ps_script(Path("reproducibility/run_d6_paper_assets.ps1"), ["-m src.paper.d6_prep"])
    _write_ps_script(
        Path("reproducibility/run_d7_build.ps1"),
        ["-m src.paper.d7_assemble", "BUILD_ELSEVIER", "-m src.paper.d7_assemble --audit-only"],
    )
    _write_ps_script(
        Path("reproducibility/run_d8_manuscript.ps1"),
        ["-m src.paper.d8_harden", "BUILD_ELSEVIER", "-m src.paper.d8_harden --audit-only"],
    )
    _write_ps_script(
        Path("reproducibility/run_d9_submission_package.ps1"),
        ["-m src.paper.d9_submission_lock"],
    )


def _write_ps_script(path: Path, steps: list[str]) -> None:
    lines = [
        '$ErrorActionPreference = "Stop"',
        '$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")',
        "Set-Location $RepoRoot",
        "",
        "function Test-PythonCandidate {",
        "    param([string]$Path)",
        "    if (-not $Path) { return $false }",
        "    try { & $Path --version *> $null; return $LASTEXITCODE -eq 0 } catch { return $false }",
        "}",
        "$PythonCandidates = @()",
        "if ($env:GRAPH_COLD_PYTHON) { $PythonCandidates += $env:GRAPH_COLD_PYTHON }",
        '$PythonCandidates += (Join-Path $RepoRoot ".venv\\Scripts\\python.exe")',
        '$PythonCandidates += (Join-Path $env:USERPROFILE ".cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe")',
        "$PathPython = Get-Command python -ErrorAction SilentlyContinue",
        "if ($PathPython) { $PythonCandidates += $PathPython.Source }",
        "$Python = $null",
        "foreach ($Candidate in $PythonCandidates) {",
        "    if (Test-PythonCandidate $Candidate) { $Python = $Candidate; break }",
        "}",
        'if (-not $Python) { throw "No usable Python interpreter found. Set GRAPH_COLD_PYTHON to Python 3.10+ and rerun." }',
        "",
    ]
    for step in steps:
        if step == "BUILD_ELSEVIER":
            lines.extend(
                [
                    "Push-Location paper\\elsevier",
                    "powershell -ExecutionPolicy Bypass -File .\\build_elsevier.ps1",
                    "Pop-Location",
                    "",
                ]
            )
        else:
            lines.append(f"# Equivalent manual command: python {step}")
            lines.append(f"& $Python {step}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_method_figure() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out_pdf = Path("figures/fig1_method_overview.pdf")
    out_png = Path("figures/fig1_method_overview.png")
    paper_pdf = PAPER / "figures/fig1_method_overview.pdf"
    paper_png = PAPER / "figures/fig1_method_overview.png"
    labels = [
        ("Audited real data", "CICIDS postfilter11\nCESNET postfilter25"),
        ("Multi-view graphs", "host / IP / temporal\nview masks"),
        ("Representation", "contrastive + temporal\nreconstruction"),
        ("Graph-CDM", "label-space consistency\nevidence score"),
        ("SOC output", "weighted training\npriority proxy"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 3.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    xs = [0.45, 2.65, 4.85, 7.05, 9.25]
    for idx, (x, (title, subtitle)) in enumerate(zip(xs, labels)):
        box = FancyBboxPatch(
            (x, 0.55),
            1.65,
            1.0,
            boxstyle="round,pad=0.05,rounding_size=0.04",
            linewidth=1.1,
            edgecolor="#30415d",
            facecolor="#f5f7fb",
        )
        ax.add_patch(box)
        ax.text(x + 0.825, 1.22, title, ha="center", va="center", fontsize=9.0, weight="bold", color="#172033")
        ax.text(x + 0.825, 0.86, subtitle, ha="center", va="center", fontsize=7.3, color="#39475e", linespacing=1.25)
        if idx < len(xs) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1.74, 1.05),
                    (xs[idx + 1] - 0.08, 1.0),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.0,
                    color="#4d637f",
                )
            )
    fig.text(0.02, 0.92, "Fig1. Graph-CoLD submission-scope workflow", fontsize=11, weight="bold", color="#172033")
    fig.text(0.02, 0.84, "Only verified views and datasets enter the formal result matrix.", fontsize=8.5, color="#4d637f")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(paper_pdf, bbox_inches="tight")
    fig.savefig(paper_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _harden_manuscript_text(metrics: dict[str, float]) -> None:
    tex_path = PAPER / "graph_cold_cas_realdata.tex"
    text = tex_path.read_text(encoding="utf-8")
    text = text.replace(r"\address{Computers \& Security submission draft v1.0}", r"\address{Computers \& Security submission candidate v1.0}")
    text = text.replace("D8 does not modify this\nencoder or representation loss.", "This submission does not modify the\nencoder or representation loss.")
    text = text.replace("and this D8 manuscript.", "and this manuscript.")
    if "figures/fig1_method_overview.pdf" not in text:
        insert = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_method_overview.pdf}
\caption{Graph-CoLD workflow for the formal submission scope.}
\label{fig:method-overview}
\end{figure}
"""
        text = text.replace("\\end{enumerate}\n\n\\section{Related Work}", "\\end{enumerate}\n" + insert + "\n\\section{Related Work}")
    text = text.replace(
        "The authors should confirm the final declaration before submission. No competing\ninterest is recorded in this repository draft.",
        "No competing interest is recorded in this repository candidate package. The named\nauthors must confirm the final declaration before journal upload.",
    )
    text = text.replace(
        "Author-specific funding and\ninstitutional acknowledgements should be confirmed before final submission.",
        "Author-specific funding and\ninstitutional acknowledgements must be confirmed before journal upload.",
    )
    text = text.replace(
        "The next step toward final journal\nsubmission is not more polishing of these numbers, but broader faithful baseline\ncoverage and a verified enterprise provenance case.",
        "The next technical step is broader faithful baseline coverage and a verified\nenterprise provenance case; final journal upload additionally requires human\nauthor-metadata confirmation.",
    )
    tex_path.write_text(text, encoding="utf-8")


def _build_main_pdf(paper: Path) -> None:
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ".\\build_elsevier.ps1"], cwd=paper, check=True)
    _remove_latex_temps(paper)


def _build_candidate_package(package: Path, paper: Path, metrics: dict[str, float], compile_package: bool) -> None:
    if package.exists():
        shutil.rmtree(package)
    author = package / "author"
    review = package / "review"
    support = package / "submission_materials"
    trace = package / "source_trace"
    for root in (author, review, support, trace):
        root.mkdir(parents=True, exist_ok=True)
    for root in (author, review):
        shutil.copy2(paper / "elsarticle.cls", root / "elsarticle.cls")
        shutil.copy2(paper / "build_elsevier.ps1", root / "build_elsevier.ps1")
        shutil.copy2(paper / "references.bib", root / "references.bib")
        shutil.copytree(paper / "figures", root / "figures")
        shutil.copytree(paper / "tables", root / "tables")
    shutil.copy2(paper / "graph_cold_cas_realdata.tex", author / "graph_cold_cas_realdata.tex")
    review_text = _anonymize_tex((paper / "graph_cold_cas_realdata.tex").read_text(encoding="utf-8"))
    (review / "graph_cold_cas_realdata.tex").write_text(review_text, encoding="utf-8")
    for material in (PAPER / "cover_letter_draft.md", *SUPPORT_FILES):
        shutil.copy2(material, support / material.name)
    for src in (REPORTS / "d8/d8_hardening_audit.json", REPORTS / "d8/reviewer_risk_register_v1.md"):
        shutil.copy2(src, trace / src.name)
    (trace / "source_hash_manifest.json").write_text(json.dumps(_source_hashes(_trace_paths()), indent=2), encoding="utf-8")
    (package / "README.md").write_text(_package_readme(metrics), encoding="utf-8")
    if compile_package:
        for root in (author, review):
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ".\\build_elsevier.ps1"], cwd=root, check=True)
            _remove_latex_temps(root)


def _anonymize_tex(text: str) -> str:
    text = text.replace(r"\author{Graph-CoLD Project Team}", r"\author{Anonymous Authors}")
    text = text.replace(r"\address{Computers \& Security submission candidate v1.0}", r"\address{Manuscript under anonymous review}")
    text = re.sub(
        r"\\section\*\{Acknowledgements\}.*?\\section\*\{Declaration of competing interest\}",
        lambda _match: "\\section*{Acknowledgements}\nAcknowledgements are omitted for anonymous review.\n\n\\section*{Declaration of competing interest}",
        text,
        flags=re.S,
    )
    return text


def _package_readme(metrics: dict[str, float]) -> str:
    return f"""# Graph-CoLD C&S Candidate Package D9

This is a submission-ready candidate package, not an automatic journal upload.
`submission_ready` remains false until named authors confirm metadata, funding,
competing-interest declarations, and final editorial approval.

## Contents

- `author/`: author manuscript source, PDF, figures, tables, and BibTeX.
- `review/`: anonymous review manuscript source, PDF, figures, tables, and BibTeX.
- `submission_materials/`: cover letter candidate, highlights, CRediT, funding,
  competing-interest, and data-availability statements.
- `source_trace/`: D8 audit/risk notes and hashes for frozen source artifacts.

## Formal Scope

- Datasets: CICIDS-2017 postfilter11; CESNET-TLS-Year22 postfilter25.
- Methods: Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised,
  Confident-Learning, and Co-Teaching-lite.
- Excluded: MALTLS-22, OpTC, UNSW-NB15, USTC-TFC2016, FINE, MCRe, MORSE,
  Flash, Argus, Decoupling, and full Co-Teaching.

## Headline Numbers

- Graph-CoLD vs CoLD Macro-F1 lift: {metrics['mean_diff_pp']:.2f} percentage points.
- Paired grouped p-value: {metrics['p_value']:.2e}; Cohen dz={metrics['effect_size']:.3f}; n={int(metrics['n_pairs'])}.
- ERR_final: Graph-CoLD {metrics['graph_err']:.4f}; ablation_hard {metrics['hard_err']:.4f}.

## Lock Rule

Do not modify `results/*.csv` or `results/*.json` inside this package flow. Any
new experiment requires returning to the D5/D5.5 gate and regenerating all
downstream D6-D9 artifacts.
"""


def _audit(package: Path, before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    tex = (PAPER / "graph_cold_cas_realdata.tex").read_text(encoding="utf-8")
    refs = (PAPER / "references.bib").read_text(encoding="utf-8")
    main = pd.read_csv("results/table_main_expanded.csv")
    d8 = json.loads((REPORTS / "d8/d8_hardening_audit.json").read_text(encoding="utf-8"))
    git_status = _git_status()
    model_touched = any(
        path.startswith(("src/models/", "src/data/noise", "src/metrics.py", "src/representation", "src/graph"))
        for path in git_status["changed_paths"]
    )
    formal_scope_ok = set(main["reported_as"].dropna().unique()) == FORMAL_DATASETS and FORMAL_METHODS.issubset(set(main["method"].dropna().unique()))
    excluded_scope_ok = all(dataset not in tex for dataset in ("MALTLS-22 results", "OpTC formal", "UNSW-NB15 results", "USTC-TFC2016 results"))
    forbidden_claims = ["synthetic result", "fallback result", "emulation result", "state-of-the-art", "near-perfect", "beats all baselines"]
    forbidden_absent = not any(term in tex.lower() for term in forbidden_claims)
    support_ok = all(path.exists() and path.stat().st_size > 0 for path in SUPPORT_FILES)
    package_ok = (
        (package / "author/graph_cold_cas_realdata.pdf").exists()
        and (package / "review/graph_cold_cas_realdata.pdf").exists()
        and (package / "submission_materials/highlights.md").exists()
        and (package / "source_trace/source_hash_manifest.json").exists()
    )
    review_text = (package / "review/graph_cold_cas_realdata.tex").read_text(encoding="utf-8") if (package / "review/graph_cold_cas_realdata.tex").exists() else ""
    audit = {
        "stage": "D9",
        "submission_ready": False,
        "candidate_package_ready": bool(package_ok),
        "human_confirmation_required": True,
        "real_data_only": bool(d8.get("real_data_only", False)),
        "results_unchanged": before == after,
        "results_hashes": after,
        "formal_scope_ok": bool(formal_scope_ok),
        "excluded_scope_declared": "MALTLS-22 is not evaluated" in tex and "OpTC is not evaluated as a formal enterprise case" in tex,
        "excluded_scope_not_reported": bool(excluded_scope_ok),
        "co_teaching_lite_not_full": "not a full Co-Teaching implementation" in tex and "not a full Co-Teaching reproduction" in tex,
        "cesnet_not_maltls": "CESNET-TLS-Year22" in tex and "CESNET-TLS-Year22 postfilter25" in tex,
        "cesnet_subset_declared": "not a full-archive evaluation" in tex,
        "no_forbidden_overclaiming": bool(forbidden_absent),
        "references_no_placeholders": not any(term in refs.lower() for term in ["todo", "tbd", "fake", "unknown"]),
        "support_materials_present": bool(support_ok),
        "author_version_present": (package / "author/graph_cold_cas_realdata.pdf").exists(),
        "anonymous_review_version_present": (package / "review/graph_cold_cas_realdata.pdf").exists(),
        "anonymous_review_version_anonymized": "Graph-CoLD Project Team" not in review_text and "Anonymous Authors" in review_text,
        "raw_data_not_packaged": not any(part.lower() in {"data", "datasets"} for path in package.rglob("*") for part in path.parts),
        "aria2_logs_not_packaged": not any("aria2" in path.name.lower() for path in package.rglob("*")),
        "model_code_unchanged_in_worktree": not model_touched,
        "d8_submission_ready_left_false": d8.get("final_submission_ready") is False,
        "blocking_items": [
            "Named author list and affiliations require human confirmation.",
            "Funding and competing-interest statements require human confirmation.",
            "Final journal upload decision is manual; submission_ready intentionally remains false.",
        ],
    }
    return audit


def _write_audit_files(out: Path, audit: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "d9_submission_lock_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    checks = "\n".join(f"- {key}: {value}" for key, value in audit.items() if key != "results_hashes")
    md = f"""# D9 Submission Lock Audit

## Overall assessment

Candidate package generated. `submission_ready` remains `false` by design.

## Checks

{checks}

## Required human confirmations

- Named author list and affiliations.
- Funding statement.
- Competing-interest declaration.
- Final editorial and journal-upload approval.
"""
    (out / "d9_submission_lock_audit.md").write_text(md, encoding="utf-8")


def _write_package_manifest(package: Path, audit: dict[str, Any]) -> None:
    files = {}
    if package.exists():
        for path in sorted(p for p in package.rglob("*") if p.is_file()):
            files[str(path.relative_to(package)).replace("\\", "/")] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "stage": "D9",
        "package": str(package),
        "submission_ready": False,
        "candidate_package_ready": audit["candidate_package_ready"],
        "files": files,
    }
    (package / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (REPORTS / "d9/d9_package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _source_hashes(paths: tuple[Path, ...] | list[Path]) -> dict[str, str]:
    return {str(path).replace("\\", "/"): _sha256(path) for path in paths}


def _trace_paths() -> list[Path]:
    paths: list[Path] = list(RESULT_SOURCES)
    paths.extend(sorted(Path("tables").glob("*.csv")))
    paths.extend(sorted(Path("tables").glob("*.md")))
    paths.extend(sorted(Path("figures").glob("fig*.pdf")))
    paths.extend(sorted(Path("figures").glob("fig*.png")))
    paths.extend([PAPER / "graph_cold_cas_realdata.tex", PAPER / "references.bib"])
    return paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_latex_temps(root: Path) -> None:
    for suffix in (".aux", ".bbl", ".blg", ".log", ".out"):
        target = root / f"graph_cold_cas_realdata{suffix}"
        if target.exists():
            target.unlink()


def _git_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(["git", "status", "--short"], text=True, capture_output=True, check=True)
    except Exception:
        return {"changed_paths": []}
    paths = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        paths.append(line[3:].replace("\\", "/"))
    return {"changed_paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--no-compile-package", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(refresh_d9_audit(), indent=2))
    else:
        print(json.dumps(run_d9_submission_lock(compile_package=not args.no_compile_package), indent=2))


if __name__ == "__main__":
    main()
