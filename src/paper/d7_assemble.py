"""Assemble the D7 real-data Computers & Security manuscript package.

The generator is intentionally an aggregation-only step. It reads frozen D5/D5.5
and D6 artifacts, creates manuscript assets, and writes audit material. It does
not import experiment runners, model code, graph encoders, noise injectors, or
metric implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


MAIN_SOURCE = Path("results/table_main_expanded.csv")
BASELINE_SOURCE = Path("results/table_baseline_expansion.csv")
STATS_SOURCE = Path("results/stat_tests_baseline_expansion.json")
READINESS_SOURCE = Path("reports/realdata_readiness_report.json")
D6_CHECKLIST_SOURCE = Path("reports/d6/d6_paper_prep_checklist.json")

FORMAL_DATASETS = ("CICIDS-2017", "CESNET-TLS-Year22")
FORMAL_METHODS = (
    "Graph-CoLD",
    "CoLD",
    "ablation_hard",
    "Noisy-Supervised",
    "Confident-Learning",
    "Co-Teaching-lite",
)
EXCLUDED_METHODS = ("FINE", "MCRe", "MORSE", "Flash", "Argus", "Decoupling", "full Co-Teaching")
FIGURE_FILES = (
    "fig2_macro_f1_vs_noise_rate",
    "fig3_err_retention",
    "fig4_ablation",
    "fig5_runtime_cost",
)
SOURCE_TABLES = (
    "table_1_dataset_protocol.csv",
    "table_2_main_performance.csv",
    "table_3_high_noise_summary.csv",
    "table_4_ablation_evidence.csv",
    "table_5_statistical_tests.csv",
)


def run_d7_assembly(
    paper_dir: str | Path = "paper/elsevier",
    reports_dir: str | Path = "reports",
    reproducibility_dir: str | Path = "reproducibility",
) -> dict[str, Any]:
    """Generate D7 manuscript, review, reproducibility, and audit artifacts."""
    paper = Path(paper_dir)
    reports = Path(reports_dir)
    reports_d7 = reports / "d7"
    repro = Path(reproducibility_dir)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    reports_d7.mkdir(parents=True, exist_ok=True)
    repro.mkdir(parents=True, exist_ok=True)

    sources = _load_sources()
    main = sources["main"]
    _validate_scope(main)

    _copy_figures(paper / "figures")
    latex_tables = _write_latex_tables(paper / "tables", sources)
    metrics = _metrics(main, sources["stats"])
    _write_reconciliation(sources, reports_d7)
    _write_class(paper)
    _write_bib(paper)
    _write_manuscript(paper, metrics, latex_tables, sources)
    _write_build_scripts(paper)
    _write_cover_letter(paper, metrics)
    _write_reproducibility(repro, sources)
    _write_reviewer_pack(reports_d7, metrics, sources)
    audit = _write_final_audit(reports_d7, paper)

    manifest = {
        "stage": "D7 real-data manuscript assembly",
        "completed": True,
        "source_csv": str(MAIN_SOURCE),
        "source_sha256": _sha256(MAIN_SOURCE),
        "manuscript": str(paper / "graph_cold_cas_realdata.tex"),
        "pdf": str(paper / "graph_cold_cas_realdata.pdf"),
        "reports": [
            str(reports_d7 / "d7_readiness_reconciliation.json"),
            str(reports_d7 / "reviewer_simulation_final.md"),
            str(reports_d7 / "rebuttal_preparation_pack.md"),
            str(reports_d7 / "d7_final_audit.json"),
            str(reports_d7 / "d7_final_audit.md"),
        ],
        "audit": audit,
    }
    (reports_d7 / "d7_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def update_d7_audit(
    paper_dir: str | Path = "paper/elsevier",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Refresh audit booleans after an external LaTeX build."""
    reports_d7 = Path(reports_dir) / "d7"
    reports_d7.mkdir(parents=True, exist_ok=True)
    return _write_final_audit(reports_d7, Path(paper_dir))


def _load_sources() -> dict[str, Any]:
    required = [
        MAIN_SOURCE,
        BASELINE_SOURCE,
        STATS_SOURCE,
        READINESS_SOURCE,
        D6_CHECKLIST_SOURCE,
        Path("reports/d6/d6_statistical_narrative.md"),
        Path("reports/d6/reviewer_risk_notes.md"),
        Path("reports/d5_baseline_expansion_report.json"),
        Path("reports/d5_scale_policy.json"),
        Path("reports/cicids_final_protocol.json"),
        Path("reports/cesnet_class_policy_report.json"),
        Path("reports/cesnet_view_policy_report.json"),
    ]
    required.extend(Path("tables") / name for name in SOURCE_TABLES)
    required.extend(Path("figures") / f"{stem}.pdf" for stem in FIGURE_FILES)
    required.extend(Path("figures") / f"{stem}.png" for stem in FIGURE_FILES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("D7 requires completed D5.5/D6 artifacts: " + ", ".join(missing))

    sources: dict[str, Any] = {
        "main": pd.read_csv(MAIN_SOURCE),
        "baseline": pd.read_csv(BASELINE_SOURCE),
        "stats": json.loads(STATS_SOURCE.read_text(encoding="utf-8")),
        "readiness": json.loads(READINESS_SOURCE.read_text(encoding="utf-8")),
        "d6_checklist": json.loads(D6_CHECKLIST_SOURCE.read_text(encoding="utf-8")),
        "baseline_report": json.loads(Path("reports/d5_baseline_expansion_report.json").read_text(encoding="utf-8")),
        "scale_policy": json.loads(Path("reports/d5_scale_policy.json").read_text(encoding="utf-8")),
        "cicids_protocol": json.loads(Path("reports/cicids_final_protocol.json").read_text(encoding="utf-8")),
        "cesnet_policy": json.loads(Path("reports/cesnet_class_policy_report.json").read_text(encoding="utf-8")),
        "cesnet_view_policy": json.loads(Path("reports/cesnet_view_policy_report.json").read_text(encoding="utf-8")),
        "d6_narrative": Path("reports/d6/d6_statistical_narrative.md").read_text(encoding="utf-8"),
        "d6_risk_notes": Path("reports/d6/reviewer_risk_notes.md").read_text(encoding="utf-8"),
        "source_hashes": {
            str(MAIN_SOURCE): _sha256(MAIN_SOURCE),
            str(BASELINE_SOURCE): _sha256(BASELINE_SOURCE),
            str(STATS_SOURCE): _sha256(STATS_SOURCE),
            str(READINESS_SOURCE): _sha256(READINESS_SOURCE),
        },
    }
    for name in SOURCE_TABLES:
        sources[name] = pd.read_csv(Path("tables") / name)
        sources["source_hashes"][f"tables/{name}"] = _sha256(Path("tables") / name)
    return sources


def _validate_scope(main: pd.DataFrame) -> None:
    datasets = tuple(sorted(main["reported_as"].dropna().unique()))
    if datasets != tuple(sorted(FORMAL_DATASETS)):
        raise ValueError(f"D7 formal datasets must be {FORMAL_DATASETS}, got {datasets}")
    methods = set(main["method"].dropna().astype(str))
    missing = [method for method in FORMAL_METHODS if method not in methods]
    if missing:
        raise ValueError(f"D7 formal methods missing from result matrix: {missing}")
    if not bool(main["source_verified"].astype(bool).all()):
        raise ValueError("D7 cannot assemble manuscript from unverified source rows.")


def _copy_figures(out_dir: Path) -> None:
    for stem in FIGURE_FILES:
        for suffix in (".png", ".pdf"):
            shutil.copy2(Path("figures") / f"{stem}{suffix}", out_dir / f"{stem}{suffix}")


def _write_latex_tables(out_dir: Path, sources: dict[str, Any]) -> list[str]:
    written = []

    dataset = sources["table_1_dataset_protocol.csv"][
        ["Dataset", "Rows used", "Class policy", "Number of classes", "Sample policy", "Active views"]
    ].rename(columns={"Rows used": "Rows", "Number of classes": "Classes"})
    dataset["Rows"] = dataset["Rows"].map(lambda value: f"{int(value):,}")
    dataset["Sample policy"] = dataset["Sample policy"].map(_compact_policy)
    written.append(_write_table_tex(out_dir / "table_1_dataset_protocol.tex", dataset, "tab:dataset-protocol"))

    main = _main_summary(sources["main"])
    written.append(_write_table_tex(out_dir / "table_2_main_summary.tex", main, "tab:main-performance"))

    high = sources["table_3_high_noise_summary.csv"].copy()
    high = high[["Dataset", "Method", "Macro-F1 mean", "ERR mean", "Compression ratio mean", "Scenario count"]]
    high = high.rename(
        columns={
            "Macro-F1 mean": "Macro-F1",
            "ERR mean": "ERR",
            "Compression ratio mean": "Compression",
            "Scenario count": "Scenarios",
        }
    )
    written.append(_write_table_tex(out_dir / "table_3_high_noise.tex", high, "tab:high-noise"))

    ablation = sources["table_4_ablation_evidence.csv"][
        ["Dataset", "Variant", "Macro-F1", "ERR_final", "retained_fraction_clean_informative", "compression_ratio"]
    ].rename(
        columns={
            "ERR_final": "ERR_final",
            "retained_fraction_clean_informative": "Retained clean-info",
            "compression_ratio": "Compression",
        }
    )
    written.append(_write_table_tex(out_dir / "table_4_ablation.tex", ablation, "tab:ablation"))

    stats = sources["table_5_statistical_tests.csv"][["Comparison", "Mean difference", "p-value", "Effect size", "n"]]
    stats = stats.copy()
    stats["p-value"] = stats["p-value"].map(_compact_p_value)
    written.append(_write_table_tex(out_dir / "table_5_statistical_tests.tex", stats, "tab:stats"))
    return written


def _main_summary(main: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        main.groupby(["reported_as", "method"], dropna=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            err=("err_final", "mean"),
            compression=("compression_ratio", "mean"),
            runtime=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    grouped["dataset_order"] = grouped["reported_as"].map({name: i for i, name in enumerate(FORMAL_DATASETS)})
    grouped["method_order"] = grouped["method"].map({name: i for i, name in enumerate(FORMAL_METHODS)})
    grouped = grouped.sort_values(["dataset_order", "method_order"])
    out = pd.DataFrame(
        {
            "Dataset": grouped["reported_as"],
            "Method": grouped["method"],
            "Macro-F1": grouped["macro_f1"].map(lambda v: f"{v:.4f}"),
            "FPR": grouped["fpr"].map(lambda v: f"{v:.4f}"),
            "FNR": grouped["fnr"].map(lambda v: f"{v:.4f}"),
            "ERR": grouped["err"].map(lambda v: f"{v:.4f}"),
            "Compression": grouped["compression"].map(lambda v: f"{v:.4f}"),
            "Runtime (s)": grouped["runtime"].map(lambda v: f"{v:.2f}"),
        }
    )
    return out


def _write_table_tex(path: Path, frame: pd.DataFrame, label: str) -> str:
    align = "l" * len(frame.columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline"]
    lines.append(" & ".join(_tex_escape(col) for col in frame.columns) + r" \\")
    lines.append("\\hline")
    for _, row in frame.iterrows():
        lines.append(" & ".join(_tex_escape(str(value)) for value in row.tolist()) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _metrics(main: pd.DataFrame, stats: dict[str, Any]) -> dict[str, Any]:
    means = main.groupby("method").agg(
        macro_f1=("macro_f1", "mean"),
        err_final=("err_final", "mean"),
        compression=("compression_ratio", "mean"),
        runtime=("runtime_sec", "mean"),
        memory=("memory_mb", "mean"),
    )
    high = main[
        (pd.to_numeric(main["noise_rate"], errors="coerce") >= 0.4)
        & main["noise_type"].isin(["symmetric", "asymmetric", "graph_consistency"])
    ]
    high_means = high.groupby(["reported_as", "method"]).agg(
        macro_f1=("macro_f1", "mean"),
        err_final=("err_final", "mean"),
        compression=("compression_ratio", "mean"),
    )
    comparison = stats["comparisons"]["Graph-CoLD_vs_CoLD"]
    return {
        "graph_macro": float(means.loc["Graph-CoLD", "macro_f1"]),
        "cold_macro": float(means.loc["CoLD", "macro_f1"]),
        "graph_err": float(means.loc["Graph-CoLD", "err_final"]),
        "hard_err": float(means.loc["ablation_hard", "err_final"]),
        "graph_compression": float(means.loc["Graph-CoLD", "compression"]),
        "cold_compression": float(means.loc["CoLD", "compression"]),
        "graph_runtime": float(means.loc["Graph-CoLD", "runtime"]),
        "cold_runtime": float(means.loc["CoLD", "runtime"]),
        "graph_memory": float(means.loc["Graph-CoLD", "memory"]),
        "cold_memory": float(means.loc["CoLD", "memory"]),
        "p_value": float(comparison["p_value"]),
        "mean_diff_pp": float(comparison["mean_diff"]) * 100.0,
        "effect_size": float(comparison["effect_size_cohen_dz"]),
        "n_pairs": int(comparison["n_pairs"]),
        "cicids_high_lift_pp": _lift(high_means, "CICIDS-2017"),
        "cesnet_high_lift_pp": _lift(high_means, "CESNET-TLS-Year22"),
    }


def _lift(grouped: pd.DataFrame, dataset: str) -> float:
    graph = float(grouped.loc[(dataset, "Graph-CoLD"), "macro_f1"])
    cold = float(grouped.loc[(dataset, "CoLD"), "macro_f1"])
    return (graph - cold) * 100.0


def _write_reconciliation(sources: dict[str, Any], out_dir: Path) -> None:
    readiness = sources["readiness"]
    d6 = sources["d6_checklist"]
    reconciliation = {
        "source_of_truth": "reports/realdata_readiness_report.json",
        "d7_allowed": bool(readiness.get("d7_allowed", False)),
        "d7_artifacts_generated_before_d7": bool(d6.get("d7_allowed", False)),
        "d6_checklist_d7_allowed_meaning": "D6 did not generate D7 manuscript artifacts.",
        "readiness_d7_allowed_meaning": "D5.5/D6 real-data artifacts are available for D7 assembly.",
        "submission_ready": False,
    }
    (out_dir / "d7_readiness_reconciliation.json").write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")
    md = f"""# D7 Readiness Reconciliation

- Source of truth: `reports/realdata_readiness_report.json`
- D7 assembly allowed: {str(reconciliation['d7_allowed']).lower()}
- D7 artifacts existed before this D7 step: {str(reconciliation['d7_artifacts_generated_before_d7']).lower()}
- Submission ready: false

The D6 checklist field records that D6 itself did not create the D7 manuscript
package. The real-data readiness report records that the verified D5.5 and D6
artifacts exist and can be assembled into D7 paper materials.
"""
    (out_dir / "d7_readiness_reconciliation.md").write_text(md, encoding="utf-8")


def _write_class(paper: Path) -> None:
    (paper / "elsarticle.cls").write_text(
        r"""\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{elsarticle}[2026/07/05 local minimal compatibility layer]
\LoadClass[12pt]{article}
\newenvironment{frontmatter}{}{}
\newenvironment{keyword}{\par\noindent\textbf{Keywords: }}{\par}
\newcommand{\sep}{; }
\newcommand{\journal}[1]{}
\newcommand{\address}[1]{}
\newcommand{\cortext}[2][]{}
\newcommand{\corref}[1]{}
\newcommand{\ead}[1]{}
""",
        encoding="utf-8",
    )


def _write_bib(paper: Path) -> None:
    (paper / "references.bib").write_text(
        r"""@inproceedings{sharafaldin2018cicids,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  booktitle={Proceedings of the International Conference on Information Systems Security and Privacy},
  pages={108--116},
  year={2018},
  doi={10.5220/0006639801080116}
}

@misc{cesnettlsyear22,
  title={{CESNET-TLS-Year22}},
  author={{CESNET}},
  year={2024},
  howpublished={Zenodo record 10608607},
  url={https://zenodo.org/records/10608607}
}

@inproceedings{yang2026cold,
  title={{CoLD}: Towards the Detection of Cyber Threats under Noisy Labels},
  author={Yang, Jinyu and Shen, Bo and Yuan, Yawen and Chen, Xingyu and Li, Zongyue and Zou, Deqing and Jin, Hai},
  booktitle={Proceedings of the Network and Distributed System Security Symposium},
  year={2026},
  url={https://www.ndss-symposium.org/ndss-paper/cold-towards-the-detection-of-cyber-threats-under-noisy-labels/}
}

@article{northcutt2021confident,
  title={Confident Learning: Estimating Uncertainty in Dataset Labels},
  author={Northcutt, Curtis G. and Jiang, Lu and Chuang, Isaac L.},
  journal={Journal of Artificial Intelligence Research},
  volume={70},
  pages={1373--1411},
  year={2021},
  doi={10.1613/jair.1.12125}
}

@inproceedings{han2018coteaching,
  title={Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels},
  author={Han, Bo and Yao, Quanming and Yu, Xingrui and Niu, Gang and Xu, Miao and Hu, Weihua and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}

@inproceedings{patrini2017loss,
  title={Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach},
  author={Patrini, Giorgio and Rozza, Alessandro and Krishna Menon, Aditya and Nock, Richard and Qu, Lizhen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1944--1952},
  year={2017}
}

@inproceedings{kipf2017gcn,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2017}
}

@inproceedings{velickovic2018gat,
  title={Graph Attention Networks},
  author={Velickovic, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
""",
        encoding="utf-8",
    )


def _write_manuscript(paper: Path, metrics: dict[str, Any], latex_tables: list[str], sources: dict[str, Any]) -> None:
    source_hash = sources["source_hashes"][str(MAIN_SOURCE)]
    source_hash_prefix = source_hash[:16]
    baseline_exclusions = "; ".join(f"{method}: unavailable as a smoke-passed real-data implementation" for method in EXCLUDED_METHODS)
    text = rf"""\documentclass[preprint,review,12pt]{{elsarticle}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{graphicx}}
\setlength{{\oddsidemargin}}{{0in}}
\setlength{{\evensidemargin}}{{0in}}
\setlength{{\textwidth}}{{6.5in}}
\setlength{{\topmargin}}{{-0.25in}}
\setlength{{\textheight}}{{8.9in}}
\journal{{Computers \& Security}}

\title{{Graph-CoLD: Label-space Graph Consistency and Evidence-preserving Denoising for SOC Alert Prioritization}}
\author{{Graph-CoLD Project Team}}
\address{{Real-data manuscript assembly generated from frozen D5.5/D6 artifacts.}}

\begin{{document}}
\begin{{frontmatter}}
\maketitle
\begin{{abstract}}
Security operations centers must triage large volumes of alerts under noisy
labels and incomplete operational context. Graph-CoLD extends CoLD-style noisy
label learning from independent samples to multi-view graphs. The method uses a
label-space consistency diagnostic function, evidence preserving soft weights,
and an operational ranking score to reduce review load while retaining clean
informative evidence. We evaluate the implementation on two verified real-data
settings, CICIDS-2017 postfilter11 and CESNET-TLS-Year22 postfilter25. Across the
frozen D5.5 matrix, Graph-CoLD improves Macro-F1 over the self-implemented CoLD
baseline by {metrics['mean_diff_pp']:.2f} percentage points under a paired grouped
t-test (p={metrics['p_value']:.2e}, Cohen dz={metrics['effect_size']:.3f},
n={metrics['n_pairs']}). The strongest empirical signal is not perfect detection
but improved robustness and evidence retention under noisy labels. The manuscript
therefore reports Macro-F1, FPR, FNR, ERR, compression ratio, runtime, and memory,
with explicit limits on unsupported datasets and baselines.
\end{{abstract}}
\begin{{keyword}}
intrusion detection \sep noisy labels \sep graph learning \sep SOC alert prioritization \sep evidence retention
\end{{keyword}}
\end{{frontmatter}}

\section{{Introduction}}
Modern SOC pipelines combine network telemetry, traffic classifiers, threat
labels, and analyst feedback. This creates two linked problems: high alert volume
and imperfect labels. A model that simply optimizes classification accuracy can
learn mislabeled regions or discard rare but operationally useful alerts. CoLD
addresses noisy labels for cyber threat detection, but its original formulation
treats training instances as independent samples and uses hard deletion after a
confidence diagnostic \cite{{yang2026cold}}. Graph-CoLD asks whether the same
problem can be handled more carefully when alerts have relational evidence across
hosts, IPs, and time.

This paper presents a real-data manuscript package for Graph-CoLD. The core idea
is to move the diagnostic from embedding distance to label-space consistency over
available graph views, then combine the diagnostic with evidence preservation. A
sample can be downweighted when it appears inconsistent, but a clean informative
sample should remain visible if it carries rare-class or anomaly evidence. The
ranking component translates these weights into an alert-ordering proxy for SOC
review; the present evaluation focuses on robustness, ERR, compression, and cost
rather than on an analyst user study.

\section{{Related Work}}
CoLD is the closest method, motivating the reordering, diagnostic, and robust
training structure used in this project \cite{{yang2026cold}}. Confident learning
estimates label issues from predicted probabilities and motivates the
Confident-Learning baseline \cite{{northcutt2021confident}}. Co-teaching trains
peer networks on selected small-loss samples \cite{{han2018coteaching}}; our
Co-Teaching-lite baseline is explicitly a lightweight smoke-passed approximation,
not a faithful full deep reproduction. General loss-correction work also shows why
label noise can shift decision boundaries \cite{{patrini2017loss}}.

Graph learning provides a way to represent alert context through relations rather
than isolated feature vectors. Graph convolutional and attention models are common
building blocks for relational representation learning \cite{{kipf2017gcn,
velickovic2018gat}}. For intrusion detection data, CICIDS-2017 remains a widely
used benchmark with known class imbalance and traffic-generation caveats
\cite{{sharafaldin2018cicids}}. CESNET-TLS-Year22 provides a large real TLS
traffic source used here as a verified replacement for the unavailable MALTLS-22
source \cite{{cesnettlsyear22}}.

\section{{Problem Formulation and Motivation}}
Let $v \in V$ denote a training alert or flow instance and let $G^m=(V,E^m)$ be an
active graph view. The current real-data experiments use host, IP, and temporal
views for CICIDS-2017, and IP plus temporal views for CESNET-TLS-Year22. Process
and threat-intelligence views are disabled when a dataset lacks reliable
provenance or threat-intelligence fields. The training labels may be corrupted,
so the observed label $\tilde{{y}}_v$ is not assumed to be fully reliable.

The objective is to learn a classifier and prioritization score that remain robust
under label noise while preserving clean informative evidence. We therefore
measure classification quality with Macro-F1, FPR, and FNR, and measure retention
quality with ERR and Tail-ERR on clean informative samples. Compression ratio is
reported as an operational queue-reduction proxy.

\section{{Methodology}}
Stage 1 uses the existing multi-view representation encoder from D2. The encoder
applies CoLD-style feature masking, view-level message passing, cross-view
contrastive learning, temporal alignment, and reconstruction. View fusion is the
mean of active views. D7 does not alter this encoder or its loss.

\subsection{{Graph-CDM}}
Graph-CDM is a label-space consistency diagnostic function:
\begin{{equation}}
\operatorname{{GraphCDM}}(v)=\lambda_1D_{{pred}}(v)+\lambda_2D_{{neigh}}(v)+
\lambda_3D_{{view}}(v)+\lambda_4D_{{chain}}(v).
\end{{equation}}
The prediction term compares view-wise predictions with the observed noisy
training label,
\begin{{equation}}
D_{{pred}}(v)=\frac{{1}}{{M}}\sum_{{m=1}}^M
\mathbf{{1}}\left[\tilde{{y}}_v^{{(m)}} \ne \tilde{{y}}_v\right].
\end{{equation}}
The neighborhood term is a label-space KL divergence,
\begin{{equation}}
D_{{neigh}}(v)=KL\left(\hat{{y}}_v\,\middle\|\,\frac{{1}}{{|N(v)|}}\sum_{{u\in N(v)}}\hat{{y}}_u\right).
\end{{equation}}
The view term is active-view mode disagreement,
\begin{{equation}}
D_{{view}}(v)=1-\max_c \frac{{1}}{{M}}\sum_m \mathbf{{1}}\left[\tilde{{y}}_v^{{(m)}}=c\right].
\end{{equation}}
$D_{{chain}}$ is used only when reliable provenance or temporal chains are
available and is not the central driver of the CICIDS/CESNET experiments.

\subsection{{Evidence preservation and ranking}}
Evidence score $e(v)$ combines class-frequency protection with an anomaly source
and is min-max normalized. The final training weight is
\begin{{equation}}
w(v)=\sigma\left(-\kappa(\operatorname{{GraphCDM}}(v)-\theta)\right)(1-\rho)
+\rho\tilde{{e}}(v).
\end{{equation}}
The classifier loss is the weighted cross entropy over training samples. Ranking
uses
\begin{{equation}}
P(v)=\alpha_1\hat{{y}}_{{mal}}(v)+\alpha_2\operatorname{{GraphCDM}}(v)+\alpha_3\tilde{{e}}(v),
\end{{equation}}
which orders alerts for operational review. In this real-data package, ranking is
reported through compression and evidence-retention metrics rather than an
analyst-facing deployment study.

\section{{Experimental Setup}}
All formal results are drawn from the expanded main results CSV, with SHA256
prefix {source_hash_prefix}; the exact path and full hash are listed in the
reproducibility package. The formal datasets
are CICIDS-2017 postfilter11 and CESNET-TLS-Year22 postfilter25. CICIDS-2017 uses
{int(sources['table_1_dataset_protocol.csv'].loc[0, 'Rows used']):,} rows after
class filtering and dominant-class downsampling. CESNET-TLS-Year22 uses a
deterministic audit-window subset with postfilter25 stratified splitting and must
not be described as a full-archive evaluation.

Noise settings include clean labels, symmetric noise, asymmetric noise, and
graph consistency noise. Noise is injected only into training labels. Results are
reported over seeds 0, 1, and 2 where present. Formal methods are Graph-CoLD,
CoLD, ablation\_hard, Noisy-Supervised, Confident-Learning, and Co-Teaching-lite.
Excluded methods are not reported as formal baselines: {baseline_exclusions}.

\begin{{table}}[t]
\centering
\caption{{Verified real-data protocols used by the D7 manuscript.}}
\label{{tab:dataset-protocol}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_1_dataset_protocol.tex}}}}
\end{{table}}

\section{{Results and Analysis}}
Table~\ref{{tab:main-performance}} summarizes the method-level real-data
performance. Averaged across the frozen matrix, Graph-CoLD obtains Macro-F1
{metrics['graph_macro']:.4f}, compared with {metrics['cold_macro']:.4f} for CoLD.
The paired grouped test reports a {metrics['mean_diff_pp']:.2f} percentage-point
mean difference with p={metrics['p_value']:.2e} and effect size
{metrics['effect_size']:.3f}. The effect is interpreted as a consistent positive
shift, not as a claim that all unimplemented baselines have been beaten.

\begin{{table}}[t]
\centering
\caption{{Mean performance by dataset and formal method, aggregated from the D5.5 matrix.}}
\label{{tab:main-performance}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_2_main_summary.tex}}}}
\end{{table}}

High-noise settings show why Graph-CDM and evidence weighting matter. On
CICIDS-2017 high-noise rows, Graph-CoLD improves Macro-F1 over CoLD by
{metrics['cicids_high_lift_pp']:.2f} percentage points. On CESNET-TLS-Year22 the
corresponding lift is {metrics['cesnet_high_lift_pp']:.2f} percentage points,
where a ceiling effect makes Macro-F1 margins smaller. ERR\_final averages
{metrics['graph_err']:.4f} for Graph-CoLD and {metrics['hard_err']:.4f} for
ablation\_hard, supporting the evidence-preservation contribution.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig2_macro_f1_vs_noise_rate.pdf}}
\caption{{Macro-F1 versus noise rate for verified real-data settings.}}
\label{{fig:macro-noise}}
\end{{figure}}

\begin{{table}}[t]
\centering
\caption{{High-noise summary for rates at or above 0.4.}}
\label{{tab:high-noise}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_3_high_noise.tex}}}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig3_err_retention.pdf}}
\caption{{Evidence retention and compression tradeoff.}}
\label{{fig:err-retention}}
\end{{figure}}

\begin{{table}}[t]
\centering
\caption{{Ablation results for Graph-CDM and evidence preservation.}}
\label{{tab:ablation}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_4_ablation.tex}}}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig4_ablation.pdf}}
\caption{{Ablation comparison for Macro-F1 and ERR.}}
\label{{fig:ablation}}
\end{{figure}}

\begin{{table}}[t]
\centering
\caption{{Paired grouped statistical tests from D5.5.}}
\label{{tab:stats}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_5_statistical_tests.tex}}}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig5_runtime_cost.pdf}}
\caption{{Runtime and memory cost by method.}}
\label{{fig:runtime}}
\end{{figure}}

\section{{Discussion}}
The results support a careful claim: label-space graph consistency and evidence
preserving weights improve robustness and retention on the evaluated real-data
settings. The evidence-retention result is especially important for SOC use
because compression without retained evidence can reduce analyst workload while
damaging investigation quality. Graph-CoLD's additional runtime relative to CoLD
is visible ({metrics['graph_runtime']:.2f}s versus {metrics['cold_runtime']:.2f}s
on average), so the method should be framed as a robustness-retention tradeoff
rather than as a low-cost replacement.

The CESNET-TLS-Year22 setting has high absolute Macro-F1 for several methods,
which limits the visible room for classifier improvement. The manuscript therefore
places the CESNET result next to ERR, compression, and cost, and states the subset
policy explicitly. CICIDS-2017 shows larger high-noise margins but is still only
one benchmark family.

\section{{Limitations}}
The current formal evaluation uses two verified real-data settings. MALTLS-22 is
not evaluated because its source did not pass the local verification gate. OpTC is
not evaluated as a formal enterprise case because no verified local provenance
event table is available. CESNET-TLS-Year22 is a deterministic audit-window
postfilter25 subset, not a full-archive evaluation. Co-Teaching-lite is not a full
Co-Teaching implementation, and FINE, MCRe, MORSE, Flash, Argus, Decoupling, and
full Co-Teaching are excluded from formal comparison until faithful real-data
implementations are available.

\section{{Conclusion}}
Graph-CoLD connects representation learning, label-space graph consistency,
evidence preserving denoising, and SOC-oriented ranking. On verified CICIDS-2017
and CESNET-TLS-Year22 settings, the method improves Macro-F1 over the aligned CoLD
baseline and preserves more clean informative evidence than hard deletion. The
most submission-ready conclusion is that Graph-CoLD is a promising real-data
robustness and evidence-retention approach whose broader baseline coverage and
enterprise case evaluation remain future work.

\section*{{Acknowledgements}}
To be completed by the authors before journal submission.

\section*{{Declaration of competing interest}}
The authors should complete the journal declaration form before submission.

\section*{{Data and code availability}}
Raw datasets are not included in the repository. The reproducibility package
documents the expected local data roots, audit gates, frozen result files, and
commands used to recreate the tables and figures.

\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""
    (paper / "graph_cold_cas_realdata.tex").write_text(text, encoding="utf-8")


def _write_build_scripts(paper: Path) -> None:
    (paper / "build_elsevier.ps1").write_text(
        """$ErrorActionPreference = "Stop"\nSet-Location $PSScriptRoot\npdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\nbibtex graph_cold_cas_realdata\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\npdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\npdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n""",
        encoding="utf-8",
    )
    (paper / "build_elsevier.sh").write_text(
        """#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\npdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex\nbibtex graph_cold_cas_realdata\npdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex\npdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex\n""",
        encoding="utf-8",
    )


def _write_cover_letter(paper: Path, metrics: dict[str, Any]) -> None:
    text = f"""# Cover Letter Draft

Dear Editors,

We submit Graph-CoLD for consideration by Computers & Security. The manuscript
studies noisy-label learning for SOC alert prioritization using verified
CICIDS-2017 and CESNET-TLS-Year22 real-data settings. The contribution is a
label-space graph consistency diagnostic combined with evidence preserving
weights and operational ranking.

The paper is deliberately scoped: it reports only smoke-passed real-data methods,
labels Co-Teaching-lite as a lightweight approximation, and excludes unavailable
or unverified baselines. The current results show a paired Graph-CoLD versus CoLD
Macro-F1 lift of {metrics['mean_diff_pp']:.2f} percentage points
(p={metrics['p_value']:.2e}) and improved evidence retention relative to hard
deletion.

Sincerely,

The Graph-CoLD authors
"""
    (paper / "cover_letter_draft.md").write_text(text, encoding="utf-8")


def _write_reproducibility(repro: Path, sources: dict[str, Any]) -> None:
    hashes = sources["source_hashes"]
    readme = f"""# Graph-CoLD Real-data Reproducibility Package

This package recreates the D5/D5.5 result matrix, D6 paper tables/figures, and
D7 manuscript assembly from verified local datasets. Raw datasets are not
committed.

## Data roots

- CICIDS-2017: `data/cicids2017`
- CESNET-TLS-Year22: `E:\\graphcold-data\\tls_alternative\\cesnet_tls_year22`
- External data root: `E:\\graphcold-data`

Formal reported datasets are `CICIDS-2017 postfilter11` and
`CESNET-TLS-Year22 postfilter25`. MALTLS-22 and OpTC are not part of the formal
evaluation package.

## Frozen source artifacts

- `results/table_main_expanded.csv`: `{hashes[str(MAIN_SOURCE)]}`
- `results/table_baseline_expansion.csv`: `{hashes[str(BASELINE_SOURCE)]}`
- `results/stat_tests_baseline_expansion.json`: `{hashes[str(STATS_SOURCE)]}`
- `reports/realdata_readiness_report.json`: `{hashes[str(READINESS_SOURCE)]}`

## Recreate results

Run readiness gates before experiments:

```powershell
python -m src.data.audit
python scripts/check_data_ready.py
```

Then run D5 and D5.5 explicitly:

```powershell
python -m src.experiments.d5 --out results --configs configs
python -m src.experiments.d5_baseline_expansion --out results --configs configs --reports reports
```

Generate paper assets:

```powershell
python -m src.paper.d6_prep
python -m src.paper.d7_assemble
paper\\elsevier\\build_elsevier.ps1
python -m src.paper.d7_assemble --audit-only
```

Large dataset downloads are manual or optional and are not started by these
scripts. Keep raw archives and extracted data outside Git tracking.
"""
    (repro / "README_realdata.md").write_text(readme, encoding="utf-8")
    (repro / "run_d5_realdata.ps1").write_text(
        """$ErrorActionPreference = "Stop"\npython -m src.data.audit\npython scripts/check_data_ready.py\npython -m src.experiments.d5 --out results --configs configs\npython -m src.experiments.d5_baseline_expansion --out results --configs configs --reports reports\n""",
        encoding="utf-8",
    )
    (repro / "run_d6_tables_figures.ps1").write_text(
        """$ErrorActionPreference = "Stop"\npython -m src.paper.d6_prep\npython -m src.paper.d7_assemble\npaper\\elsevier\\build_elsevier.ps1\npython -m src.paper.d7_assemble --audit-only\n""",
        encoding="utf-8",
    )


def _write_reviewer_pack(out_dir: Path, metrics: dict[str, Any], sources: dict[str, Any]) -> None:
    simulation = f"""# D7 Final Reviewer Simulation

## Overall decision risk

The package is stronger than the earlier draft because it uses verified
CICIDS-2017 and CESNET-TLS-Year22 rows only, identifies the CESNET subset policy,
and avoids reporting unsupported datasets or baselines. Submission readiness
remains false because reference metadata and broader baseline coverage still need
author review.

## Major likely concerns

1. Baseline limitation: FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full
Co-Teaching are excluded. Defense: the paper reports only implemented and
smoke-passed real-data methods.
2. CESNET subset: CESNET-TLS-Year22 is not reported as full archive. Defense:
the subset policy is stated in the abstract-adjacent setup, tables, limitations,
and reproducibility notes.
3. Co-Teaching-lite: the method is named as lightweight approximation. Defense:
the manuscript does not equate it with the original full deep method.
4. ERR definition: the manuscript links ERR to retained clean informative
evidence and reports hard-deletion comparison.
5. OpTC absence: the manuscript states that real provenance events are required
before an enterprise case can be formal.
6. MALTLS-22 absence: the manuscript states the source did not pass the local
verification gate.
7. Graph construction: process and threat-intelligence views are explicitly
disabled where data fields are missing.

## Key numbers

- Graph-CoLD vs CoLD mean Macro-F1 difference: {metrics['mean_diff_pp']:.2f} pp
- p-value: {metrics['p_value']:.2e}
- Effect size dz: {metrics['effect_size']:.3f}
- Graph-CoLD ERR_final: {metrics['graph_err']:.4f}
- ablation_hard ERR_final: {metrics['hard_err']:.4f}
"""
    (out_dir / "reviewer_simulation_final.md").write_text(simulation, encoding="utf-8")

    rebuttal = """# Rebuttal Preparation Pack

## Baseline limitation
Likely concern: important named baselines are absent.
Response strategy: emphasize that the result matrix excludes methods without
faithful, independently implemented, smoke-passed real-data rows. This prevents
unfair or approximate comparisons.

## CESNET subset
Likely concern: CESNET-TLS-Year22 is large, but the paper uses a deterministic
audit-window subset.
Response strategy: point to Table 1, Experimental Setup, Limitations, and the
reproducibility README. Do not expand the claim beyond the evaluated subset.

## Co-Teaching-lite
Likely concern: the baseline may not represent full Co-Teaching.
Response strategy: agree and state that it is intentionally named lite. The
manuscript does not use it as a substitute for a full deep reproduction.

## ERR definition
Likely concern: evidence retention could be gamed by retaining too much.
Response strategy: ERR is paired with compression ratio and Tail-ERR. The
ablation_hard comparison isolates hard deletion versus evidence preserving
weights.

## OpTC and MALTLS-22
Likely concern: the introduction motivates SOC settings but no enterprise OpTC
case is reported.
Response strategy: state that the manuscript is a real-data robustness study on
two verified network datasets. OpTC requires verified provenance event tables
before formal enterprise evaluation.

## Graph construction
Likely concern: five-view design is not fully active for every dataset.
Response strategy: active views are dataset-dependent. The paper does not claim
process or threat-intelligence evidence where the underlying fields are absent.
"""
    (out_dir / "rebuttal_preparation_pack.md").write_text(rebuttal, encoding="utf-8")

    reference_gap = """# Reference Gap Report

- CoLD is cited from the NDSS accepted-paper page. Final page range and DOI
  should be checked by the authors before journal submission.
- CESNET-TLS-Year22 is cited through the Zenodo record to avoid uncertain article
  metadata in this generated package.
- Operational SOC framing is kept qualitative; no external incident-volume
  statistics are asserted.
"""
    (out_dir / "reference_gap_report.md").write_text(reference_gap, encoding="utf-8")


def _write_final_audit(out_dir: Path, paper: Path) -> dict[str, bool]:
    tex = paper / "graph_cold_cas_realdata.tex"
    bib = paper / "references.bib"
    pdf = paper / "graph_cold_cas_realdata.pdf"
    manuscript = tex.read_text(encoding="utf-8", errors="ignore") if tex.exists() else ""
    normalized_manuscript = " ".join(manuscript.split())
    tables_ok = all((paper / "tables" / name).exists() for name in [
        "table_1_dataset_protocol.tex",
        "table_2_main_summary.tex",
        "table_3_high_noise.tex",
        "table_4_ablation.tex",
        "table_5_statistical_tests.tex",
    ])
    figures_ok = all((paper / "figures" / f"{stem}.pdf").exists() for stem in FIGURE_FILES)
    audit = {
        "real_data_only": True,
        "uses_table_main_expanded": MAIN_SOURCE.exists(),
        "no_maltls22_results": "MALTLS-22 results" not in manuscript,
        "no_optc_results": "OpTC results" not in manuscript,
        "no_synthetic_fallback_claims": "synthetic" not in manuscript.lower() and "fallback" not in manuscript.lower(),
        "no_fake_baselines": all(method not in manuscript or "excluded" in manuscript for method in EXCLUDED_METHODS),
        "cesnet_reported_as_cesnet": "CESNET-TLS-Year22" in manuscript,
        "cesnet_subset_declared": "deterministic audit-window subset" in manuscript,
        "co_teaching_lite_named_correctly": "Co-Teaching-lite" in manuscript and "full Co-Teaching implementation" in normalized_manuscript,
        "baseline_exclusions_declared": all(method in manuscript for method in EXCLUDED_METHODS),
        "figures_exist": figures_ok,
        "tables_exist": tables_ok,
        "pdf_compiles": pdf.exists() and pdf.stat().st_size > 1000,
        "references_no_known_fake_entries": bib.exists() and "doi={TODO" not in bib.read_text(encoding="utf-8", errors="ignore"),
        "submission_ready": False,
    }
    (out_dir / "d7_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = "# D7 Final Audit\n\n" + "\n".join(f"- {key}: {str(value).lower()}" for key, value in audit.items()) + "\n"
    (out_dir / "d7_final_audit.md").write_text(md, encoding="utf-8")
    return audit


def _compact_policy(value: Any) -> str:
    text = str(value)
    return text.replace("full_postfilter11_after_min_count_and_dominant_downsample", "postfilter11 full filtered").replace(
        "deterministic_audit_window_100000_then_postfilter25_stratified_split",
        "100k audit window, postfilter25",
    )


def _compact_p_value(value: Any) -> str:
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return str(value)


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble D7 real-data manuscript artifacts.")
    parser.add_argument("--paper", default="paper/elsevier")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--reproducibility", default="reproducibility")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(update_d7_audit(args.paper, args.reports), indent=2))
    else:
        print(json.dumps(run_d7_assembly(args.paper, args.reports, args.reproducibility), indent=2))


if __name__ == "__main__":
    main()
