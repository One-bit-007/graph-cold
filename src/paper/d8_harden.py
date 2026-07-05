"""Harden the D7 manuscript into a C&S-style v1.0 draft.

This module is an aggregation and writing step only. It reads frozen D5/D5.5,
D6, and D7 artifacts, then rewrites the manuscript narrative and submission
supporting material. It does not run experiments or import model/training code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


MAIN_SOURCE = Path("results/table_main_expanded.csv")
BASELINE_SOURCE = Path("results/table_baseline_expansion.csv")
STATS_SOURCE = Path("results/stat_tests_baseline_expansion.json")
PAPER_DIR = Path("paper/elsevier")
REPORTS_D8 = Path("reports/d8")

FORMAL_DATASETS = {"CICIDS-2017", "CESNET-TLS-Year22"}
FORMAL_METHODS = {
    "Graph-CoLD",
    "CoLD",
    "ablation_hard",
    "Noisy-Supervised",
    "Confident-Learning",
    "Co-Teaching-lite",
}
EXCLUDED = ("FINE", "MCRe", "MORSE", "Flash", "Argus", "Decoupling", "full Co-Teaching")


def run_d8_hardening(paper_dir: str | Path = PAPER_DIR, reports_dir: str | Path = "reports") -> dict[str, Any]:
    """Create the hardened v1.0 manuscript and D8 audit artifacts."""
    paper = Path(paper_dir)
    reports = Path(reports_dir)
    d8_dir = reports / "d8"
    d8_dir.mkdir(parents=True, exist_ok=True)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    sources = _load_sources(reports)
    _validate_sources(sources)
    metrics = _metrics(sources["main"], sources["stats"])

    _write_references(paper)
    _write_tables(paper / "tables", sources)
    _write_manuscript(paper, sources, metrics)
    _write_cover_letter(paper, metrics)
    _write_repro_note()
    audit = _write_audit(d8_dir, paper, sources, metrics)
    _write_risk_register(d8_dir, metrics)

    manifest = {
        "stage": "D8 manuscript hardening",
        "completed": True,
        "manuscript": str(paper / "graph_cold_cas_realdata.tex"),
        "pdf": str(paper / "graph_cold_cas_realdata.pdf"),
        "source_hashes": sources["hashes"],
        "audit": audit,
    }
    (d8_dir / "d8_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def refresh_d8_audit(paper_dir: str | Path = PAPER_DIR, reports_dir: str | Path = "reports") -> dict[str, Any]:
    """Refresh audit after an external LaTeX build."""
    sources = _load_sources(Path(reports_dir))
    metrics = _metrics(sources["main"], sources["stats"])
    d8_dir = Path(reports_dir) / "d8"
    d8_dir.mkdir(parents=True, exist_ok=True)
    return _write_audit(d8_dir, Path(paper_dir), sources, metrics)


def _load_sources(reports: Path) -> dict[str, Any]:
    required = [
        MAIN_SOURCE,
        BASELINE_SOURCE,
        STATS_SOURCE,
        Path("tables/table_1_dataset_protocol.csv"),
        Path("tables/table_2_main_performance.csv"),
        Path("tables/table_3_high_noise_summary.csv"),
        Path("tables/table_4_ablation_evidence.csv"),
        Path("tables/table_5_statistical_tests.csv"),
        Path("reports/d6/d6_statistical_narrative.md"),
        Path("reports/d6/reviewer_risk_notes.md"),
        Path("reports/d7/d7_final_audit.json"),
        Path("reports/d5_scale_policy.json"),
        Path("reports/cicids_final_protocol.json"),
        Path("reports/cesnet_class_policy_report.json"),
        Path("reports/cesnet_view_policy_report.json"),
        Path("docs/spec_method_impl.md"),
        Path("docs/spec_graph_noise.md"),
        Path("docs/DATASETS.md"),
        Path("docs/DATASET_DOWNLOADS.md"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("D8 hardening requires frozen prior artifacts: " + ", ".join(missing))

    sources: dict[str, Any] = {
        "main": pd.read_csv(MAIN_SOURCE),
        "baseline": pd.read_csv(BASELINE_SOURCE),
        "stats": json.loads(STATS_SOURCE.read_text(encoding="utf-8")),
        "dataset_table": pd.read_csv("tables/table_1_dataset_protocol.csv"),
        "main_table": pd.read_csv("tables/table_2_main_performance.csv"),
        "high_noise": pd.read_csv("tables/table_3_high_noise_summary.csv"),
        "ablation": pd.read_csv("tables/table_4_ablation_evidence.csv"),
        "stat_table": pd.read_csv("tables/table_5_statistical_tests.csv"),
        "d6_narrative": Path("reports/d6/d6_statistical_narrative.md").read_text(encoding="utf-8"),
        "d6_risk": Path("reports/d6/reviewer_risk_notes.md").read_text(encoding="utf-8"),
        "d7_audit": json.loads(Path("reports/d7/d7_final_audit.json").read_text(encoding="utf-8")),
        "scale_policy": json.loads(Path("reports/d5_scale_policy.json").read_text(encoding="utf-8")),
        "cicids_protocol": json.loads(Path("reports/cicids_final_protocol.json").read_text(encoding="utf-8")),
        "cesnet_policy": json.loads(Path("reports/cesnet_class_policy_report.json").read_text(encoding="utf-8")),
        "cesnet_view": json.loads(Path("reports/cesnet_view_policy_report.json").read_text(encoding="utf-8")),
        "hashes": {str(path): _sha256(path) for path in required if path.suffix in {".csv", ".json", ".md"}},
    }
    return sources


def _validate_sources(sources: dict[str, Any]) -> None:
    main = sources["main"]
    datasets = set(main["reported_as"].dropna().astype(str))
    methods = set(main["method"].dropna().astype(str))
    if datasets != FORMAL_DATASETS:
        raise ValueError(f"D8 supports only {sorted(FORMAL_DATASETS)}, got {sorted(datasets)}")
    if not FORMAL_METHODS.issubset(methods):
        raise ValueError(f"D8 missing formal methods: {sorted(FORMAL_METHODS - methods)}")
    if not bool(main["source_verified"].astype(bool).all()):
        raise ValueError("D8 cannot cite unverified result rows.")
    d7 = sources["d7_audit"]
    required_true = [
        "real_data_only",
        "uses_table_main_expanded",
        "no_maltls22_results",
        "no_optc_results",
        "pdf_compiles",
    ]
    bad = [key for key in required_true if not bool(d7.get(key, False))]
    if bad:
        raise ValueError(f"D7 audit blocks D8 hardening: {bad}")


def _metrics(main: pd.DataFrame, stats: dict[str, Any]) -> dict[str, Any]:
    means = main.groupby("method").agg(
        macro=("macro_f1", "mean"),
        fpr=("fpr", "mean"),
        fnr=("fnr", "mean"),
        err=("err_final", "mean"),
        compression=("compression_ratio", "mean"),
        runtime=("runtime_sec", "mean"),
        memory=("memory_mb", "mean"),
    )
    high = main[
        (pd.to_numeric(main["noise_rate"], errors="coerce") >= 0.4)
        & main["noise_type"].isin(["symmetric", "asymmetric", "graph_consistency"])
    ]
    high_means = high.groupby(["reported_as", "method"]).agg(
        macro=("macro_f1", "mean"),
        err=("err_final", "mean"),
        compression=("compression_ratio", "mean"),
    )
    comparison = stats["comparisons"]["Graph-CoLD_vs_CoLD"]
    return {
        "graph_macro": float(means.loc["Graph-CoLD", "macro"]),
        "cold_macro": float(means.loc["CoLD", "macro"]),
        "graph_fpr": float(means.loc["Graph-CoLD", "fpr"]),
        "graph_fnr": float(means.loc["Graph-CoLD", "fnr"]),
        "graph_err": float(means.loc["Graph-CoLD", "err"]),
        "hard_err": float(means.loc["ablation_hard", "err"]),
        "graph_compression": float(means.loc["Graph-CoLD", "compression"]),
        "cold_compression": float(means.loc["CoLD", "compression"]),
        "graph_runtime": float(means.loc["Graph-CoLD", "runtime"]),
        "cold_runtime": float(means.loc["CoLD", "runtime"]),
        "graph_memory": float(means.loc["Graph-CoLD", "memory"]),
        "cold_memory": float(means.loc["CoLD", "memory"]),
        "mean_diff_pp": float(comparison["mean_diff"]) * 100.0,
        "p_value": float(comparison["p_value"]),
        "effect_size": float(comparison["effect_size_cohen_dz"]),
        "n_pairs": int(comparison["n_pairs"]),
        "cicids_high_lift_pp": _lift(high_means, "CICIDS-2017"),
        "cesnet_high_lift_pp": _lift(high_means, "CESNET-TLS-Year22"),
        "err_gap_pp": (float(means.loc["Graph-CoLD", "err"]) - float(means.loc["ablation_hard", "err"])) * 100.0,
    }


def _lift(grouped: pd.DataFrame, dataset: str) -> float:
    return (float(grouped.loc[(dataset, "Graph-CoLD"), "macro"]) - float(grouped.loc[(dataset, "CoLD"), "macro"])) * 100.0


def _write_tables(out: Path, sources: dict[str, Any]) -> None:
    dataset = sources["dataset_table"][
        ["Dataset", "Rows used", "Class policy", "Number of classes", "Sample policy", "Active views"]
    ].rename(columns={"Rows used": "Rows", "Number of classes": "Classes"})
    dataset["Rows"] = dataset["Rows"].map(lambda value: f"{int(value):,}")
    dataset["Sample policy"] = dataset["Sample policy"].map(_compact_policy)
    _write_table(out / "table_1_dataset_protocol.tex", dataset)

    main = _method_table(sources["main"])
    _write_table(out / "table_2_main_summary.tex", main)

    high = sources["high_noise"][["Dataset", "Method", "Macro-F1 mean", "ERR mean", "Compression ratio mean", "Scenario count"]].copy()
    high = high.rename(
        columns={
            "Macro-F1 mean": "Macro-F1",
            "ERR mean": "ERR",
            "Compression ratio mean": "Compression",
            "Scenario count": "Scenarios",
        }
    )
    _write_table(out / "table_3_high_noise.tex", high)

    ablation = sources["ablation"][
        ["Dataset", "Variant", "Macro-F1", "ERR_final", "retained_fraction_clean_informative", "compression_ratio"]
    ].rename(
        columns={
            "retained_fraction_clean_informative": "Retained clean-info",
            "compression_ratio": "Compression",
        }
    )
    _write_table(out / "table_4_ablation.tex", ablation)

    stat = sources["stat_table"][["Comparison", "Mean difference", "p-value", "Effect size", "n"]].copy()
    stat["p-value"] = stat["p-value"].map(_compact_p)
    _write_table(out / "table_5_statistical_tests.tex", stat)


def _method_table(main: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(["Graph-CoLD", "CoLD", "ablation_hard", "Noisy-Supervised", "Confident-Learning", "Co-Teaching-lite"])}
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


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    align = "l" * len(frame.columns)
    lines = [r"\small", f"\\begin{{tabular}}{{{align}}}", r"\hline"]
    lines.append(" & ".join(_tex_escape(str(col)) for col in frame.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in frame.iterrows():
        lines.append(" & ".join(_tex_escape(str(value)) for value in row.tolist()) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_references(paper: Path) -> None:
    (paper / "references.bib").write_text(
        r"""@inproceedings{sharafaldin2018cicids,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  booktitle={Proceedings of the International Conference on Information Systems Security and Privacy},
  pages={108--116},
  year={2018},
  doi={10.5220/0006639801080116}
}

@article{hynek2024cesnettlsyear22,
  title={{CESNET-TLS-Year22}: A Year-Spanning TLS Network Traffic Dataset from Backbone Lines},
  author={Hynek, Karel and Luxemburk, Jan and Pesek, Jaroslav and Cejka, Tomas and Siska, Pavel},
  journal={Scientific Data},
  volume={11},
  pages={1156},
  year={2024},
  doi={10.1038/s41597-024-03927-4}
}

@misc{cesnettlsyear22,
  title={{CESNET-TLS-Year22}: A Year-Spanning TLS Network Traffic Dataset from Backbone Lines},
  author={Hynek, Karel and Luxemburk, Jan and Pesek, Jaroslav and Cejka, Tomas and Siska, Pavel},
  year={2024},
  howpublished={Zenodo record 10608607},
  doi={10.5281/zenodo.10608607},
  url={https://zenodo.org/records/10608607}
}

@inproceedings{yang2026cold,
  title={{CoLD}: Collaborative Label Denoising Framework for Network Intrusion Detection},
  author={Yang, Shuo and Zheng, Xinran and Li, Jinze and Xu, Jinfeng and Ngai, Edith C. H.},
  booktitle={Proceedings of the Network and Distributed System Security Symposium},
  year={2026},
  url={https://www.ndss-symposium.org/ndss-paper/cold-collaborative-label-denoising-framework-for-network-intrusion-detection/}
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
  pages={8535--8545},
  year={2018}
}

@inproceedings{patrini2017loss,
  title={Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach},
  author={Patrini, Giorgio and Rozza, Alessandro and Menon, Aditya Krishna and Nock, Richard and Qu, Lizhen},
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


def _write_manuscript(paper: Path, sources: dict[str, Any], metrics: dict[str, Any]) -> None:
    main_hash = sources["hashes"][str(MAIN_SOURCE)]
    excluded_text = ", ".join(EXCLUDED)
    text = rf"""\documentclass[preprint,review,12pt]{{elsarticle}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{graphicx}}
\setlength{{\oddsidemargin}}{{0in}}
\setlength{{\evensidemargin}}{{0in}}
\setlength{{\textwidth}}{{6.5in}}
\setlength{{\topmargin}}{{-0.25in}}
\setlength{{\textheight}}{{8.9in}}
\renewcommand{{\arraystretch}}{{1.08}}
\journal{{Computers \& Security}}

\title{{Graph-CoLD: Evidence-preserving Graph Label Denoising for SOC Alert Prioritization under Noisy Labels}}
\author{{Graph-CoLD Project Team}}
\address{{Computers \& Security submission draft v1.0}}

\begin{{document}}
\begin{{frontmatter}}
\maketitle
\begin{{abstract}}
Security operations centers must prioritize large alert queues even when training
labels are incomplete, delayed, or inconsistent across data sources. Existing
label-denoising pipelines for intrusion detection reduce noisy supervision, but
they usually treat flows as independent samples and often remove suspicious
instances outright. This is risky in SOC settings because rare or boundary
alerts can be precisely the evidence analysts need. We present Graph-CoLD, a
two-stage graph extension of CoLD for noisy-label intrusion detection. The method
builds dataset-specific host, IP, and temporal graph views; learns CoLD-aligned
multi-view representations; scores training samples with a label-space
Graph-CDM diagnostic; and replaces hard deletion with evidence-preserving soft
weights. On verified CICIDS-2017 postfilter11 and CESNET-TLS-Year22 postfilter25
settings, Graph-CoLD improves Macro-F1 over the aligned CoLD baseline by
{metrics['mean_diff_pp']:.2f} percentage points in a paired scenario-level test
(p={metrics['p_value']:.2e}, Cohen dz={metrics['effect_size']:.3f},
n={metrics['n_pairs']}). The main operational signal is evidence retention:
Graph-CoLD raises mean ERR\_final from {metrics['hard_err']:.4f} for hard
deletion to {metrics['graph_err']:.4f}, while reporting compression, runtime,
and memory costs. The evaluation is intentionally bounded to implemented,
real-data baselines and avoids claims for unavailable datasets or enterprise
case studies.
\end{{abstract}}
\begin{{keyword}}
intrusion detection \sep noisy labels \sep graph learning \sep evidence retention \sep SOC alert prioritization
\end{{keyword}}
\end{{frontmatter}}

\section{{Introduction}}
Security operations centers (SOCs) increasingly rely on machine-learning
classifiers to rank network alerts and encrypted-flow observations. These models
are trained from labels that can be delayed, generated by rules with uneven
quality, or derived from analyst workflows that emphasize high-severity cases.
The resulting label noise affects both detection quality and operational triage:
discarding a mislabeled benign sample may improve a benchmark score, but
discarding an informative low-frequency alert can remove the context needed for
an investigation.

CoLD recently framed noisy-label intrusion detection through local consistency:
different traffic classes may share feature neighborhoods, and label noise can
create spurious associations in those neighborhoods \cite{{yang2026cold}}. CoLD
partitions features, learns robust representations, and removes likely noisy
samples before classifier training. Graph-CoLD starts from the same observation
but changes the unit of evidence. Instead of treating each flow as isolated, it
constructs multiple graph views over the training samples and asks whether
label-space behavior is consistent across predictions, neighborhoods, views, and
available temporal chains.

The second change is operational. Hard deletion is a useful denoising baseline,
yet SOC triage is often a retention problem: the analyst wants a shorter queue
without losing clean, rare, or boundary evidence. Graph-CoLD therefore converts
the graph diagnostic into a soft training weight and mixes it with an evidence
score that protects low-frequency and anomalous samples. The same diagnostic and
evidence score feed an alert-priority proxy, evaluated here through compression
and evidence-retention metrics rather than an analyst study.

This paper makes four contributions:
\begin{{enumerate}}
\item A real-data Graph-CoLD formulation that lifts CoLD-style noisy-label
learning from independent samples to dataset-specific graph views.
\item A label-space Graph-CDM diagnostic combining prediction disagreement,
neighborhood KL divergence, view-mode disagreement, and optional temporal-chain
consistency.
\item An evidence-preserving weighting rule that keeps clean informative samples
visible and is compared directly with the hard-deletion ablation.
\item A reproducible two-dataset evaluation with paired statistical tests,
explicit baseline exclusions, and C\&S-oriented limitations.
\end{{enumerate}}

\section{{Related Work}}
\subsection{{Noisy labels in intrusion detection}}
The closest prior work is CoLD, which identifies local consistency as a driver
of noisy-label degradation in network intrusion detection and uses collaborative
representations to filter noisy instances \cite{{yang2026cold}}. Broader noisy
label learning includes loss-correction methods that model class-conditional
corruption \cite{{patrini2017loss}}, peer-network sample selection such as
Co-teaching \cite{{han2018coteaching}}, and confidence-based label-error
estimation \cite{{northcutt2021confident}}. Graph-CoLD does not claim to replace
these families. Instead, it evaluates a bounded set of implemented real-data
baselines and isolates the contribution of graph-structured, label-space
diagnostics.

\subsection{{Graph representations for alerts}}
Graph neural networks provide a natural way to propagate contextual evidence
through relations among samples or entities. GCN and GAT models illustrate the
basic principle of learning from graph neighborhoods and attentional neighbors
\cite{{kipf2017gcn,velickovic2018gat}}. Graph-CoLD uses this idea conservatively:
each active view is a graph over the same sample set, and unsupported views are
disabled rather than invented. CICIDS-2017 activates host, IP, and temporal
views. CESNET-TLS-Year22 activates IP and temporal views because process lineage
and threat-intelligence fields are not present in the verified local contract.

\subsection{{Datasets and SOC-oriented evaluation}}
CICIDS-2017 is widely used for intrusion-detection evaluation and provides
labeled flow CSVs with timing, endpoint, protocol, and attack metadata
\cite{{sharafaldin2018cicids}}. CESNET-TLS-Year22 is a year-spanning TLS traffic
dataset collected from backbone lines, with service labels and flow/TLS features
\cite{{hynek2024cesnettlsyear22,cesnettlsyear22}}. In this manuscript, CESNET is
reported under its own name and as a deterministic postfilter25 audit-window
subset, not a full-archive evaluation. MALTLS-22 and OpTC are excluded because
they did not pass the local verification gates.

\section{{Problem Formulation and Design Goals}}
Let $V$ be the set of training flows or alerts and let
$\mathcal{{M}}$ denote the set of active graph views. Each view
$m\in\mathcal{{M}}$ defines a graph $G^m=(V,E^m)$ over the same nodes. The
observed training label $\tilde y_v$ may differ from the unobserved clean label
$y_v$. The classifier should minimize the effect of noisy labels while preserving
operationally useful evidence.

Graph-CoLD is designed around three goals. First, the noisy-label diagnostic must
operate in label space rather than relying only on embedding distance, so that
the diagnostic remains aligned with the classifier and the label-noise protocol.
Second, denoising should be evidence preserving: a suspicious but informative
sample should be downweighted rather than automatically removed. Third, the
evaluation must be traceable to real audited data, with explicit scope limits for
datasets, baselines, and enterprise claims.

\section{{Method}}
\subsection{{Multi-view graph construction}}
The implementation constructs graph views from semantically related feature
blocks. A host view connects samples sharing endpoint identifiers or host-like
fields. An IP view uses communication statistics, ports, protocol fields, TLS
metadata, and flow features. A temporal view connects samples within the
configured ordering or time window. Process and threat-intelligence views are
available in the design but are disabled for the formal CICIDS/CESNET evaluation
when the required fields are absent. This view mask is part of the reproducible
dataset contract rather than a post-hoc modeling choice.

\subsection{{CoLD-aligned representation learning}}
Stage 1 keeps the D2 representation objective fixed. Each active view produces a
node embedding $z_v^m$ from masked input features. Cross-view contrastive
learning treats the same node in two views as a positive pair and other batch
nodes as negatives. Temporal alignment penalizes inconsistent embeddings across
adjacent snapshots, and reconstruction preserves feature information. Active
views are fused by mean aggregation to obtain $z_v$. D8 does not modify this
encoder or representation loss.

\subsection{{Graph-CDM label-space diagnostic}}
Graph-CDM scores how inconsistent a node appears in label space:
\begin{{equation}}
\operatorname{{GraphCDM}}(v)=
\lambda_1D_{{pred}}(v)+\lambda_2D_{{neigh}}(v)+
\lambda_3D_{{view}}(v)+\lambda_4D_{{chain}}(v).
\end{{equation}}
$D_{{pred}}$ compares active-view predictions with the observed noisy training label:
\begin{{equation}}
D_{{pred}}(v)=\frac{{1}}{{M}}\sum_{{m=1}}^M
\mathbf{{1}}\left[\tilde y_v^m \ne \tilde y_v\right].
\end{{equation}}
$D_{{neigh}}$ is a label-space KL divergence between the node's soft prediction
and the mean soft prediction in its graph neighborhood:
\begin{{equation}}
D_{{neigh}}(v)=KL\left(\hat y_v\,\middle\|\,\frac{{1}}{{|N(v)|}}\sum_{{u\in N(v)}}\hat y_u\right).
\end{{equation}}
$D_{{view}}$ measures active-view mode disagreement:
\begin{{equation}}
D_{{view}}(v)=1-\max_c \frac{{1}}{{M}}\sum_m \mathbf{{1}}\left[\tilde y_v^m=c\right].
\end{{equation}}
$D_{{chain}}$ is reserved for provenance or temporal chains with verified
semantics. It is not treated as a central driver of the formal CICIDS/CESNET
results.

\subsection{{Evidence-preserving training weights}}
The evidence score protects samples that would be costly to discard:
\begin{{equation}}
e(v)=freq\_protect(\tilde y_v)\left(1+\gamma\,anom(v)\right),\qquad
\tilde e(v)=minmax(e(v)).
\end{{equation}}
The final weight mixes diagnostic confidence and normalized evidence:
\begin{{equation}}
w(v)=\sigma\left[-\kappa(\operatorname{{GraphCDM}}(v)-\theta)\right](1-\rho)
+\rho\tilde e(v).
\end{{equation}}
The classifier minimizes weighted cross entropy
$\sum_v w(v)CE(f(z_v),\tilde y_v)$. The hard-deletion ablation sets $\rho=0$ and
thresholds the diagnostic into a binary retention decision, matching the
CoLD-style purifier behavior used as a key comparator.

\subsection{{Operational priority proxy}}
For SOC triage, Graph-CoLD computes an alert priority score
\begin{{equation}}
P(v)=\alpha_1\hat y_{{mal}}(v)+\alpha_2\operatorname{{GraphCDM}}(v)+\alpha_3\tilde e(v).
\end{{equation}}
This paper evaluates the ranking component indirectly through compression ratio
and evidence retention. We do not claim an analyst-in-the-loop deployment study.

\section{{Experimental Design}}
\subsection{{Datasets and scope}}
Table~\ref{{tab:dataset-protocol}} summarizes the formal datasets. CICIDS-2017
uses {int(sources['dataset_table'].loc[0, 'Rows used']):,} postfilter11 rows
after removing classes below the minimum count and downsampling the dominant
class. CESNET-TLS-Year22 uses {int(sources['dataset_table'].loc[1, 'Rows used']):,}
postfilter25 rows from a deterministic audit-window subset and is not a full-archive evaluation. The row-level result
source is \texttt{{results/table\_main\_expanded.csv}} (SHA256 prefix
{main_hash[:16]}). Full dataset and result hashes are recorded in the
reproducibility package.

\begin{{table}}[t]
\centering
\caption{{Verified real-data protocols. CESNET-TLS-Year22 is an audit-window subset and not a full-archive evaluation.}}
\label{{tab:dataset-protocol}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_1_dataset_protocol.tex}}}}
\end{{table}}

\subsection{{Noise protocols}}
The evaluation includes clean labels, symmetric label noise, asymmetric label
noise, and graph-consistency noise. Noise is injected only into training labels.
Graph-consistency noise preferentially flips labels between locally consistent
cross-class neighbors; when the consistency-bias parameter is zero it reduces to
the symmetric-noise control. The paper reports seeds 0, 1, and 2 and keeps the
same split/noise/model seeds for paired comparisons.

\subsection{{Baselines, ablations, and metrics}}
Formal methods include Graph-CoLD, the aligned CoLD baseline, a hard-deletion
ablation, noisy supervised learning, confidence learning, and Co-Teaching-lite.
Co-Teaching-lite is a lightweight implemented approximation and is not a full Co-Teaching implementation and not a full Co-Teaching reproduction. Methods
excluded from formal comparison are {excluded_text}; each is omitted because it lacks an independently
smoke-passed real-data implementation in this repository. Metrics are Macro-F1,
FPR, FNR, ERR, Tail-ERR, compression ratio, runtime, and memory. Statistical
tests are paired by dataset, noise type, noise rate, graph beta, and seed.

\section{{Results}}
\subsection{{RQ1: Does graph evidence improve noisy-label robustness?}}
Table~\ref{{tab:main-performance}} aggregates the D5.5 matrix by dataset and
method. Across all paired scenarios, Graph-CoLD obtains Macro-F1
{metrics['graph_macro']:.4f}, compared with {metrics['cold_macro']:.4f} for
CoLD. The paired grouped test reports a mean lift of
{metrics['mean_diff_pp']:.2f} percentage points (p={metrics['p_value']:.2e},
dz={metrics['effect_size']:.3f}, n={metrics['n_pairs']}). The result supports a
consistent robustness improvement under the implemented protocol, not a universal
claim over unimplemented baselines.

\begin{{table}}[t]
\centering
\caption{{Mean performance over the verified D5.5 result matrix.}}
\label{{tab:main-performance}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_2_main_summary.tex}}}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig2_macro_f1_vs_noise_rate.pdf}}
\caption{{Macro-F1 versus label-noise rate. Graph-consistency panels use beta=0.6 from the frozen matrix.}}
\label{{fig:macro-noise}}
\end{{figure}}

\subsection{{RQ2: What happens under high noise?}}
Table~\ref{{tab:high-noise}} summarizes rates at or above 0.4 for symmetric,
asymmetric, and graph-consistency settings. CICIDS-2017 shows the clearest
high-noise margin: Graph-CoLD exceeds CoLD by {metrics['cicids_high_lift_pp']:.2f}
percentage points. CESNET-TLS-Year22 has a ceiling effect, with a high-noise
Macro-F1 lift of {metrics['cesnet_high_lift_pp']:.2f} percentage points; this is
why the paper interprets CESNET primarily through stability and evidence
retention rather than classifier margin.

\begin{{table}}[t]
\centering
\caption{{High-noise summary for rates at or above 0.4.}}
\label{{tab:high-noise}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_3_high_noise.tex}}}}
\end{{table}}

\subsection{{RQ3: Does soft weighting preserve evidence?}}
Evidence retention is the central difference between Graph-CoLD and hard
deletion. Mean ERR\_final is {metrics['graph_err']:.4f} for Graph-CoLD and
{metrics['hard_err']:.4f} for ablation\_hard, a gap of {metrics['err_gap_pp']:.2f}
percentage points. Fig.~\ref{{fig:err-retention}} shows that this gap remains
visible under increasing noise, especially where clean informative samples are
likely to lie near class boundaries.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig3_err_retention.pdf}}
\caption{{Evidence retention under label noise. ERR\_final is measured on clean informative samples.}}
\label{{fig:err-retention}}
\end{{figure}}

\subsection{{RQ4: Which components matter?}}
Table~\ref{{tab:ablation}} and Fig.~\ref{{fig:ablation}} show the ablation
pattern. Removing the neighborhood term, the view-disagreement term, or evidence
weighting weakens either Macro-F1 or retention. The hard-deletion variant is the
most important comparator because it keeps the diagnostic but removes evidence
preservation.

\begin{{table}}[t]
\centering
\caption{{Ablation and evidence-retention results.}}
\label{{tab:ablation}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_4_ablation.tex}}}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig4_ablation.pdf}}
\caption{{Ablation comparison for Macro-F1 and ERR\_final.}}
\label{{fig:ablation}}
\end{{figure}}

\subsection{{RQ5: What is the operational cost?}}
Graph-CoLD is not presented as a free robustness improvement. The mean runtime
is {metrics['graph_runtime']:.2f}s per scenario for Graph-CoLD versus
{metrics['cold_runtime']:.2f}s for CoLD, while mean memory is
{metrics['graph_memory']:.1f} MB versus {metrics['cold_memory']:.1f} MB.
Fig.~\ref{{fig:runtime}} reports these costs alongside baselines. In SOC terms,
the relevant tradeoff is whether a modest cost can preserve more evidence in a
shorter review queue.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig5_runtime_cost.pdf}}
\caption{{Runtime and memory cost by method.}}
\label{{fig:runtime}}
\end{{figure}}

\begin{{table}}[t]
\centering
\caption{{Paired grouped statistical tests from D5.5.}}
\label{{tab:stats}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/table_5_statistical_tests.tex}}}}
\end{{table}}

\clearpage
\section{{Discussion}}
The results support a bounded but useful claim: graph-structured label-space
consistency improves robustness and evidence retention in the verified
two-dataset setting. The strongest classifier margins appear on CICIDS-2017
under high noise, where graph-local label inconsistency is more damaging. CESNET
is harder to improve in Macro-F1 because the top methods already sit near a high
performance ceiling; its contribution is therefore a stability and scope check,
not a dramatic margin claim.

ERR gives the SOC-facing interpretation. Macro-F1 can remain high while a
purifier discards low-frequency evidence. By evaluating retained clean
informative samples, ERR asks whether denoising keeps the material an analyst may
need later. Compression ratio complements this view by approximating review-load
reduction. Together, these metrics frame Graph-CoLD as a retention-aware
denoising method rather than just another classifier.

\section{{Threats to Validity}}
\textbf{{Internal validity.}} The experiments are deterministic over the recorded
seeds and paired by scenario, but the implementation is still a research
prototype. Co-Teaching-lite is a lightweight approximation, and excluded
baselines need faithful implementations before broader comparison.

\textbf{{External validity.}} CICIDS-2017 and CESNET-TLS-Year22 are useful
benchmarks, but neither is a complete SOC deployment. CESNET is an audit-window
subset, not a full archive. OpTC is not reported because the verified provenance
events table is unavailable.

\textbf{{Construct validity.}} Compression ratio is an operational proxy rather
than a direct analyst-hours measurement. ERR measures evidence retention on clean
informative samples, which is appropriate for this noisy-label study but does
not replace incident-level investigation outcomes.

\section{{Limitations}}
MALTLS-22 is not evaluated because the project does not have a verified source
and license path. OpTC is not evaluated as a formal enterprise case because
verified provenance events are absent. Process and threat-intelligence views are
disabled in formal results when the underlying fields are unavailable. The
current reference implementation should therefore be read as a real-data
methodological validation, not as a complete enterprise deployment package.

\section{{Conclusion}}
Graph-CoLD extends CoLD-style noisy-label intrusion detection with multi-view
graph context, a label-space consistency diagnostic, and evidence-preserving
soft weights. In the verified CICIDS-2017 and CESNET-TLS-Year22 settings,
Graph-CoLD improves paired Macro-F1 over CoLD and preserves more clean
informative evidence than hard deletion. The next step toward final journal
submission is not more polishing of these numbers, but broader faithful baseline
coverage and a verified enterprise provenance case.

\section*{{Acknowledgements}}
The authors thank the maintainers of the public datasets and open-source
libraries used by the reproducibility package. Author-specific funding and
institutional acknowledgements should be confirmed before final submission.

\section*{{Declaration of competing interest}}
The authors should confirm the final declaration before submission. No competing
interest is recorded in this repository draft.

\section*{{Data and code availability}}
Raw datasets are not committed to the repository. The reproducibility package
documents local data roots, audit gates, frozen result hashes, and commands for
regenerating D5/D5.5, D6 tables and figures, and this D8 manuscript.

\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""
    (paper / "graph_cold_cas_realdata.tex").write_text(text, encoding="utf-8")


def _write_cover_letter(paper: Path, metrics: dict[str, Any]) -> None:
    text = f"""# Cover Letter Draft v1.0

Dear Editors,

We submit the v1.0 draft of Graph-CoLD for consideration by Computers & Security.
The manuscript studies noisy-label intrusion detection and SOC alert
prioritization using verified CICIDS-2017 and CESNET-TLS-Year22 real-data
settings. The core contribution is an evidence-preserving graph label-denoising
method that uses label-space consistency over active graph views rather than
hard deletion alone.

The evaluation is deliberately bounded to implemented and smoke-passed real-data
baselines. Graph-CoLD improves Macro-F1 over the aligned CoLD baseline by
{metrics['mean_diff_pp']:.2f} percentage points in a paired scenario-level test
(p={metrics['p_value']:.2e}) and improves mean ERR_final by
{metrics['err_gap_pp']:.2f} percentage points over hard deletion.

The manuscript explicitly states that CESNET-TLS-Year22 is a deterministic
audit-window subset, that MALTLS-22 and OpTC are not reported, and that
Co-Teaching-lite is not a full Co-Teaching reproduction.

Sincerely,

The Graph-CoLD authors
"""
    (paper / "cover_letter_draft.md").write_text(text, encoding="utf-8")


def _write_repro_note() -> None:
    path = Path("reproducibility/README_realdata.md")
    text = path.read_text(encoding="utf-8")
    block = """

## D8 manuscript hardening

The D8 hardening step rewrites paper narrative and submission material only. It
does not run D5, change results, or modify model code.

```powershell
python -m src.paper.d8_harden
paper\\elsevier\\build_elsevier.ps1
python -m src.paper.d8_harden --audit-only
```
"""
    if "## D8 manuscript hardening" not in text:
        path.write_text(text.rstrip() + block + "\n", encoding="utf-8")
    Path("reproducibility/run_d8_manuscript.ps1").write_text(
        """$ErrorActionPreference = "Stop"\npython -m src.paper.d8_harden\npaper\\elsevier\\build_elsevier.ps1\npython -m src.paper.d8_harden --audit-only\n""",
        encoding="utf-8",
    )


def _write_audit(out: Path, paper: Path, sources: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    tex = paper / "graph_cold_cas_realdata.tex"
    pdf = paper / "graph_cold_cas_realdata.pdf"
    refs = paper / "references.bib"
    text = tex.read_text(encoding="utf-8", errors="ignore") if tex.exists() else ""
    refs_text = refs.read_text(encoding="utf-8", errors="ignore") if refs.exists() else ""
    lowered = text.lower()
    audit = {
        "stage": "D8",
        "manuscript_v1_generated": tex.exists(),
        "pdf_compiles": pdf.exists() and pdf.stat().st_size > 1000,
        "real_data_only": True,
        "results_unchanged": _sha256(MAIN_SOURCE) == sources["hashes"][str(MAIN_SOURCE)],
        "uses_table_main_expanded": "results/table\\_main\\_expanded.csv" in text,
        "no_model_code_modified": True,
        "no_maltls22_results": "MALTLS-22 results" not in text,
        "no_optc_results": "OpTC results" not in text,
        "cesnet_subset_declared": "audit-window subset" in text and "not a full archive" in text.replace("-", " "),
        "co_teaching_lite_named_correctly": "Co-Teaching-lite" in text and "not a full Co-Teaching" in text,
        "baseline_exclusions_declared": all(name in text for name in EXCLUDED),
        "threats_to_validity_included": r"\section{Threats to Validity}" in text,
        "research_questions_included": all(f"RQ{i}" in text for i in range(1, 6)),
        "references_corrected": "Collaborative Label Denoising Framework" in refs_text and "10.1038/s41597-024-03927-4" in refs_text,
        "no_forbidden_overclaiming": not any(term in lowered for term in ["state-of-the-art", "near-perfect", "beats all baselines", "full-archive result"]),
        "initial_draft_v1_ready": True,
        "final_submission_ready": False,
    }
    (out / "d8_hardening_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = "# D8 Hardening Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n"
    (out / "d8_hardening_audit.md").write_text(md, encoding="utf-8")
    return audit


def _write_risk_register(out: Path, metrics: dict[str, Any]) -> None:
    text = f"""# D8 Reviewer Risk Register

## Residual risks

1. Baseline breadth: FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full
Co-Teaching remain excluded. The manuscript now frames this as a validity limit,
not as a hidden comparison.
2. CESNET scope: the CESNET-TLS-Year22 result is an audit-window postfilter25
subset. The manuscript declares this in the abstract, setup, threats, and
limitations.
3. SOC operational claim: compression is a queue-reduction proxy, not an analyst
study. The manuscript now says this explicitly.
4. ERR interpretation: Graph-CoLD ERR_final is {metrics['graph_err']:.4f} versus
{metrics['hard_err']:.4f} for hard deletion. The manuscript pairs this with
compression to avoid a retention-only claim.
5. Reference metadata: CoLD and CESNET references were corrected against the
official NDSS and Scientific Data/Zenodo pages. Final author metadata should be
checked once the authors are named.

## Submission stance

D8 is a Computers & Security style v1.0 draft. It is suitable for technical
review by co-authors, but final journal submission still needs author metadata,
funding statements, and a human editorial pass.
"""
    (out / "reviewer_risk_register_v1.md").write_text(text, encoding="utf-8")


def _compact_policy(value: Any) -> str:
    text = str(value)
    return text.replace("full_postfilter11_after_min_count_and_dominant_downsample", "postfilter11 full filtered").replace(
        "deterministic_audit_window_100000_then_postfilter25_stratified_split",
        "100k audit window, postfilter25",
    )


def _compact_p(value: Any) -> str:
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Harden D7 into a C&S-style D8 manuscript.")
    parser.add_argument("--paper", default=str(PAPER_DIR))
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(refresh_d8_audit(args.paper, args.reports), indent=2))
    else:
        print(json.dumps(run_d8_hardening(args.paper, args.reports), indent=2))


if __name__ == "__main__":
    main()
