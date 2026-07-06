"""Revision-round evidence-benefit artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.evidence_downstream import write_downstream_benefit_csv


def generate_evidence_downstream_artifacts(
    results_csv: str | Path = "results/table_main.csv",
    out_csv: str | Path = "results/evidence_downstream_benefit.csv",
    figure_path: str | Path = "figures/fig_p0_evidence_downstream_benefit.pdf",
) -> dict[str, str]:
    """Generate traceable downstream-benefit CSV and figure from formal results."""

    frame = pd.read_csv(results_csv)
    comparison = write_downstream_benefit_csv(frame, out_csv)
    fig_path = Path(figure_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_high_noise_fnr_delta(comparison, fig_path)
    return {"csv": str(out_csv), "figure": str(fig_path), "rows": str(len(comparison))}


def _plot_high_noise_fnr_delta(comparison: pd.DataFrame, figure_path: Path) -> None:
    high = comparison[pd.to_numeric(comparison["noise_rate"], errors="coerce") >= 0.4].copy()
    if high.empty:
        high = comparison.copy()
    if high.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No paired rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)
        return
    grouped = high.groupby(["dataset", "noise_type"], dropna=False)["fnr_delta_graphcold_minus_hard"].mean().reset_index()
    labels = grouped["dataset"].astype(str) + "\n" + grouped["noise_type"].astype(str)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(grouped)), 3.2))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.bar(labels, grouped["fnr_delta_graphcold_minus_hard"], color="#4c78a8")
    ax.set_ylabel("FNR delta (Graph-CoLD - hard)")
    ax.set_xlabel("Dataset / noise")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="results/table_main.csv")
    parser.add_argument("--out-csv", default="results/evidence_downstream_benefit.csv")
    parser.add_argument("--figure", default="figures/fig_p0_evidence_downstream_benefit.pdf")
    args = parser.parse_args()
    print(generate_evidence_downstream_artifacts(args.results_csv, args.out_csv, args.figure))


if __name__ == "__main__":
    main()
