"""P5-B8: regenerate the Figure 1 method-overview asset.

Removes the stale in-figure draft caption ("Fig1. Graph-CoLD submission-scope
workflow" — the LaTeX caption supersedes it) and updates the dataset/view
labels to the final three-dataset scope (adds UNSW-NB15, renames the UNSW view
to "behavioral"). Style replicates the original d9_submission_lock figure.
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out_pdf = Path("figures/fig1_method_overview.pdf")
    out_png = Path("figures/fig1_method_overview.png")
    labels = [
        ("Audited real data", "CICIDS postfilter11\nCESNET postfilter25\nUNSW-NB15 partition"),
        ("Multi-view graphs", "host / IP / temporal /\nbehavioral (view masks)"),
        ("Representation", "contrastive + temporal\nreconstruction"),
        ("Graph-CDM", "label-space consistency\nevidence score"),
        ("SOC output", "weighted training\npriority proxy"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 2.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0.35, 1.75)
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
        ax.text(x + 0.825, 1.28, title, ha="center", va="center", fontsize=9.0, weight="bold", color="#172033")
        ax.text(x + 0.825, 0.86, subtitle, ha="center", va="center", fontsize=7.0, color="#39475e", linespacing=1.25)
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
    fig.tight_layout(pad=0.6)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
