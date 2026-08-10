# Reproducibility Export Plan (Public Repository / Zenodo)

This plan defines exactly what leaves the private working repository when the
code and artifacts are published. It exists so that publication is a checklist,
not a judgment call.

## Include (public)

| Path | Contents |
| --- | --- |
| `src/` | All model, data, experiment, and paper-asset code |
| `configs/` | Experiment configurations |
| `tests/` | Test suite |
| `results/` | Frozen result CSV/JSON files (hashes recorded in `reports/p5_pre_run_hashes.json` and the P2f reports) |
| `tables/` | Paper-ready tables (CSV + MD) |
| `figures/` | Paper figures (PDF + PNG) |
| `reports/` | Audit and phase reports (P1–P5 narrative and JSON) |
| `paper/graph_cold_cas_final.tex` | Manuscript source |
| `paper/submission/` | Cover letter, highlights, declarations, this plan |
| `reproducibility/` | Reproduction documentation and scripts |
| `scripts/` | Dataset download/preparation helpers |
| `README.md`, `requirements.txt`, `CITATION.cff` | Top-level documentation |

## Exclude (never publish)

| Path | Reason |
| --- | --- |
| `.venv/` | Local virtual environment |
| Internal company document directory (untracked) | Must never be pushed or archived |
| `work/` | Scratch/intermediate outputs |
| `data/` | Raw datasets (redistributable only from original providers) |
| External data roots (`E:\graphcold-data`, anything under `GRAPHCOLD_DATA_ROOT`) | Raw datasets |
| `paper/archive_do_not_submit/` | Superseded drafts |
| `*.aux`, `*.log`, `*.out`, `*.spl`, `__pycache__/` | Build/cache artifacts |
| `docs/prompts/` | Internal working prompts (optional; keep private) |

## Publication steps

1. Create a clean public repository containing only the "Include" set above
   (verify with a fresh `git archive` or a clean-room copy, then `grep -ri
   h3c` on the export to confirm zero company references).
2. Fill in author metadata in `CITATION.cff`, `paper/submission/*.md`, and the
   manuscript front matter.
3. Tag the release (`rev-p5` or the acceptance tag) in the public repository.
4. Create a Zenodo archive from that tag; record the DOI.
5. Replace `[REPO/DOI PLACEHOLDER]` in the manuscript Data availability section
   and in `paper/submission/data_availability.md` with the real URL/DOI.
6. Recompile the manuscript (two `pdflatex` passes) so the final PDF carries
   the real DOI.
