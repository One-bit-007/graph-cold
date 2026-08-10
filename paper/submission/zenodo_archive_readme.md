# Graph-CoLD — Zenodo Archive README

This archive accompanies the manuscript *"Graph-Diagnosed Collaborative Label
Denoising for Network Intrusion Detection: A Leakage Audit and Mechanism-Level
Re-evaluation"* (submitted to Computers & Security, Elsevier).

## What is in this archive

- Full source code (`src/`), configurations (`configs/`), and tests (`tests/`).
- Frozen experimental results (`results/`), paper-ready tables (`tables/`),
  and figures (`figures/`).
- Audit trail: phase reports P1–P5 under `reports/`, including the leakage
  audit (P2c–P2f) and the external-baseline evaluation (P5) that compares
  Graph-CoLD against Confident Learning and full Co-Teaching under the
  identical audited protocol.
- Manuscript source (`paper/graph_cold_cas_final.tex`) and submission
  statements (`paper/submission/`).
- Reproduction documentation (`reproducibility/`).

## What is NOT in this archive

Raw datasets. All three datasets are publicly available from their original
providers (see `paper/submission/data_availability.md`):

- CIC-IDS-2017 — Canadian Institute for Cybersecurity, UNB.
- CESNET-TLS-Year22 — Zenodo DOI: 10.5281/zenodo.10608607.
- UNSW-NB15 — University of New South Wales, Canberra.

Place the datasets at the layout described in
`reproducibility/README_realdata.md`, optionally setting `GRAPHCOLD_DATA_ROOT`.

## Reproducing the results

Follow `reproducibility/README_realdata.md`. The authoritative submission
numbers come from the P2f clean-deduplication protocol (`reports/p2f_tighten.md`);
the external-anchor numbers are reproduced by `src/paper/p5_external_baselines.py`.

## Citation

See `CITATION.cff`. If you use this archive, please cite both the manuscript
and this artifact.
