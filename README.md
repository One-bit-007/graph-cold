# Graph-CoLD

**Evidence-Preserving Graph Denoising for Noisy-Label Intrusion Detection:
A Leakage-Audited Study of When Graph Structure Helps**

Companion repository for the manuscript submitted to *Computers & Security*
(Elsevier). See `CITATION.cff` for citation metadata.

## What this study is (and is not)

This repository contains a leakage-audited re-evaluation of collaborative
label denoising for intrusion detection, built around three bounded findings:

1. **Leakage audit.** Naive graph denoising results are easy to inflate —
   e.g., 24% exact duplicate records in CICIDS-2017 and clean-label signal
   paths. Removing these collapses an apparent +28-point advantage to a
   defensible +6.7 points. All reported numbers use the audited P2f
   clean-deduplication protocol.
2. **Mechanism-level finding.** Within a shared pipeline, evidence-preserving
   soft weighting is statistically indistinguishable from hard deletion on
   aggregate Macro-F1, yet significantly improves rare/tail attack-class
   Macro-F1 under high asymmetric noise at no aggregate cost.
3. **Audited external-baseline ordering.** Under the identical protocol,
   Confident Learning outperforms the proposed weighting on all three datasets,
   while full Co-Teaching collapses on rare classes — diagnostic precision,
   not the weighting scheme, is the deciding factor.

This is deliberately **not** a state-of-the-art accuracy claim. The primary
artifacts are the audit, the graph-informativeness diagnostic, and the
reproducibility package.

## Datasets

- **CICIDS-2017** (postfilter11) — Canadian Institute for Cybersecurity, UNB
- **CESNET-TLS-Year22** (postfilter25) — Zenodo DOI: 10.5281/zenodo.10608607
- **UNSW-NB15** — University of New South Wales, Canberra

Raw datasets are not distributed here; see `reproducibility/README_realdata.md`
for the expected layout and the `GRAPHCOLD_DATA_ROOT` environment variable.
MALTLS-22 and OpTC are not part of the evaluation package (no verified
license/provenance path).

## Repo map

```
src/          model, data, graph, experiment, and paper-asset code
configs/      experiment configurations
tests/        test suite
results/      frozen result CSV/JSON (audited protocol)
tables/       paper-ready tables (CSV + MD)
figures/      paper figures (PDF + PNG)
reports/      audit trail: phase reports P1–P5
paper/        manuscript source + submission statements
reproducibility/  reproduction documentation and scripts
scripts/      dataset download/preparation helpers
```

## Reproduce

Start with `reproducibility/README.md` and
`reproducibility/README_realdata.md`. The authoritative submission numbers come
from the P2f clean-deduplication protocol (`reports/p2f_tighten.md`); the
external-anchor results (Confident Learning, full Co-Teaching) are reproduced
by `src/paper/p5_external_baselines.py`. Earlier D5/D9 pipeline scripts are
retained for audit history only and must not be used to regenerate submission
numbers.

## License

TODO — see `CITATION.cff` (`license` field) before public release.
