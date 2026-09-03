# Graph-CoLD Reproducibility Package (Revision / Frozen-Protocol Edition)

This package documents how every number in the revised manuscript
(`paper/springer/graph_cold_sn_blind.tex`) is produced from tracked, frozen
experimental artifacts, and how to re-run the full grid from scratch.

## Authoritative result sources (frozen protocol)

All headline results come from the **frozen final protocol**: proxy-masked
features (tier-1 CICIDS / tier-2 CESNET+UNSW), strongest-available
group-disjoint splits, deterministic audit windows (train 10,000 / test 5,000),
asymmetric tail noise {0.4, 0.6, 0.8}, seeds 0–9, pre-registered Holm families
A (9) / B (18) / C (18), TOST δ=0.01 with per-class guard, and mixed-effects
robustness checks. Frozen constants live in the runners themselves and in
`configs/*.yaml`; the noise mapping is `reproducibility/noise_map.json`.

| Manuscript object | Frozen artifact | Producing runner |
|---|---|---|
| Table 2 (main grid) + Tables 4/6 (families A/C) | `results/p5_factor_ablation_full.csv`, `tables/table_p5_family_a.csv`, `tables/table_p5_family_c.csv`, `tables/table_p5_perclass_guard.csv` | `src/paper/p5_factor_ablation.py`, `src/paper/p5_perclass_guard.py` |
| Table 7 (family B, G2 gate) | `results/p4_cl_graph_combo_full_family_b_complete.csv`, `reports/p4_cl_graph_combo_full_family_b_complete.json` | `src/paper/p4_cl_graph_combo.py --mode full` |
| Table 8 (leakage decomposition) | `tables/table_p6_leakage_ablation_fixedtail_all.csv` | `src/paper/p6_leakage_ablation.py` |
| 100k scale validation | `results/p6_scale_validation_cicids.csv`, `reports/p6_scale_validation_cicids.json` | `src/paper/p6_scale_validation.py` |
| Table 5 (real labelling errors) | `results/p7_natural_noise.csv`, `reports/p7_natural_noise.json`, `reports/p7_engelen_diff_summary.json` | `src/paper/p7_engelen_labels.py` + P7 natural-noise runner |
| Table 3 (legacy-window FINE/Co-Teaching, non-comparable) | `results/p7_baseline_expansion.csv` | `src/paper/p7_baseline_expansion.py` |
| Proxy audit / split taxonomy (S0–S3) | `reports/p3_proxy_audit_*.json`, `reports/p3_proxy_audit_screening.md` | `src/paper/p3_proxy_audit.py` |
| CICIDS entity-isolation infeasibility | `reports/p6_cicids_entity_isolation.md`, `reports/p6_cicids_raw_reconcile.json` | loaders (`cicids2017_raw` config), WP-3 analysis |
| Mixed-effects confirmations | `reports/p5_family_a_mixed_effects.json` + family-B JSON | P0-5 stage of the runners above |

## Manuscript table and number generation (no retraining)

```powershell
./.venv/Scripts/python.exe -m src.paper.w5_result_tables   # 7 LaTeX fragments
./.venv/Scripts/python.exe -m src.paper.w5_number_audit    # 19 inline-number checks
```

`paper/springer/tables_w5/traceability_manifest.json` maps each fragment to its
source artifacts. Fragments carry an AUTO-GENERATED banner; a claim gate fails
if the banner is removed.

## Guard and claim tests

```powershell
./.venv/Scripts/python.exe -m pytest -q    # full suite; 275 passed expected
```

Key gates: `test_no_oracle_leakage` (clean labels/noise masks can never enter
features, graphs, diagnostics, weights); `test_proxy_field_regression`
(label-construction fields and proxies stay out); `test_cicids_entity_isolation`
(entity columns are split-only metadata; honest fallback on giant components);
`test_p4_cl_graph_combo` / `test_p5_factor_ablation` / `test_p6_leakage_ablation`
(protocol constants and statistics machinery frozen); `test_w5_claim_gates`
(withdrawn claims stay withdrawn, required disclosures stay present, number
audit re-run as a subcheck). Legacy D6–D11 manuscript gates are archived under
`tests/legacy/` (collection-excluded).

## Full re-run from scratch (order)

1. Place datasets per `configs/datasets.yaml` (raw CICIDS under
   `E:/graphcold-data/cicids2017_raw/`; corrected-label rules under
   `E:/graphcold-data/cicids2017_corrected/CNS2022_Code/`).
2. `python -m src.paper.p3_proxy_audit` — audit screening.
3. `python -m src.paper.p5_factor_ablation --tier full` — families A/C (360 rows).
4. `python -m src.paper.p4_cl_graph_combo --mode full` — family B (450 rows).
5. `python -m src.paper.p6_leakage_ablation`; `python -m src.paper.p6_scale_validation`.
6. P7 runners — real-noise experiment and legacy-window baselines.
7. `python -m src.paper.w5_result_tables && python -m src.paper.w5_number_audit`.

Runners write resumable partial CSVs; interrupting and re-running is safe.

## Protocol history and amendments

- `reproducibility/protocol_history.md` — legacy protocol lineage (D5.x/P2d/P2f).
  Legacy numbers are not comparable to the frozen protocol and appear in the
  manuscript only where explicitly labelled (Table 3) or inside the leakage
  decomposition.
- Manuscript Appendix A discloses the two protocol amendments (PA-1 selective
  family-B testing, remediated; PA-2 claim ahead of evidence tier, remediated).
- `run_all.sh` and the `run_d5_*`–`run_d9_*` wrappers target the pre-audit
  pipeline and are retained for audit history only. Do **not** regenerate
  submission numbers with them.

## Manuscript

The revision manuscript is `paper/springer/graph_cold_sn_blind.tex`
(double-anonymous; compile with two `pdflatex` passes from `paper/springer/`),
with author metadata and declarations in `paper/springer/title_page_sn.tex`.
The point-by-point response is `paper/springer/response_letter_cyse.md`;
Editorial Manager paste fields are under `paper/submission/revision_2026-09/`.
Initial-submission artifacts are archived under
`paper/submission/initial_submission_2026-08-21/` and
`paper/archive_do_not_submit/`; they describe superseded claims and must not
be submitted.

## Traceability note

All numeric claims in the manuscript are generated from `results/`, `tables/`,
and the audit reports under `reports/` by `src/paper/w5_result_tables.py`, and
verified by `src/paper/w5_number_audit.py` plus `tests/test_w5_claim_gates.py`.
Raw datasets are not committed; configure dataset roots via
`configs/datasets.yaml` (the `E:/graphcold-data/...` paths in this document are
the authors' local example layout, not requirements) and see
`README_realdata.md` / the `GRAPHCOLD_DATA_ROOT` environment variable.
