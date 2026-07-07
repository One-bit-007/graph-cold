# P2c Leakage and Per-Dataset Report

## 1. P2b Gate
- Regenerated robustness table: True
- Outcome: `B_protocol_explained`
- Result numbers changed: False
- Number consistency green: True
- Frozen hash intact: True

## 2. G1 Leakage Verdict
- Leakage found in frozen CICIDS rows: True
- Current D5 runner fixed: True
- De-leaked CICIDS table: `tables/table_p2c_cicids_deleaked_per_rate.csv`
- Old CICIDS headline must not be used for claims until a full P2c-safe D5 rerun refreshes the formal matrix.

## 3. G2 Per-Dataset Pattern
- Table: `tables/table_p2c_per_dataset_vs_cold.csv`
| dataset | mean_delta_macro_f1_vs_cold | mean_delta_err_vs_cold | mean_delta_fnr_vs_cold | p_macro_f1_vs_cold |
| --- | --- | --- | --- | --- |
| CESNET-TLS-Year22 | 0.002725 | 0.119598 | 0.000002 | 0.000000 |
| CICIDS-2017 | 0.286653 | 0.351890 | -0.014179 | 0.000000 |
| UNSW-NB15 | -0.012460 | 0.237686 | -0.000008 | 0.000437 |

## 4. G3 Graph Informativeness
- Table: `tables/table_p2c_graph_informativeness.csv`
- Figure: `figures/fig_p2c_informativeness_margin.pdf`
- Pearson r (n=3): 0.993055
| dataset | active_views | informativeness_score | post_p2c_graphcold_minus_cold_macro_f1 | interpretation |
| --- | --- | --- | --- | --- |
| CICIDS-2017 | host|ip|temporal | 0.570440 | 0.066702 | positive_when_views_have_structural_signal |
| CESNET-TLS-Year22 | ip|temporal | 0.373653 | 0.002725 | ceiling_case_graph_signal_cannot_create_large_macro_f1_lift |
| UNSW-NB15 | temporal|process | 0.288213 | -0.012460 | boundary_case_low_view_support_no_positive_margin |

## 5. G4 Claims-Input Block
| dataset | graphcold_minus_cold_macro_f1 | source_for_delta | informativeness_score | claim_framing |
| --- | --- | --- | --- | --- |
| CICIDS-2017 | 0.066702 | P2c de-oracle CICIDS audit sample | 0.570440 | Do not use old +28 pp CICIDS headline; rerun formal matrix with P2c-safe D5 before submission. |
| CESNET-TLS-Year22 | 0.002725 | formal D5.5 frozen matrix | 0.373653 | ceiling_case_graph_signal_cannot_create_large_macro_f1_lift |
| UNSW-NB15 | -0.012460 | formal D5.5 frozen matrix | 0.288213 | boundary_case_low_view_support_no_positive_margin |

Framing sentence 1: Graph-CoLD should be claimed as helpful when graph views carry measurable structural signal, not as a uniform pooled improvement.

Framing sentence 2: CESNET is a ceiling case and UNSW is a weak-view boundary case; CICIDS requires a full P2c-safe formal rerun before any headline gain is manuscript-safe.

## 6. Reject-Risk Re-Estimate
- medium-high until formal D5 is rerun with the P2c-safe runner
- Residual weakness: CICIDS correction is an audit ablation on real stratified data, not a full replacement for the frozen D5 all-dataset matrix.

## 7. Reproduction Commands
- `python -m src.paper.p2c_leakage_perdataset --configs configs --reports reports --tables tables --figures figures`
- `python -m pytest tests/test_p2c_leakage_perdataset.py tests/test_p2b_baseline_fidelity.py tests/test_number_consistency.py -q`
