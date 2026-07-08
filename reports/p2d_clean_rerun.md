# P2d Clean Rerun Report

## 1. P2c Gate
- Passed: True
- CICIDS exact-dedup removed rows: 107054
- Split-crossing edges: 0
- Evidence/CDM source: observed_noisy_labels_plus_unsupervised_feature_anomaly

## 2. G1 Fresh Canonical Outputs
- Scale policy: `real_data_deterministic_audit_window_train_1500_test_4000`
- Expanded source hash: `b9a7f26563e27bced0c2e77b8864bcfe19521bbe1cda7424afad261e63c113a9`
- Canonical headline: `tables/table_p2_canonical_headline.csv`
- Stale tables superseded: results/table_main.csv, results/table_main_expanded.csv, results/table_ablation.csv, tables/table_p2_canonical_headline.csv, tables/table_2_main_performance.csv, tables/table_3_high_noise_summary.csv

## 3. G2 Core Contribution Verdict
- Verdict: `benefit_vanishes`
- Pooled ERR delta vs hard: 0.000000
- Pooled tail-recall delta vs hard: 0.002747
- Pooled FNR delta vs hard: -0.000721

| dataset | paired_rows | graphcold_macro_f1_mean | hard_macro_f1_mean | delta_macro_f1_vs_hard | p_macro_f1_vs_hard | graphcold_err_final_mean | hard_err_final_mean | delta_err_final_vs_hard | p_err_final_vs_hard | graphcold_err_tail_mean | hard_err_tail_mean | delta_err_tail_vs_hard | p_err_tail_vs_hard | graphcold_tail_recall_mean | hard_tail_recall_mean | delta_tail_recall_vs_hard | p_tail_recall_vs_hard | graphcold_fnr_mean | hard_fnr_mean | delta_fnr_vs_hard | p_fnr_vs_hard | err_not_trivially_one |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CESNET-TLS-Year22 | 48.000000 | 0.827400 | 0.826872 | 0.000527 | 0.679725 | 0.942988 | 0.942988 | 0.000000 | 1.000000 | 0.946490 | 0.946490 | 0.000000 | 1.000000 | 0.802519 | 0.802100 | 0.000419 | 0.840040 | 0.088740 | 0.088386 | 0.000353 | 0.533200 | 1.000000 |
| CICIDS-2017 | 48.000000 | 0.521370 | 0.519822 | 0.001548 | 0.466126 | 0.823866 | 0.823866 | 0.000000 | 1.000000 | 0.767625 | 0.767625 | 0.000000 | 1.000000 | 0.371081 | 0.365718 | 0.005363 | 0.169438 | 0.240830 | 0.243870 | -0.003039 | 0.065774 | 1.000000 |
| UNSW-NB15 | 48.000000 | 0.405580 | 0.408504 | -0.002924 | 0.046987 | 0.561877 | 0.561877 | 0.000000 | 1.000000 | 0.532335 | 0.532335 | 0.000000 | 1.000000 | 0.350040 | 0.347582 | 0.002458 | 0.456602 | 0.135274 | 0.134750 | 0.000524 | 0.722912 | 1.000000 |
| pooled | 48.000000 | 0.584783 | 0.585066 | -0.000283 | 0.764637 | 0.776243 | 0.776243 | 0.000000 | 1.000000 | 0.748817 | 0.748817 | 0.000000 | 1.000000 | 0.507880 | 0.505133 | 0.002747 | 0.149211 | 0.154948 | 0.155669 | -0.000721 | 0.372437 | 1.000000 |

## 4. G3 Per-Dataset Headline
| dataset | delta_macro_f1_vs_cold | delta_err_final_vs_hard |
| --- | --- | --- |
| CESNET-TLS-Year22 | 0.004551 | 0.000000 |
| CICIDS-2017 | 0.057219 | 0.000000 |
| UNSW-NB15 | 0.003256 | 0.000000 |
| pooled | 0.021675 | 0.000000 |

## 5. G4 Downstream Artifacts
- Graph informativeness: `tables/table_p2d_graph_informativeness.csv`
- Figures: `figures/fig_p2d_macro_f1_vs_noise_rate.pdf`, `figures/fig_p2d_err_retention.pdf`, `figures/fig_p2d_informativeness_margin.pdf`

## 6. G5 Guard Test
- `tests/test_no_oracle_leakage.py` covers flip-mask and clean-label canaries.

## 7. G6 Final Claims Input
| dataset | macro_f1_delta_vs_cold_mean | err_delta_vs_hard_mean | tail_recall_delta_vs_hard_mean | fnr_delta_vs_hard_mean | framing |
| --- | --- | --- | --- | --- | --- |
| CICIDS-2017 | 0.057219 | 0.000000 | 0.005363 | -0.003039 | Evidence-preservation gain vanishes here; rescope the claim. |
| CESNET-TLS-Year22 | 0.004551 | 0.000000 | 0.000419 | 0.000353 | Detection is near ceiling; emphasize retention/operational metrics rather than Macro-F1. |
| UNSW-NB15 | 0.003256 | 0.000000 | 0.002458 | 0.000524 | Weak process/temporal-only views are a boundary case; do not claim universal gains. |
| pooled | 0.021675 | 0.000000 | 0.002747 | -0.000721 | Evidence-preservation gain vanishes here; rescope the claim. |

Framing: gains should be stated per dataset; graph signal helps when informative, is neutral near ceiling, and may be weak or negative when available views are poor.

## 8. Reject-Risk
- Estimate: high
- P2d matrix uses deterministic real-data audit windows for tractability; do not call it raw full-CICIDS exhaustive.
- UNSW no longer shows a negative mean margin in this clean run.

## 9. Reproduction Commands
- `python -m src.paper.p2d_clean_rerun --configs configs --out results --reports reports --tables tables --figures figures`
- `python -m pytest tests/test_no_oracle_leakage.py tests/test_number_consistency.py -q`
