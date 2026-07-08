# P2e Salvage Report

## 1. A0 Gate

- Passed: True
- Executable guard: `python -m pytest tests/test_no_oracle_leakage.py -q`

## 2. A1 Tension Gate

- Gate passed: True
- Max tension rate: 0.3753
- Pooled clean-rare weighted tension rate: 0.1045
- Figure: `figures/fig_p2e_cdm_tension.pdf`

## Decision

- Decision: `part_b_salvaged`
- Reason: Part B ran on pre-registered tail-asymmetric real-data comparisons.

## 3. Part B Salvage Evaluation

- Pre-registration: `reports\p2e_preregistration.md`
- Verdict: `salvaged`
- Success: True
- Partial signal: True
- Results: `results/p2e_tail_salvage.csv`
- Tail breakdown: `tables/table_p2e_tail_breakdown.csv`
- Success tests: `tables/table_p2e_success_tests.csv`

### Part B Summary

| reported_as | method | macro_f1 | tail_macro_f1 | tail_recall | rare_recovery_rate | rare_retained_rate | err_final_degree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CESNET-TLS-Year22 | CoLD | 0.657688 | 0.411205 | 0.306287 | 0.000000 | 0.000000 | 0.799517 |
| CESNET-TLS-Year22 | Graph-CoLD | 0.691787 | 0.477037 | 0.342836 | 1.000000 | 1.000000 | 0.843557 |
| CESNET-TLS-Year22 | Graph-CoLD-semisup | 0.680462 | 0.455186 | 0.330044 | 0.134758 | 1.000000 | 0.843557 |
| CESNET-TLS-Year22 | Graph-CoLD-soft | 0.691787 | 0.477037 | 0.342836 | 1.000000 | 1.000000 | 0.843557 |
| CESNET-TLS-Year22 | ablation_hard | 0.673396 | 0.441244 | 0.317982 | 0.000000 | 0.000000 | 0.901175 |
| CICIDS-2017 | CoLD | 0.530187 | 0.028131 | 0.024155 | 0.000000 | 0.000000 | 0.491857 |
| CICIDS-2017 | Graph-CoLD | 0.715000 | 0.412844 | 0.333333 | 1.000000 | 1.000000 | 0.861331 |
| CICIDS-2017 | Graph-CoLD-semisup | 0.694050 | 0.364107 | 0.289855 | 0.015432 | 1.000000 | 0.861331 |
| CICIDS-2017 | Graph-CoLD-soft | 0.715000 | 0.412844 | 0.333333 | 1.000000 | 1.000000 | 0.861331 |
| CICIDS-2017 | ablation_hard | 0.689112 | 0.353761 | 0.285024 | 0.000000 | 0.000000 | 0.906186 |
| UNSW-NB15 | CoLD | 0.464069 | 0.106197 | 0.219136 | 0.000000 | 0.000000 | 0.578074 |
| UNSW-NB15 | Graph-CoLD | 0.486448 | 0.173924 | 0.257716 | 0.972235 | 1.000000 | 0.774330 |
| UNSW-NB15 | Graph-CoLD-semisup | 0.470076 | 0.126861 | 0.225309 | 0.179158 | 1.000000 | 0.774330 |
| UNSW-NB15 | Graph-CoLD-soft | 0.486448 | 0.173924 | 0.257716 | 0.972235 | 1.000000 | 0.774330 |
| UNSW-NB15 | ablation_hard | 0.478915 | 0.136018 | 0.240741 | 0.000000 | 0.000000 | 0.728813 |

### Pre-registered Tests

| paired_rows | method_mean | baseline_mean | mean_delta | p_value_greater | cohens_dz | dataset | method | metric | no_significant_aggregate_macro_f1_regression | aggregate_macro_f1_regression_p_less | aggregate_macro_f1_delta_vs_hard | meets_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.477037 | 0.441244 | 0.035793 | 0.000644 | 1.613339 | CESNET-TLS-Year22 | Graph-CoLD-soft | tail_macro_f1 | True | 0.999222 | 0.018392 | True |
| 9 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 99.000000 | CESNET-TLS-Year22 | Graph-CoLD-soft | rare_recovery_rate | True | 0.999222 | 0.018392 | True |
| 9 | 0.455186 | 0.441244 | 0.013942 | 0.014224 | 0.889346 | CESNET-TLS-Year22 | Graph-CoLD-semisup | tail_macro_f1 | True | 0.982216 | 0.007067 | True |
| 9 | 0.134758 | 0.000000 | 0.134758 | 0.019764 | 0.818866 | CESNET-TLS-Year22 | Graph-CoLD-semisup | rare_recovery_rate | True | 0.982216 | 0.007067 | True |
| 9 | 0.412844 | 0.353761 | 0.059083 | 0.073562 | 0.535056 | CICIDS-2017 | Graph-CoLD-soft | tail_macro_f1 | True | 0.929758 | 0.025887 | False |
| 9 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 99.000000 | CICIDS-2017 | Graph-CoLD-soft | rare_recovery_rate | True | 0.929758 | 0.025887 | True |
| 9 | 0.364107 | 0.353761 | 0.010346 | 0.284438 | 0.198025 | CICIDS-2017 | Graph-CoLD-semisup | tail_macro_f1 | True | 0.718521 | 0.004937 | False |
| 9 | 0.015432 | 0.000000 | 0.015432 | 0.089300 | 0.491473 | CICIDS-2017 | Graph-CoLD-semisup | rare_recovery_rate | True | 0.718521 | 0.004937 | False |
| 9 | 0.173924 | 0.136018 | 0.037906 | 0.060985 | 0.576510 | UNSW-NB15 | Graph-CoLD-soft | tail_macro_f1 | True | 0.817491 | 0.007533 | False |
| 9 | 0.972235 | 0.000000 | 0.972235 | 0.000000 | 48.971185 | UNSW-NB15 | Graph-CoLD-soft | rare_recovery_rate | True | 0.817491 | 0.007533 | True |
| 9 | 0.126861 | 0.136018 | -0.009158 | 0.734601 | -0.218360 | UNSW-NB15 | Graph-CoLD-semisup | tail_macro_f1 | True | 0.096795 | -0.008839 | False |
| 9 | 0.179158 | 0.000000 | 0.179158 | 0.000042 | 2.430050 | UNSW-NB15 | Graph-CoLD-semisup | rare_recovery_rate | True | 0.096795 | -0.008839 | True |
| 27 | 0.354602 | 0.310341 | 0.044261 | 0.002065 | 0.605212 | pooled | Graph-CoLD-soft | tail_macro_f1 | True | 0.996149 | 0.017271 | False |
| 27 | 0.990745 | 0.000000 | 0.990745 | 0.000000 | 57.278904 | pooled | Graph-CoLD-soft | rare_recovery_rate | True | 0.996149 | 0.017271 | False |
| 27 | 0.315385 | 0.310341 | 0.005043 | 0.256669 | 0.127542 | pooled | Graph-CoLD-semisup | tail_macro_f1 | True | 0.611610 | 0.001055 | False |
| 27 | 0.109783 | 0.000000 | 0.109783 | 0.000046 | 0.888347 | pooled | Graph-CoLD-semisup | rare_recovery_rate | True | 0.611610 | 0.001055 | False |

### Honest Verdict

The pre-registered P2e salvage criterion is met: evidence-preserving soft/semi-supervised retention improves a tail evidence metric against hard deletion without a significant aggregate Macro-F1 regression.

## Reproduction Commands

- `python -m pytest tests/test_no_oracle_leakage.py -q`
- `python -m src.paper.p2e_salvage --configs configs --out results --reports reports --tables tables --figures figures --train-size 10000 --test-size 1000`
