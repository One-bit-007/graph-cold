# P2f Tighten Report

## 1. P2e Gate Note

- Soft-not-hard probe passed: True
- Tension regenerated: True
- Max tension rate: 0.375315

## 2. Corrected Rare-recovery

- Definition: recovered samples are clean rare/tail suspicious samples whose final trained classifier predicts the clean true class.
- Retention is reported separately and does not enter the recovery numerator.
- An out-of-fold recovery column is included only as a memorization-sensitivity audit.
- Non-constant proof: hard max=0.290323; soft min=0.925714.

## 3. Powered Per-dataset Results

- Paired rows per dataset/metric: 30
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Test size target: 5000

| paired_rows | method_mean | baseline_mean | mean_delta | p_value_greater | cohens_dz | bootstrap_ci95_low | bootstrap_ci95_high | dataset | method | metric | aggregate_macro_f1_delta_vs_hard | aggregate_macro_f1_regression_p_less | no_significant_aggregate_macro_f1_regression | p_holm | meets_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.474642 | 0.452833 | 0.021809 | 0.000012 | 0.914879 | 0.013395 | 0.029821 | CESNET-TLS-Year22 | Graph-CoLD-soft | tail_macro_f1 | 0.011815 | 0.999993 | True | 0.000037 | True |
| 30 | 1.000000 | 0.063641 | 0.936359 | 0.000000 | 10.636698 | 0.904581 | 0.965375 | CESNET-TLS-Year22 | Graph-CoLD-soft | rare_recovery_rate | 0.011815 | 0.999993 | True | 0.000000 | True |
| 30 | 0.450464 | 0.452833 | -0.002369 | 0.859595 | -0.200648 | -0.006597 | 0.001888 | CESNET-TLS-Year22 | Graph-CoLD-semisup | tail_macro_f1 | -0.001021 | 0.194140 | True | 0.859595 | False |
| 30 | 0.126572 | 0.063641 | 0.062931 | 0.000132 | 0.758115 | 0.035750 | 0.094440 | CESNET-TLS-Year22 | Graph-CoLD-semisup | rare_recovery_rate | -0.001021 | 0.194140 | True | 0.000264 | True |
| 30 | 0.434171 | 0.361041 | 0.073129 | 0.000000 | 1.281637 | 0.053252 | 0.092386 | CICIDS-2017 | Graph-CoLD-soft | tail_macro_f1 | 0.032684 | 1.000000 | True | 0.000000 | True |
| 30 | 1.000000 | 0.002222 | 0.997778 | 0.000000 | 81.975809 | 0.993333 | 1.000000 | CICIDS-2017 | Graph-CoLD-soft | rare_recovery_rate | 0.032684 | 1.000000 | True | 0.000000 | True |
| 30 | 0.357481 | 0.361041 | -0.003560 | 0.715570 | -0.105217 | -0.016370 | 0.007885 | CICIDS-2017 | Graph-CoLD-semisup | tail_macro_f1 | -0.001932 | 0.289250 | True | 0.715570 | False |
| 30 | 0.011590 | 0.002222 | 0.009368 | 0.023683 | 0.378105 | 0.001852 | 0.018845 | CICIDS-2017 | Graph-CoLD-semisup | rare_recovery_rate | -0.001932 | 0.289250 | True | 0.047366 | True |
| 30 | 0.172986 | 0.132494 | 0.040492 | 0.000000 | 1.285745 | 0.029944 | 0.052338 | UNSW-NB15 | Graph-CoLD-soft | tail_macro_f1 | 0.013758 | 0.999998 | True | 0.000000 | True |
| 30 | 0.970370 | 0.041268 | 0.929102 | 0.000000 | 14.962161 | 0.905960 | 0.950519 | UNSW-NB15 | Graph-CoLD-soft | rare_recovery_rate | 0.013758 | 0.999998 | True | 0.000000 | True |
| 30 | 0.129917 | 0.132494 | -0.002577 | 0.798561 | -0.154992 | -0.008682 | 0.003181 | UNSW-NB15 | Graph-CoLD-semisup | tail_macro_f1 | -0.002276 | 0.042503 | False | 0.798561 | False |
| 30 | 0.186668 | 0.041268 | 0.145400 | 0.000000 | 2.807713 | 0.127679 | 0.162738 | UNSW-NB15 | Graph-CoLD-semisup | rare_recovery_rate | -0.002276 | 0.042503 | False | 0.000000 | False |

## 4. Pre-registered Verdict

- Global verdict: `robust`
- Dataset pass map: `{"CESNET-TLS-Year22": true, "CICIDS-2017": true, "UNSW-NB15": true}`

## 5. Claims Input

Evidence-aware soft retention significantly improves rare-class tail metrics over hard deletion under high asymmetric noise on 3 datasets; remaining dataset-level effects are reported without overclaim.

## 6. Summary Table

| reported_as | method | macro_f1 | tail_macro_f1 | tail_recall | rare_recovery_rate | rare_recovery_rate_oof_audit | rare_retained_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CESNET-TLS-Year22 | CoLD | 0.657784 | 0.411247 | 0.305858 | 0.054878 | 0.054579 | 0.000000 |
| CESNET-TLS-Year22 | Graph-CoLD-semisup | 0.678689 | 0.450464 | 0.326727 | 0.126572 | 0.126572 | 1.000000 |
| CESNET-TLS-Year22 | Graph-CoLD-soft | 0.691526 | 0.474642 | 0.344494 | 1.000000 | 1.000000 | 1.000000 |
| CESNET-TLS-Year22 | ablation_hard | 0.679710 | 0.452833 | 0.330874 | 0.063641 | 0.069164 | 0.000000 |
| CICIDS-2017 | CoLD | 0.534492 | 0.047320 | 0.043229 | 0.002778 | 0.002778 | 0.000000 |
| CICIDS-2017 | Graph-CoLD-semisup | 0.686565 | 0.357481 | 0.271875 | 0.011590 | 0.011590 | 1.000000 |
| CICIDS-2017 | Graph-CoLD-soft | 0.721181 | 0.434171 | 0.328385 | 1.000000 | 1.000000 | 1.000000 |
| CICIDS-2017 | ablation_hard | 0.688497 | 0.361041 | 0.276823 | 0.002222 | 0.007222 | 0.000000 |
| UNSW-NB15 | CoLD | 0.456497 | 0.095687 | 0.186928 | 0.037355 | 0.035545 | 0.000000 |
| UNSW-NB15 | Graph-CoLD-semisup | 0.468802 | 0.129917 | 0.194102 | 0.186668 | 0.186668 | 1.000000 |
| UNSW-NB15 | Graph-CoLD-soft | 0.484836 | 0.172986 | 0.228712 | 0.970370 | 0.970370 | 1.000000 |
| UNSW-NB15 | ablation_hard | 0.471078 | 0.132494 | 0.204949 | 0.041268 | 0.038679 | 0.000000 |

## 7. Honest Reject-risk Re-estimate

Reject risk is reduced for the evidence-retention claim, but the manuscript must still avoid presenting corrected rare-recovery as an ERR substitute.

## 8. Reproduction Commands

- `python -m pytest tests/test_soft_not_hard.py tests/test_rare_recovery_nontautological.py -q`
- `python -m src.paper.p2f_tighten --configs configs --out results --reports reports --tables tables --figures figures --train-size 10000 --test-size 5000`
