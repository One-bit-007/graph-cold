# P2e Tension Gate

- Completed: True
- Real data only: True
- Scale policy: `tail_preserving_real_audit_window_train_10000_test_1000`
- Gate rule: pass if any real dataset/rate has clean-rare GraphCDM tension >= 5%
- Gate threshold: 5.00%
- Gate passed: True
- Max tension rate: 0.3753
- Pooled clean-rare weighted tension rate: 0.1045
- Figure: `figures/fig_p2e_cdm_tension.pdf`
- CSV: `reports/p2e_tension_gate.csv`

## Rows

| dataset | reported_as | sample_policy | train_rows | num_classes | tail_labels | tail_label_names | noise_type | tail_flip_rate | effective_flip_rate_train | theta | clean_rare_count | flipped_count | tension_rate_clean_rare_cdm_gt_theta | noisy_flag_rate_cdm_gt_theta | clean_rare_cdm_mean | noisy_flipped_cdm_mean | clean_rare_cdm_q90 | noisy_flipped_cdm_q90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cicids2017 | CICIDS-2017 | p2d_real_audit_window_train_10000_test_1000 | 10000 | 11 | [1, 5, 6, 9, 10] | ["1", "5", "6", "9", "10"] | tail_asymmetric | 0.400000 | 0.010400 | 0.500000 | 158 | 104 | 0.075949 | 0.028846 | 0.204296 | 0.076502 | 0.233808 | 0.104824 |
| cicids2017 | CICIDS-2017 | p2d_real_audit_window_train_10000_test_1000 | 10000 | 11 | [1, 5, 6, 9, 10] | ["1", "5", "6", "9", "10"] | tail_asymmetric | 0.600000 | 0.015700 | 0.500000 | 105 | 157 | 0.171429 | 0.006369 | 0.316643 | 0.040900 | 0.914308 | 0.066337 |
| cesnet_tls_year22 | CESNET-TLS-Year22 | p2d_real_audit_window_train_10000_test_1000 | 10000 | 25 | [1, 3, 4, 5, 6, 7, 8, 10, 16, 19, 21, 23, 24] | ["1", "3", "4", "5", "6", "7", "8", "10", "16", "19", "21", "23", "24"] | tail_asymmetric | 0.400000 | 0.124800 | 0.500000 | 1872 | 1248 | 0.014957 | 0.045673 | 0.130988 | 0.095099 | 0.159019 | 0.115359 |
| cesnet_tls_year22 | CESNET-TLS-Year22 | p2d_real_audit_window_train_10000_test_1000 | 10000 | 25 | [1, 3, 4, 5, 6, 7, 8, 10, 16, 19, 21, 23, 24] | ["1", "3", "4", "5", "6", "7", "8", "10", "16", "19", "21", "23", "24"] | tail_asymmetric | 0.600000 | 0.187200 | 0.500000 | 1248 | 1872 | 0.060096 | 0.006410 | 0.214145 | 0.041811 | 0.234467 | 0.066966 |
| unsw_nb15 | UNSW-NB15 | p2d_real_audit_window_train_10000_test_1000 | 10000 | 9 | [1, 2, 7, 8] | ["1", "2", "7", "8"] | tail_asymmetric | 0.400000 | 0.039600 | 0.500000 | 595 | 396 | 0.294118 | 0.277778 | 0.365031 | 0.291770 | 0.937089 | 0.805653 |
| unsw_nb15 | UNSW-NB15 | p2d_real_audit_window_train_10000_test_1000 | 10000 | 9 | [1, 2, 7, 8] | ["1", "2", "7", "8"] | tail_asymmetric | 0.600000 | 0.059400 | 0.500000 | 397 | 594 | 0.375315 | 0.181818 | 0.440029 | 0.206186 | 0.958632 | 0.804650 |
