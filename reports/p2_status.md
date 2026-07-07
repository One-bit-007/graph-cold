# P2 Status Report

## P1 Gate Report

| Gate | Status | Evidence |
|---|---:|---|
| A1 | PASS | `{"Graph-CoLD": {"macro_f1": 0.8545440670330817, "err_final": 1.0}, "ablation_hard": {"macro_f1": 0.8412705685037982, "err_final": 0.8809047394709622}, "CoLD": {"macro_f1": 0.7591699312276443, "err_final": 0.7488344227479083}}` |
| A2 | PASS | `{"passed": true, "rows": 96}` |
| A3 | PASS | `{"test": "scenario_level_paired_t_test_greater", "metric": "macro_f1", "method_a": "Graph-CoLD", "method_b": "CoLD", "pairing_keys": ["dataset", "noise_type", "noise_rate", "graph_beta"], "aggregation": "mean_over_seeds_per_scenario", "effective_n": 51, "n_pairs": 51, "method_a_mean": 0.8546474750918766, "method_a_std": 0.1954653233124737, "method_b_mean": 0.7623415738285781, "method_b_std": 0.1738276244118039, "mean_diff": 0.09230590126329834, "mean_diff_ci95": [0.05479050860211299, 0.13135732241128378], "effect_size_cohen_dz": 0.6462789945002151, "t_stat": 4.615355184093925, "p_value": 1.3821341559761935e-05, "significant_p_lt_0_05": true, "naive_pooled_test_used": false}` |
| A4 | PASS | `{"ratio": 0.3, "betas": [0.0, 0.3, 0.6, 1.0], "num_classes": 4, "n": 600, "beta0_matches_symmetric": true, "beta0_transition_l1": 0.0, "transition_concentration": {"0.0": 0.4426273090066194, "0.3": 0.5662629399585921, "0.6": 0.7676301958910654, "1.0": 1.0}, "concentration_increases": true, "figure": "figures/fig_p1_graph_noise_beta_sweep.pdf"}` |
| grep | PASS | `"no matches"` |

## P2 Results Delta

### G1
- Status: completed
- headline_csv: `tables/table_p2_canonical_headline.csv`

### G2
- Status: completed
- table: `tables/table_p2_baseline_sanity.csv`
- margin_table: `tables/table_p2_baseline_margin_after.csv`
- Clean-label sanity passed: True

### G3
- Status: completed
- report: `reports/unsw_ingest.md`
- Layout: partition
- Active views: temporal, process
- Blocking reasons: []

### G4
- Status: completed
- table: `tables/table_p2_prioritization_reframe.csv`
- claim: `rescoped_to_evidence_retention_not_raw_topk_priority`

### G5
- Status: completed
- table: `tables/table_p2_contribution_decomposition.csv`
- figure: `figures/fig_p2_contribution_decomposition.pdf`

## Honest Post-P2 Risk

- Estimated reject risk: approximately 25-35%
- Residual weaknesses:
  - Direct Top-K prioritization remains tied; manuscript claims should emphasize evidence-aware retention/compression.

## Reproduction Commands

- `python -m src.data.unsw_policy --data-root E:/graphcold-data --out reports`
- `python -m src.experiments.d5 --out results --configs configs`
- `python -m src.experiments.d5_baseline_expansion --out results --configs configs --reports reports`
- `python -m src.paper.d6_prep`
- `python -m src.analysis.prioritization --out results --configs configs --reports reports --figure figures/fig_p1_queue_load_curve.pdf`
- `python -m src.paper.p2_status --results results --reports reports --tables tables --figures figures`
