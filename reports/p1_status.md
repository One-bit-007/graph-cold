# P1 Status Report

## P0 Gate Report

| Gate | Status | Evidence |
|---|---:|---|
| A1 | PASS | `{"Graph-CoLD": {"macro_f1": 0.9913381705213394, "err_final": 1.0, "fnr": 0.0006313241781812748, "tail_recall": 0.9900003419504423}, "ablation_hard": {"macro_f1": 0.9718091852129992, "err_final": 0.8887439473541461, "fnr": 0.0026182574767778318, "tail_recall": 0.9407166362034296}, "CoLD": {"macro_f1": 0.8422581625873913, "err_final": 0.7495220605670916, "fnr": 0.007931488621025936, "tail_recall": 0.7529579293549166}}` |
| A2 | PASS | `{"method_counts": {"Co-Teaching": 102, "CoLD": 102, "Confident-Learning": 102, "Decoupling": 102, "FINE": 102, "Graph-CoLD": 102, "MCRe": 102, "MORSE": 102, "Noisy-Supervised": 102, "ablation_hard": 102}, "graphcold_vs_comparisons": {"Graph-CoLD_vs_Co-Teaching": {"mean_diff": 0.26848801850783116, "p_value_holm": 8.133662813764403e-14}, "Graph-CoLD_vs_FINE": {"mean_diff": 0.3722718904140924, "p_value_holm": 7.22245776007129e-11}, "Graph-CoLD_vs_MCRe": {"mean_diff": 0.3104762631231712, "p_value_holm": 7.22245776007129e-11}, "Graph-CoLD_vs_MORSE": {"mean_diff": 0.31949565833774196, "p_value_holm": 3.3427362418434105e-11}}}` |
| A3 | PASS | `{"rows": 102, "tail_recall_delta_mean": 0.04637517350492328, "tail_recall_delta_p_greater": 8.478693450225591e-11, "high_noise_fnr_delta_mean": -0.002269754732938494, "high_noise_fnr_delta_p_less": 2.7888295470459795e-06}` |
| A4 | PASS | `"no matches"` |

## P1 Results Delta

### G1
- Status: blocked
- Blocker: ['dataset root does not exist: E:\\graphcold-data\\unsw_nb15', 'label column missing: contract label candidate', 'required column groups missing: label, ip_or_flow, numeric_features', 'row count 0 below min_samples 10000']
- Alternative: No already-available verified third dataset is present locally; USTC-TFC2016 remains candidate-only.

### G2
- Status: completed
- Table: `results/table_prioritization.csv`
- Curve: `results/prioritization_curve.csv`
- Figure: `figures/fig_p1_queue_load_curve.pdf`
- Graph-CoLD mean Top-K precision=0.999167, compression@90=0.662931.

### G3
- Status: completed
- Table: `tables/table_p1_statistical_rigor.csv`
- Scenario-level mean diff=0.14748943631910588, CI=[0.07608944627795731, 0.22298107710452278], p=0.0005327159332021416, effective_n=18.

### G4
- Status: completed
- Figure: `figures/fig_p1_graph_noise_beta_sweep.pdf`
- beta0 matches symmetric: True
- concentration increases: True

## Acceptance-Risk Self-Assessment

- Estimated reject risk after P1: approximately 30-40%.
- Remaining weaknesses:
  - UNSW-NB15 is verified in contract but absent locally, so the third-dataset claim remains blocked.
  - Direct prioritization metrics are now measured, but Graph-CoLD is essentially tied with CoLD and ablation_hard on Top-K precision in the audit-window ranking evaluation.
  - Prioritization evaluation uses deterministic real-data audit windows to avoid local memory exhaustion on full CICIDS loading.

## Reproduction Commands

- `python -m src.experiments.p1_refresh_cold --out results --configs configs --reports reports`
- `python -m src.analysis.prioritization --out results --configs configs --reports reports --figure figures/fig_p1_queue_load_curve.pdf`
- `python -m src.data.unsw_policy --data-root E:/graphcold-data --out reports`
- `python -c "from src.analysis.graph_noise_validation import beta_sweep_report; beta_sweep_report(figure_path='figures/fig_p1_graph_noise_beta_sweep.pdf')"`
- `python -m src.paper.p1_status --reports reports --results results --tables tables`
