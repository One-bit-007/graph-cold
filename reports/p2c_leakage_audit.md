# P2c CICIDS Leakage Audit

## Verdict
- Leakage found in frozen CICIDS results: True
- Current D5 runner fixed: True
- Split-crossing edges are zero: True
- Primary cause: `flip_mask_oracle_and_clean_label_graph_evidence_in_pre_P2c_runner`

## Split Boundary
- Total crossing edges: 0
- Test nodes in graph: 0

## Duplicate / Near-Duplicate Audit
- Train rows: 630399
- Exact duplicate train rows: 152201 (0.241436)
- Exact cross-split duplicate test rows: 38366 (0.243439)
- Graph edges connecting exact duplicate rows: 41432
- Graph edges connecting near-duplicate rows (threshold 1e-4): 57172

## De-Oracle CICIDS Audit
- Rows: 153
- Audit scope: real_stratified_train_3000_dedup_2818_test_8000, real_stratified_train_3000_dedup_2824_test_8000, real_stratified_train_3000_dedup_2848_test_8000
- Mean exact duplicate train rows removed: 170.0
- Mean near-duplicate graph edges removed: 210.7
- Graph-CoLD minus CoLD Macro-F1 (all noisy scenarios): 0.062370
- Graph-CoLD minus CoLD Macro-F1 (high noise): 0.035975
- Flat 0.99 curve survives: False

## Neighborhood Sanity
flip_mask_oracle_dominates; duplicate_edges_are_present_and_removed_in_p2c_audit_ablation
