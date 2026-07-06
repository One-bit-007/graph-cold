# Revision P0 Acceptance Report

## Implementation summary
- `ablation_hard` now shares the Graph-CoLD graph/representation/CDM/evidence context and differs only by hard evidence weighting (`rho=0`).
- Added verified baseline adapters for MCRe, MORSE, full Co-Teaching, FINE, plus Decoupling/Confident-Learning/Noisy-Supervised in the expanded matrix.
- Added downstream-benefit analysis for tail recall, high-noise FNR, and soft-retained/hard-deleted counterfactual checks.

## Key numeric checks
### Before
- Graph-CoLD: Macro-F1=0.991338, ERR=1.000000, FNR=0.000631
- CoLD: Macro-F1=0.971912, ERR=0.888744, FNR=0.002531
- ablation_hard: Macro-F1=0.971912, ERR=0.888744, FNR=0.002531
### After
- Graph-CoLD: Macro-F1=0.991338, ERR=1.000000, FNR=0.000631, Tail recall=0.990000
- CoLD: Macro-F1=0.971912, ERR=0.888744, FNR=0.002531, Tail recall=0.940825
- ablation_hard: Macro-F1=0.971809, ERR=0.888744, FNR=0.002618, Tail recall=0.940717

## Baseline coverage
- Before: Co-Teaching-lite, CoLD, Confident-Learning, Graph-CoLD, Noisy-Supervised, ablation_hard
- After: Co-Teaching, CoLD, Confident-Learning, Decoupling, FINE, Graph-CoLD, MCRe, MORSE, Noisy-Supervised, ablation_hard

## Downstream benefit
- Paired downstream rows: 102
- Mean tail-recall delta (Graph-CoLD - hard): 0.046375
- Mean high-noise FNR delta (Graph-CoLD - hard): -0.002270
- Counterfactual helper: `src.analysis.evidence_downstream.counterfactual_soft_retention`.

## Jargon purge proof
- Current paper-facing generated artifacts grep clean: true
- Scope: manuscript, submission statements, paper sections, tables, hardening reports, and reproducibility README.
- Result: no matches.

## Tests
- `pytest -q`: 193 passed, 2 skipped, 13 warnings.

## Reproduction commands
- `& 'C:/Users/g14370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m src.experiments.formal_matrix --out results --configs configs`
- `& 'C:/Users/g14370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m src.experiments.baseline_matrix --out results --configs configs --reports reports`
- `& 'C:/Users/g14370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m src.paper.revision_p0 --results-csv results/table_main.csv --out-csv results/evidence_downstream_benefit.csv --figure figures/fig_p0_evidence_downstream_benefit.pdf`
- `& 'C:/Users/g14370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m pytest tests/test_err.py tests/test_baseline_registry.py tests/test_baseline_coteaching.py tests/test_baseline_mcre_morse.py tests/test_evidence_downstream.py -q`
