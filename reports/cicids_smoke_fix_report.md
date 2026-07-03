# CICIDS Smoke Fix Report

## Status

CICIDS-2017 smoke repair completed for seed 42 on the audited real dataset.

- Dataset hash: `2585508ac445a94a3eb2244aa64778678928d201555396b4b9afc1ed6a2f1ab4`
- Data source: `C:\Users\g14370\graph-cold\data\cicids2017`
- D5 allowed: `false`
- Submission ready: `false`

## Before vs After

| Scenario | CoLD Macro-F1 | ablation_hard Macro-F1 | Graph-CoLD Macro-F1 | ERR Graph-CoLD | ERR hard |
|---|---:|---:|---:|---:|---:|
| Initial clean | 0.99245 | n/a | 0.99243 | n/a | n/a |
| Initial symmetric_20 | 0.99178 | n/a | 0.64574 | n/a | n/a |
| Fixed clean | 0.992454 | 0.992454 | 0.991597 | 1.000000 | 1.000000 |
| Fixed symmetric_20 | 0.966864 | 0.966864 | 0.991959 | 1.000000 | 0.843156 |

## Fix Summary

- CICIDS active views are now `host`, `ip`, and `temporal`; unsupported `process` and `threat_intel` views are inactive.
- Hard deletion is implemented by fitting on retained samples instead of passing zero sample weights into `class_weight="balanced"`.
- Tree smoke soft weighting now uses explicit finite class-balance sample weights and applies `tau_ret=0.1` to suppress below-threshold noisy samples.
- Clean smoke CDM no longer marks boundary samples for deletion when `flip_mask` is empty.
- Evidence min-max normalization returns ones for constant evidence, avoiding accidental all-zero evidence.

## Gate Results

- Graph-CoLD symmetric_20 Macro-F1 >= CoLD - 0.03: pass (`0.991959` vs `0.966864`)
- ablation_hard ~= CoLD: pass (`diff=0.0`)
- ERR(Graph-CoLD) > ERR(ablation_hard): pass (`1.0 > 0.843156`)
- retained clean informative Graph-CoLD >= hard: pass (`1.0 >= 0.979997`)
- active views correct: pass
- weighted CE normalized: pass
- no test-label noise: pass
- no perfect/zero metric anomaly: pass

## Reproduce

```powershell
python -m src.experiments.smoke_realdata --dataset cicids2017 --configs configs --out reports
python -m src.diagnostics.cicids_smoke_diagnosis --dataset cicids2017 --noise symmetric_20 --seed 42 --configs configs --out reports
python -m src.experiments.smoke_ablation --dataset cicids2017 --noise symmetric_20 --seed 42 --configs configs --out reports
python -m pytest -q tests/test_active_view_mask.py tests/test_weighted_loss_normalization.py tests/test_noisy_label_protocol.py tests/test_smoke_ablation_gate.py tests/test_cicids_smoke_diagnosis.py tests/test_err.py tests/test_smoke_realdata_gate.py
```

## Blocking Items

D5 remains blocked because MALTLS-22 is not source-verified locally and OpTC case-study data is unavailable. Full D5/D6/D7 artifacts were intentionally not generated.
