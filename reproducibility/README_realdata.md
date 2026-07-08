# Graph-CoLD Real-data Reproducibility Package

This package recreates the evaluation matrix, paper tables/figures, and
manuscript assembly from verified local datasets. Raw datasets are not
committed.

## Data roots

- CICIDS-2017: `data/cicids2017`
- CESNET-TLS-Year22: `E:\graphcold-data\tls_alternative\cesnet_tls_year22`
- UNSW-NB15: `E:\graphcold-data\unsw_nb15`
- External data root: `E:\graphcold-data`

Formal reported datasets are `CICIDS-2017 postfilter11`,
`CESNET-TLS-Year22 postfilter25`, and `UNSW-NB15` under the verified local
partition layout. MALTLS-22 and OpTC are not part of the formal evaluation
package.

## Frozen source artifacts

- `results/table_main_expanded.csv`: `b9a7f26563e27bced0c2e77b8864bcfe19521bbe1cda7424afad261e63c113a9`
- `results/table_baseline_expansion.csv`: `b9a7f26563e27bced0c2e77b8864bcfe19521bbe1cda7424afad261e63c113a9`
- `results/stat_tests_baseline_expansion.json`: `30acbbe745348a31951a6458b79d9c73b563af5f27b9dea8f738e893768ba477`
- `reports/realdata_readiness_report.json`: `0d61fb1a829ee6ea0b0ae1d9bab94d2b200e31ce008b302db4e88f1591ede733`

## Recreate results

Run readiness gates before experiments:

```powershell
python -m src.data.audit
python scripts/check_data_ready.py
```

Then run the formal matrix and baseline expansion explicitly:

```powershell
python -m src.experiments.d5 --out results --configs configs
python -m src.experiments.d5_baseline_expansion --out results --configs configs --reports reports
```

Generate paper assets:

```powershell
python -m src.paper.d6_prep
python -m src.paper.d7_assemble
paper\elsevier\build_elsevier.ps1
python -m src.paper.d7_assemble --audit-only
```

Large dataset downloads are manual or optional and are not started by these
scripts. Keep raw archives and extracted data outside Git tracking.

## Manuscript hardening

The manuscript hardening step rewrites paper narrative and submission material
only. It does not run experiments, change results, or modify model code.

```powershell
& $env:GRAPH_COLD_PYTHON -m src.paper.d8_harden
powershell -ExecutionPolicy Bypass -File paper\elsevier\build_elsevier.ps1
& $env:GRAPH_COLD_PYTHON -m src.paper.d8_harden --audit-only
```

