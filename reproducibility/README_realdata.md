# Graph-CoLD Real-data Reproducibility Package

This package recreates the evaluation matrix, paper tables/figures, and
manuscript assembly from verified local datasets. Raw datasets are not
committed.

## Data roots

- CICIDS-2017: `data/cicids2017`
- CESNET-TLS-Year22: `E:\graphcold-data\tls_alternative\cesnet_tls_year22`
- External data root: `E:\graphcold-data`

Formal reported datasets are `CICIDS-2017 postfilter11` and
`CESNET-TLS-Year22 postfilter25`. MALTLS-22 and OpTC are not part of the formal
evaluation package.

## Frozen source artifacts

- `results/table_main_expanded.csv`: `d044d4847edc1706fcdf904142085895baca70130287beb9cd2ad8c17d20dfb8`
- `results/table_baseline_expansion.csv`: `fd3c0c02d18bf10d30302a1e62cd946036801d5491d0b480e45f600e9bb11c79`
- `results/stat_tests_baseline_expansion.json`: `a8c806847a8849cb094e5aba555a09e3bdfb6ffd873a937fabe7a922bde73abb`
- `reports/realdata_readiness_report.json`: `3c284a3a4f09b023bac4e20400e589f31c61ecf717caf1886e737e93f0b98e0e`

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

