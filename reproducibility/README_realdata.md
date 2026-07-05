# Graph-CoLD Real-data Reproducibility Package

This package recreates the D5/D5.5 result matrix, D6 paper tables/figures, and
D7 manuscript assembly from verified local datasets. Raw datasets are not
committed.

## Data roots

- CICIDS-2017: `data/cicids2017`
- CESNET-TLS-Year22: `E:\graphcold-data\tls_alternative\cesnet_tls_year22`
- External data root: `E:\graphcold-data`

Formal reported datasets are `CICIDS-2017 postfilter11` and
`CESNET-TLS-Year22 postfilter25`. MALTLS-22 and OpTC are not part of the formal
evaluation package.

## Frozen source artifacts

- `results/table_main_expanded.csv`: `c7d998d6c918ecfbcb9cc56bd494dcec73b3fa6826b2046fb53e2ca2109519cd`
- `results/table_baseline_expansion.csv`: `b74a3552b9a11b87ee847df2fa5490197fcb4c4fbe59973c7ec3593945b9d158`
- `results/stat_tests_baseline_expansion.json`: `6aff31cb1d29cbae5a63bb586eb73fdf63b0fe38391c5819f8cdf9ac2fcfd7e4`
- `reports/realdata_readiness_report.json`: `3c284a3a4f09b023bac4e20400e589f31c61ecf717caf1886e737e93f0b98e0e`

## Recreate results

Run readiness gates before experiments:

```powershell
python -m src.data.audit
python scripts/check_data_ready.py
```

Then run D5 and D5.5 explicitly:

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

## D8 manuscript hardening

The D8 hardening step rewrites paper narrative and submission material only. It
does not run D5, change results, or modify model code.

```powershell
python -m src.paper.d8_harden
paper\elsevier\build_elsevier.ps1
python -m src.paper.d8_harden --audit-only
```

