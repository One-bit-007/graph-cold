# Graph-CoLD Real-data Reproducibility Package

This package recreates the D5/D5.5 result matrix, D6 paper tables/figures, D7
assembly, D8 manuscript hardening, and D9 submission-lock package from verified
local datasets. Raw datasets are not committed.

## Formal Scope

- Datasets: CICIDS-2017 postfilter11; CESNET-TLS-Year22 postfilter25.
- Methods: Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised,
  Confident-Learning, and Co-Teaching-lite.
- Excluded from formal results: MALTLS-22, OpTC, UNSW-NB15, USTC-TFC2016,
  FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full Co-Teaching.
- MALTLS-22 and OpTC are not part of the formal evaluation package.

## Data Roots

- CICIDS-2017: `data/cicids2017`
- CESNET-TLS-Year22: `E:\graphcold-data\tls_alternative\cesnet_tls_year22`
- External data root: `E:\graphcold-data`

## Frozen Source Artifacts

- `results/table_main_expanded.csv`: `c7d998d6c918ecfbcb9cc56bd494dcec73b3fa6826b2046fb53e2ca2109519cd`
- `results/table_baseline_expansion.csv`: `b74a3552b9a11b87ee847df2fa5490197fcb4c4fbe59973c7ec3593945b9d158`
- `results/stat_tests_baseline_expansion.json`: `6aff31cb1d29cbae5a63bb586eb73fdf63b0fe38391c5819f8cdf9ac2fcfd7e4`
- `reports/realdata_readiness_report.json`: `3c284a3a4f09b023bac4e20400e589f31c61ecf717caf1886e737e93f0b98e0e`

## Entry Points

Run readiness gates:

```powershell
python -m src.data.audit
python scripts/check_data_ready.py
```

Recreate D5/D5.5 real-data results:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\run_d5_realdata.ps1
```

Regenerate D6 paper assets:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\run_d6_paper_assets.ps1
```

Rebuild the D7 manuscript:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\run_d7_build.ps1
```

Regenerate the D8 hardened manuscript:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\run_d8_manuscript.ps1
```

Regenerate the D9 candidate submission package:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\run_d9_submission_package.ps1
```

Large dataset downloads are manual or optional and are not started by the paper
asset scripts. Keep raw archives and extracted data outside Git tracking.
