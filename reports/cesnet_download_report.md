# CESNET-TLS-Year22 Download Report

- Mode: auto
- Download attempted: True
- Download success: False
- Large download confirmed: True
- Source: https://zenodo.org/records/10608607
- DataZoo source: https://cesnet.github.io/cesnet-datazoo/
- Output path: `data\tls_alternative\cesnet_tls_year22`
- Dataset hash: `None`
- Rows/classes: 0 / 0
- Error: Insufficient free disk space for CESNET-TLS-Year22. Need at least 35859993025 bytes including extraction margin; free space is 2396061696 bytes.

## Files
- CESNET-TLS-Year22.zip: 30491259088 bytes
- servicemap.csv: 24817 bytes

## Manual Instructions
- Download CESNET-TLS-Year22 from https://zenodo.org/records/10608607 or export it through https://cesnet.github.io/cesnet-datazoo/.
- Place CSV/Parquet/DataZoo-exported files under data\tls_alternative\cesnet_tls_year22.
- If using a local archive, run: python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode local-archive --archive path/to/archive --out data/tls_alternative/cesnet_tls_year22
- Then run: python -m src.data.audit --dataset cesnet_tls_year22

## Blocking Reasons
- dataset root does not exist: data\tls_alternative\cesnet_tls_year22
- label column missing: contract label candidate
- required column groups missing: label, tls_or_flow_features, timestamp
- row count 0 below min_samples 10000
