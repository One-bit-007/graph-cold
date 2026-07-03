# CESNET-TLS-Year22 Download Report

- Mode: auto
- Download attempted: True
- Download success: False
- Source: https://zenodo.org/records/10608607
- Data root: `E:\graphcold-data`
- Download cache: `E:\graphcold-data\_downloads\cesnet_tls_year22`
- Archive path: `E:\graphcold-data\_downloads\cesnet_tls_year22\CESNET-TLS-Year22.zip`
- Partial archive path: `E:\graphcold-data\_downloads\cesnet_tls_year22\CESNET-TLS-Year22.zip.part`
- Partial archive bytes: 31330024
- Actual data path: `E:\graphcold-data\tls_alternative\cesnet_tls_year22`
- Error: Zenodo download was reachable but throughput stayed near 110 KB/s with an estimated 70+ hours remaining; download stopped intentionally and partial file retained for resume.

## Zenodo Files
- CESNET-TLS-Year22.zip: 30491259088 bytes
- servicemap.csv: 24817 bytes

## Blocking Reasons
- no readable CSV files found
- label column missing: contract label candidate
- required column groups missing: label, tls_or_flow_features, timestamp
- row count 0 below min_samples 10000

## Manual Instructions
- Resume the official Zenodo download on a faster network using the same command; curl will continue the .part file.
- Or manually place CESNET-TLS-Year22.zip at E:\graphcold-data\_downloads\cesnet_tls_year22\CESNET-TLS-Year22.zip and run local-archive mode.
- Then rerun audit, smoke, and cesnet_mini_matrix with --data-root E:\graphcold-data.
