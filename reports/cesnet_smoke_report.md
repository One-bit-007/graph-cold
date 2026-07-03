# Real-Data Smoke Report

- Dataset: cesnet_tls_year22
- Status: blocked
- Passed: False
- Dataset hash: `None`
- Seed: 42
- Blocking reasons:
  - dataset root does not exist: data/tls_alternative/cesnet_tls_year22
  - label column missing: contract label candidate
  - required column groups missing: label, tls_or_flow_features, timestamp
  - row count 0 below min_samples 10000
