# Real-Data Readiness Report

- D5 allowed: False
- D6/D7 allowed: False
- Submission ready: False

## Datasets

### cicids2017
- Available: True
- Audit passed: True
- Ready for smoke: True
- Ready for D5/case: True
- Blocking reasons:
  - none

### maltls22
- Available: False
- Audit passed: False
- Ready for smoke: False
- Ready for D5/case: False
- Blocking reasons:
  - dataset root does not exist: data/maltls22
  - label column missing: label
  - required column groups missing: flow_or_tls_features, timestamp
  - row count 0 below min_samples 10000
  - dataset source is not verified; do not report this dataset

### cesnet_tls_year22
- Available: False
- Audit passed: False
- Ready for smoke: False
- Ready for D5/case: False
- Blocking reasons:
  - no readable CSV files found
  - label column missing: contract label candidate
  - required column groups missing: label, tls_or_flow_features, timestamp
  - row count 0 below min_samples 10000

### optc
- Available: False
- Audit passed: False
- Ready for smoke: False
- Ready for D5/case: False
- Blocking reasons:
  - dataset root does not exist: data/optc
  - missing expected files: events.csv
  - label column missing: label
  - missing required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score
  - row count 0 below min_samples 1000

## Next Actions
- MALTLS-22 remains unevaluated unless source verification changes.
- Resolve cesnet_tls_year22: no readable CSV files found; label column missing: contract label candidate; required column groups missing: label, tls_or_flow_features, timestamp; row count 0 below min_samples 10000
- OpTC unavailable; keep it out of formal experiments or provide real events.csv.
- Do not run D5 until CICIDS and CESNET-TLS-Year22 components pass.
