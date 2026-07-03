# Real-Data Readiness Report

- D5 allowed: False
- Submission ready: False

## Datasets

### cicids2017
- audit_passed: True
- smoke_passed: True
- mini_matrix_passed: True
- ready_for_d5_component: True
- available: True

### maltls22
- audit_passed: False
- ready_for_d5_component: False
- source_verified: False
- available: False
- Blocking reasons:
  - dataset root does not exist: data/maltls22
  - label column missing: label
  - required column groups missing: flow_or_tls_features, timestamp
  - row count 0 below min_samples 10000
  - dataset source is not verified; do not report this dataset

### optc
- audit_passed: False
- available: False
- ready_for_case_study: False
- Blocking reasons:
  - dataset root does not exist: data/optc
  - missing expected files: events.csv
  - label column missing: label
  - missing required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score
  - row count 0 below min_samples 1000

## Blocking Reasons
- MALTLS-22 source unverified or replacement dataset not selected
- OpTC events.csv unavailable
