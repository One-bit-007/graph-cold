# Real-Data Readiness Report

- D5 completed: True
- D6 allowed: True
- D7 allowed: False
- Submission ready: False
- D5 scope: CICIDS-2017, CESNET-TLS-Year22

## Datasets

### cicids2017
- available: True
- audit_passed: True
- ready_for_smoke: True
- ready_for_d5: True
- ready_for_d5_component: True
- Blocking reasons: none

### maltls22
- available: False
- audit_passed: False
- ready_for_smoke: False
- ready_for_d5: False
- ready_for_d5_component: False
- source_verified: False
- Blocking reasons:
  - dataset root does not exist: data/maltls22
  - label column missing: label
  - required column groups missing: flow_or_tls_features, timestamp
  - row count 0 below min_samples 10000
  - dataset source is not verified; do not report this dataset

### cesnet_tls_year22
- available: True
- audit_passed: True
- ready_for_smoke: True
- ready_for_d5: True
- ready_for_d5_component: True
- Blocking reasons: none

### unsw_nb15
- available: False
- audit_passed: False
- ready_for_smoke: False
- ready_for_d5: False
- ready_for_d5_component: False
- Blocking reasons:
  - dataset root does not exist: data/unsw_nb15
  - label column missing: contract label candidate
  - required column groups missing: label, ip_or_flow, numeric_features
  - row count 0 below min_samples 10000

### ustc_tfc2016
- available: False
- audit_passed: False
- ready_for_smoke: False
- ready_for_d5: False
- ready_for_d5_component: False
- Blocking reasons:
  - dataset root does not exist: data/ustc_tfc2016
  - label column missing: contract label candidate
  - required column groups missing: label, flow_or_payload
  - row count 0 below min_samples 10000
  - dataset source is not verified; do not report this dataset

### optc
- available: False
- audit_passed: False
- formal_experiment: False
- future_case_study_only: True
- Blocking reasons:
  - dataset root does not exist: data/optc
  - missing expected files: events.csv
  - label column missing: label
  - missing required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score
  - row count 0 below min_samples 1000

## Next Actions
- MALTLS-22 remains unevaluated unless source verification changes.
- Resolve unsw_nb15: dataset root does not exist: data/unsw_nb15; label column missing: contract label candidate; required column groups missing: label, ip_or_flow, numeric_features; row count 0 below min_samples 10000
- Resolve ustc_tfc2016: dataset root does not exist: data/ustc_tfc2016; label column missing: contract label candidate; required column groups missing: label, flow_or_payload; row count 0 below min_samples 10000; dataset source is not verified; do not report this dataset
- OpTC unavailable; keep it out of formal experiments or provide real events.csv.
