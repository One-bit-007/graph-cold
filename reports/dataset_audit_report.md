# Dataset Audit Report

## cicids2017

- Root: `data\cicids2017`
- Available files: 8
- Rows / columns: 2830743 / 79
- Classes: 15
- Dataset hash: `2585508ac445a94a3eb2244aa64778678928d201555396b4b9afc1ed6a2f1ab4`
- Ready for smoke: True
- Ready for D5: True
- View support: {"host": "missing", "ip": "missing", "process": "not_expected", "temporal": "derived_limited", "threat_intel": "not_expected"}
- Blocking reasons:
  - none

## maltls22

- Root: `data\maltls22`
- Available files: 0
- Rows / columns: 0 / 0
- Classes: 0
- Dataset hash: `None`
- Ready for smoke: False
- Ready for D5: False
- View support: {"host": "missing", "ip": "missing", "process": "not_expected", "temporal": "derived_limited", "threat_intel": "not_expected"}
- Blocking reasons:
  - dataset root does not exist: data/maltls22
  - label column missing: label
  - required column groups missing: flow_or_tls_features, timestamp
  - row count 0 below min_samples 10000
  - dataset source is not verified; do not report this dataset

## optc

- Root: `data\optc`
- Available files: 0
- Rows / columns: 0 / 0
- Classes: 0
- Dataset hash: `None`
- Ready for smoke: False
- Ready for D5: False
- View support: {"host": "missing", "ip": "missing", "process": "missing", "temporal": "derived_limited", "threat_intel": "missing"}
- Blocking reasons:
  - dataset root does not exist: data/optc
  - missing expected files: events.csv
  - label column missing: label
  - missing required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score
  - row count 0 below min_samples 1000
