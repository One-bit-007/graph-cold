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

## cesnet_tls_year22

- Root: `E:\graphcold-data\tls_alternative\cesnet_tls_year22`
- Available files: 365
- Rows / columns: 100000 / 45
- Classes: 178
- Dataset hash: `490bb4402e111c4ef752272e17d8faee44ce9ef25a0063c7c84ff4d3cb7f9084`
- Ready for smoke: True
- Ready for D5: True
- View support: {"host": "not_expected", "ip": "available", "process": "not_expected", "temporal": "available", "threat_intel": "not_expected"}
- Blocking reasons:
  - none

## unsw_nb15

- Root: `data\unsw_nb15`
- Available files: 0
- Rows / columns: 0 / 0
- Classes: 0
- Dataset hash: `None`
- Ready for smoke: False
- Ready for D5: False
- View support: {"host": "not_expected", "ip": "missing", "process": "not_expected", "temporal": "derived_limited", "threat_intel": "not_expected"}
- Blocking reasons:
  - dataset root does not exist: data/unsw_nb15
  - label column missing: contract label candidate
  - required column groups missing: label, ip_or_flow, numeric_features
  - row count 0 below min_samples 10000

## ustc_tfc2016

- Root: `data\ustc_tfc2016`
- Available files: 0
- Rows / columns: 0 / 0
- Classes: 0
- Dataset hash: `None`
- Ready for smoke: False
- Ready for D5: False
- View support: {"host": "not_expected", "ip": "missing", "process": "not_expected", "temporal": "not_expected", "threat_intel": "not_expected"}
- Blocking reasons:
  - dataset root does not exist: data/ustc_tfc2016
  - label column missing: contract label candidate
  - required column groups missing: label, flow_or_payload
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
