# Dataset Audit Report

## cicids2017

- Root: `data\cicids2017`
- Available files: 0
- Rows / columns: 0 / 0
- Classes: 0
- Dataset hash: `None`
- Ready for smoke: False
- Ready for D5: False
- View support: {"host": "missing", "ip": "missing", "process": "not_expected", "temporal": "derived_limited", "threat_intel": "not_expected"}
- Blocking reasons:
  - dataset root does not exist: data/cicids2017
  - missing expected files: Monday-WorkingHours.pcap_ISCX.csv, Tuesday-WorkingHours.pcap_ISCX.csv, Wednesday-workingHours.pcap_ISCX.csv, Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv, Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv, Friday-WorkingHours-Morning.pcap_ISCX.csv, Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv, Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  - label column missing: Label
  - IP/port/protocol columns are all missing
  - row count 0 below min_samples 10000

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
