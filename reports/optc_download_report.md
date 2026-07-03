# OpTC Preparation Report

- Mode: local-events
- Events CSV present: True
- Audit passed: False
- Full download attempted: False
- Dataset hash: `ba1ad8b4d2c65070ea9dee7b246c4bba960a4b29ee321f028ddfacaffb9f8a76`

## Instructions
- Review OpTC data source: https://github.com/FiveDirections/OpTC-data
- Obtain the authorized OpTC release through the official distribution channel.
- Convert eCAR/Bro/JSON provenance records into data/optc/events.csv.
- Required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score.
- Run: python scripts/download_optc.py --mode local-events --events path/to/events.csv --out data/optc
