# OpTC Preparation Report

- Mode: local-events
- Events CSV present: True
- Audit passed: False
- Full download attempted: False
- Dataset hash: `70d6898a5c75a603f4010e33ba77d66d9a70d8a0855baa9c9d0dbd831c2de7e1`

## Instructions
- Review OpTC data source: https://github.com/FiveDirections/OpTC-data
- Obtain the authorized OpTC release through the official distribution channel.
- Convert eCAR/Bro/JSON provenance records into data/optc/events.csv.
- Required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score.
- Run: python scripts/download_optc.py --mode local-events --events path/to/events.csv --out data/optc
