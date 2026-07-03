# D5 Readiness Guard Report

- Implemented: true
- Readiness required: true
- D5 allowed: false
- D5 scope: none
- Old hardcoded datasets removed: true
- Forbidden formal datasets: MALTLS-22, OpTC
- Formal outputs generated: false

## Readiness Inputs

- `reports/second_dataset_selection_gate.json`
- `reports/realdata_readiness_report.json`
- `reports/two_dataset_readiness_report.json`

## Blocking Reasons

- No verified second dataset is ready.

The D5 runner now fails before creating formal result tables when the readiness
gate is false.
