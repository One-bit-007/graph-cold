# CESNET Class Policy Report

- Dataset: `cesnet_tls_year22`
- Reported as: CESNET-TLS-Year22
- Selected policy: `postfilter`
- Ready for smoke: False

## Raw
- Classes: 0
- Suitable: False

## Postfilter
- Classes: 0
- Removed classes: {}
- Min class count: 1000

## Blocking Reasons
- dataset root does not exist: data/tls_alternative/cesnet_tls_year22
- label column missing: contract label candidate
- required column groups missing: label, tls_or_flow_features, timestamp
- row count 0 below min_samples 10000
