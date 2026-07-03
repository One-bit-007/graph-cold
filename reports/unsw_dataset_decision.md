# UNSW-NB15 Dataset Decision

- Dataset: `unsw_nb15`
- Reported as: UNSW-NB15
- Source verified: True
- Selected policy: `postfilter`
- Ready for smoke: False
- Ready for D5 component: False
- Active views: ip | temporal

## Raw Class Policy
- Classes: 0
- Suitable: False

## Postfilter Policy
- Classes: 0
- Removed classes: {}
- Downsample rule: No dominant-class downsampling needed.

## Blocking Reasons
- dataset root does not exist: E:\graphcold-data\unsw_nb15
- label column missing: contract label candidate
- required column groups missing: label, ip_or_flow, numeric_features
- row count 0 below min_samples 10000
