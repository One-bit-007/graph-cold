# CICIDS Class Policy Report

- Selected policy: `postfilter11`
- Selected classes: 11
- Dataset hash: `2585508ac445a94a3eb2244aa64778678928d201555396b4b9afc1ed6a2f1ab4`
- Paper statement: CICIDS-2017 post-filtered 11-class setting

## Policy A: raw15
- Classes: 15
- Has <1000 classes: True
- Suitable: False

## Policy B: postfilter11
- Classes: 11
- Removed classes: {"Heartbleed": 11, "Infiltration": 36, "Web Attack � Sql Injection": 21, "Web Attack � XSS": 652}
- Consistent with smoke: True

## Policy C: refined9
- Enabled: False
- Reason: No strict refined9_mapping is defined in docs or configs; refined9 is therefore not used.

## Decision
No authoritative refined9 mapping exists in docs/configs; current deterministic smoke policy is postfilter11 and every retained class has >= min_class_count samples.
