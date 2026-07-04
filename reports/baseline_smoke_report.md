# D5.5 Baseline Smoke Report

- Seed: 42
- Passed baselines: Co-Teaching-lite, Confident-Learning, Noisy-Supervised

## Excluded
- FINE: excluded: faithful FINE requires a validated embedding outlier protocol; D5.5 keeps it out instead of reporting approximate or fake rows
- FINE-style: excluded: faithful FINE requires a validated embedding outlier protocol; D5.5 keeps it out instead of reporting approximate or fake rows
- MCRe: excluded: no independently implemented and smoke-passed real-data implementation in this repository
- MORSE: excluded: no independently implemented and smoke-passed real-data implementation in this repository
- Decoupling: excluded: no independently implemented and smoke-passed real-data implementation in this repository
- Flash: excluded: provenance case-study method; no formal two-dataset label-noise implementation
- Argus: excluded: provenance case-study method; no formal two-dataset label-noise implementation

## Rows
- cicids2017 clean Noisy-Supervised: Macro-F1=0.990268, finite=True, uses_noisy=True
- cicids2017 clean Confident-Learning: Macro-F1=0.930350, finite=True, uses_noisy=True
- cicids2017 clean Co-Teaching-lite: Macro-F1=0.710943, finite=True, uses_noisy=True
- cicids2017 symmetric Noisy-Supervised: Macro-F1=0.615333, finite=True, uses_noisy=True
- cicids2017 symmetric Confident-Learning: Macro-F1=0.558390, finite=True, uses_noisy=True
- cicids2017 symmetric Co-Teaching-lite: Macro-F1=0.639065, finite=True, uses_noisy=True
- cesnet_tls_year22 clean Noisy-Supervised: Macro-F1=0.994894, finite=True, uses_noisy=True
- cesnet_tls_year22 clean Confident-Learning: Macro-F1=0.974409, finite=True, uses_noisy=True
- cesnet_tls_year22 clean Co-Teaching-lite: Macro-F1=0.880795, finite=True, uses_noisy=True
- cesnet_tls_year22 symmetric Noisy-Supervised: Macro-F1=0.884349, finite=True, uses_noisy=True
- cesnet_tls_year22 symmetric Confident-Learning: Macro-F1=0.889479, finite=True, uses_noisy=True
- cesnet_tls_year22 symmetric Co-Teaching-lite: Macro-F1=0.889563, finite=True, uses_noisy=True
