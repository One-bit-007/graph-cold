# Baseline Readiness Report

- Formal D5 methods: Graph-CoLD, CoLD, ablation_hard
- Expanded D5 methods: Graph-CoLD, CoLD, ablation_hard, Co-Teaching-lite, Confident-Learning, Noisy-Supervised

- Graph-CoLD: included=True; reason=reused from verified real D5 run
- CoLD: included=True; reason=reused from verified real D5 run
- ablation_hard: included=True; reason=reused from verified real D5 run
- Co-Teaching-lite: included=True; reason=implemented and smoke-passed on CICIDS-2017 and CESNET-TLS-Year22
- Confident-Learning: included=True; reason=implemented and smoke-passed on CICIDS-2017 and CESNET-TLS-Year22
- Noisy-Supervised: included=True; reason=implemented and smoke-passed on CICIDS-2017 and CESNET-TLS-Year22
- FINE: included=False; reason=excluded: faithful FINE requires a validated embedding outlier protocol; D5.5 keeps it out instead of reporting approximate or fake rows
- FINE-style: included=False; reason=excluded: faithful FINE requires a validated embedding outlier protocol; D5.5 keeps it out instead of reporting approximate or fake rows
- MCRe: included=False; reason=excluded: no independently implemented and smoke-passed real-data implementation in this repository
- MORSE: included=False; reason=excluded: no independently implemented and smoke-passed real-data implementation in this repository
- Decoupling: included=False; reason=excluded: no independently implemented and smoke-passed real-data implementation in this repository
- Flash: included=False; reason=excluded: provenance case-study method; no formal two-dataset label-noise implementation
- Argus: included=False; reason=excluded: provenance case-study method; no formal two-dataset label-noise implementation
- cleanlab: included=False; reason=legacy audit key; official cleanlab is represented by the Confident-Learning method row
- Co-Teaching: included=False; reason=legacy full deep baseline name; D5.5 includes Co-Teaching-lite instead
- Co-Teaching+: included=False; reason=not independently implemented and smoke-passed on real data
