# D5.5 Baseline Expansion Report

- Completed: True
- Status: strong
- Original D5 rows unchanged: True
- Expanded rows: 612
- Added baseline rows: 306
- Included baselines: Co-Teaching-lite, Confident-Learning, Noisy-Supervised

## Key Metrics
- Co-Teaching-lite: Macro-F1 mean=0.718569, ERR mean=0.823117, compression mean=0.991068
- CoLD: Macro-F1 mean=0.973171, ERR mean=0.895288, compression mean=0.990853
- Confident-Learning: Macro-F1 mean=0.767907, ERR mean=0.824528, compression mean=0.993595
- Graph-CoLD: Macro-F1 mean=0.991457, ERR mean=1.000000, compression mean=0.989613
- Noisy-Supervised: Macro-F1 mean=0.683640, ERR mean=1.000000, compression mean=0.997891
- ablation_hard: Macro-F1 mean=0.973171, ERR mean=0.895288, compression mean=0.990853

## Statistical Comparisons
- Graph-CoLD_vs_Co-Teaching-lite: n=102, diff=0.27288734987758057, dz=2.208871733377313, p=4.0477999562302313e-41
- Graph-CoLD_vs_CoLD: n=102, diff=0.018285494916563578, dz=0.4577987163803332, p=5.5956741051131204e-06
- Graph-CoLD_vs_Confident-Learning: n=102, diff=0.22354953239876246, dz=1.3418402035755759, p=9.240692796934754e-25
- Graph-CoLD_vs_Noisy-Supervised: n=102, diff=0.30781666442762434, dz=1.4238327525681225, p=1.7529374305140513e-26
- Graph-CoLD_vs_ablation_hard: n=102, diff=0.018285494916563578, dz=0.4577987163803332, p=5.5956741051131204e-06

## Excluded
- FINE: excluded: faithful FINE requires a validated embedding outlier protocol; D5.5 keeps it out instead of reporting approximate or fake rows
- FINE-style: excluded: faithful FINE requires a validated embedding outlier protocol; D5.5 keeps it out instead of reporting approximate or fake rows
- MCRe: excluded: no independently implemented and smoke-passed real-data implementation in this repository
- MORSE: excluded: no independently implemented and smoke-passed real-data implementation in this repository
- Decoupling: excluded: no independently implemented and smoke-passed real-data implementation in this repository
- Flash: excluded: provenance case-study method; no formal two-dataset label-noise implementation
- Argus: excluded: provenance case-study method; no formal two-dataset label-noise implementation
