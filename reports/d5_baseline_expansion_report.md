# Baseline Expansion Report

- Completed: True
- Status: strong
- Original matrix rows unchanged: True
- Expanded rows: 1530
- Added baseline rows: 1071
- Included baselines: Co-Teaching, Confident-Learning, Decoupling, FINE, MCRe, MORSE, Noisy-Supervised

## Key Metrics
- Co-Teaching: Macro-F1 mean=0.704563, ERR mean=0.599736, compression mean=0.925580
- CoLD: Macro-F1 mean=0.762342, ERR mean=0.763609, compression mean=0.914903
- Confident-Learning: Macro-F1 mean=0.672665, ERR mean=0.706523, compression mean=0.920199
- Decoupling: Macro-F1 mean=0.563241, ERR mean=0.375693, compression mean=0.944462
- FINE: Macro-F1 mean=0.563424, ERR mean=0.679688, compression mean=0.973892
- Graph-CoLD: Macro-F1 mean=0.854647, ERR mean=1.000000, compression mean=0.907017
- MCRe: Macro-F1 mean=0.620243, ERR mean=0.755938, compression mean=0.931545
- MORSE: Macro-F1 mean=0.619660, ERR mean=0.938089, compression mean=0.929598
- Noisy-Supervised: Macro-F1 mean=0.613149, ERR mean=1.000000, compression mean=0.972850
- ablation_hard: Macro-F1 mean=0.842201, ERR mean=0.887910, compression mean=0.916426

## Statistical Comparisons
- Graph-CoLD_vs_Co-Teaching: n=153, diff=0.1500842494096197, dz=1.0603581078215287, p=4.5735427982828894e-27
- Graph-CoLD_vs_CoLD: n=153, diff=0.09230590126329832, dz=0.6503353634133742, p=1.1449023614620517e-13
- Graph-CoLD_vs_Confident-Learning: n=153, diff=0.18198209754020353, dz=1.1954256618648096, p=1.5899060767748167e-31
- Graph-CoLD_vs_Decoupling: n=153, diff=0.291406333519473, dz=2.21184068567874, p=6.725689618398308e-61
- Graph-CoLD_vs_FINE: n=153, diff=0.2912232607142284, dz=1.3838270993812145, p=1.3929962421954552e-37
- Graph-CoLD_vs_MCRe: n=153, diff=0.23440467850184085, dz=1.1446768223196138, p=7.382210894709315e-30
- Graph-CoLD_vs_MORSE: n=153, diff=0.23498750005667637, dz=1.1169285454846283, p=6.085981822411748e-29
- Graph-CoLD_vs_Noisy-Supervised: n=153, diff=0.24149821777589753, dz=1.170372507334095, p=1.053678302771617e-30
- Graph-CoLD_vs_ablation_hard: n=153, diff=0.012446625088628855, dz=0.3717444744348079, p=4.454008545799064e-06

## Excluded
- Flash: excluded: provenance case-study method; no formal real-data label-noise implementation
- Argus: excluded: provenance case-study method; no formal real-data label-noise implementation
