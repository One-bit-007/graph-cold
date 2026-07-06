# Baseline Expansion Report

- Completed: True
- Status: strong
- Original matrix rows unchanged: True
- Expanded rows: 1020
- Added baseline rows: 714
- Included baselines: Co-Teaching, Confident-Learning, Decoupling, FINE, MCRe, MORSE, Noisy-Supervised

## Key Metrics
- Co-Teaching: Macro-F1 mean=0.722969, ERR mean=0.824732, compression mean=0.990124
- CoLD: Macro-F1 mean=0.973171, ERR mean=0.895288, compression mean=0.990853
- Confident-Learning: Macro-F1 mean=0.767907, ERR mean=0.824528, compression mean=0.993595
- Decoupling: Macro-F1 mean=0.644506, ERR mean=0.314434, compression mean=0.989741
- FINE: Macro-F1 mean=0.619185, ERR mean=0.642386, compression mean=0.998683
- Graph-CoLD: Macro-F1 mean=0.991457, ERR mean=1.000000, compression mean=0.989613
- MCRe: Macro-F1 mean=0.680981, ERR mean=0.782408, compression mean=0.989121
- MORSE: Macro-F1 mean=0.671961, ERR mean=0.947247, compression mean=0.998129
- Noisy-Supervised: Macro-F1 mean=0.683640, ERR mean=1.000000, compression mean=0.997891
- ablation_hard: Macro-F1 mean=0.973074, ERR mean=0.895288, compression mean=0.990762

## Statistical Comparisons
- Graph-CoLD_vs_Co-Teaching: n=102, diff=0.26848801850783116, dz=2.193328480661416, p=7.326753454118152e-41
- Graph-CoLD_vs_CoLD: n=102, diff=0.018285494916563578, dz=0.4577987163803332, p=5.5956741051131204e-06
- Graph-CoLD_vs_Confident-Learning: n=102, diff=0.22354953239876246, dz=1.3418402035755759, p=9.240692796934754e-25
- Graph-CoLD_vs_Decoupling: n=102, diff=0.3469510333717703, dz=2.7745797788978304, p=9.786881880399386e-50
- Graph-CoLD_vs_FINE: n=102, diff=0.3722718904140924, dz=1.6937231984683854, p=6.732317201503154e-32
- Graph-CoLD_vs_MCRe: n=102, diff=0.3104762631231711, dz=1.6782312981541738, p=1.3430412425030052e-31
- Graph-CoLD_vs_MORSE: n=102, diff=0.31949565833774196, dz=1.7241813565492767, p=1.7475480709545108e-32
- Graph-CoLD_vs_Noisy-Supervised: n=102, diff=0.30781666442762434, dz=1.4238327525681225, p=1.7529374305140513e-26
- Graph-CoLD_vs_ablation_hard: n=102, diff=0.018382631348990538, dz=0.46324647084918513, p=4.49238347554102e-06

## Excluded
- Flash: excluded: provenance case-study method; no formal two-dataset label-noise implementation
- Argus: excluded: provenance case-study method; no formal two-dataset label-noise implementation
