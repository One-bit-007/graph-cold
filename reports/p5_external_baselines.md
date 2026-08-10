# P5 External Baselines (Confident Learning + full Co-Teaching) under the P2f Protocol

- Rows: 180 / expected 180
- Failures: 0
- Dataset hashes match frozen P2f: {'CESNET-TLS-Year22': True, 'CICIDS-2017': True, 'UNSW-NB15': True}

## Mean metrics (over seeds x rates)

| dataset | method | macro_f1 | tail_macro_f1 | tail_recall |
| --- | --- | --- | --- | --- |
| CESNET-TLS-Year22 | Co-Teaching | 0.4877 | 0.1788 | 0.1811 |
| CESNET-TLS-Year22 | Confident-Learning | 0.8053 | 0.7081 | 0.6276 |
| CICIDS-2017 | Co-Teaching | 0.5667 | 0.1182 | 0.1276 |
| CICIDS-2017 | Confident-Learning | 0.7693 | 0.6053 | 0.5174 |
| UNSW-NB15 | Co-Teaching | 0.4188 | 0.0552 | 0.1196 |
| UNSW-NB15 | Confident-Learning | 0.4987 | 0.2541 | 0.3173 |

## Scenario-paired tests vs Graph-CoLD-soft (Holm-corrected per dataset)

| dataset | method | metric | mean delta | p_holm | cohens_dz | CI95 |
| --- | --- | --- | --- | --- | --- | --- |
| CESNET-TLS-Year22 | Confident-Learning | macro_f1 | +0.1138 | 1.47e-11 | +2.01 | [+0.0940, +0.1333] |
| CESNET-TLS-Year22 | Confident-Learning | tail_macro_f1 | +0.2335 | 8.29e-12 | +2.08 | [+0.1938, +0.2722] |
| CESNET-TLS-Year22 | Confident-Learning | tail_recall | +0.2831 | 6.00e-12 | +3.28 | [+0.2532, +0.3140] |
| CESNET-TLS-Year22 | Co-Teaching | macro_f1 | -0.2038 | 1.00e+00 | -3.01 | [-0.2276, -0.1813] |
| CESNET-TLS-Year22 | Co-Teaching | tail_macro_f1 | -0.2959 | 1.00e+00 | -2.21 | [-0.3434, -0.2528] |
| CESNET-TLS-Year22 | Co-Teaching | tail_recall | -0.1634 | 1.00e+00 | -1.38 | [-0.2063, -0.1243] |
| CICIDS-2017 | Confident-Learning | macro_f1 | +0.0481 | 1.89e-03 | +0.67 | [+0.0235, +0.0740] |
| CICIDS-2017 | Confident-Learning | tail_macro_f1 | +0.1711 | 1.83e-07 | +1.32 | [+0.1262, +0.2172] |
| CICIDS-2017 | Confident-Learning | tail_recall | +0.1891 | 1.83e-07 | +1.31 | [+0.1383, +0.2417] |
| CICIDS-2017 | Co-Teaching | macro_f1 | -0.1545 | 1.00e+00 | -3.28 | [-0.1711, -0.1381] |
| CICIDS-2017 | Co-Teaching | tail_macro_f1 | -0.3159 | 1.00e+00 | -2.87 | [-0.3552, -0.2779] |
| CICIDS-2017 | Co-Teaching | tail_recall | -0.2008 | 1.00e+00 | -1.73 | [-0.2414, -0.1599] |
| UNSW-NB15 | Confident-Learning | macro_f1 | +0.0138 | 1.16e-02 | +0.54 | [+0.0046, +0.0224] |
| UNSW-NB15 | Confident-Learning | tail_macro_f1 | +0.0811 | 1.04e-05 | +1.04 | [+0.0529, +0.1090] |
| UNSW-NB15 | Confident-Learning | tail_recall | +0.0886 | 4.24e-04 | +0.79 | [+0.0476, +0.1277] |
| UNSW-NB15 | Co-Teaching | macro_f1 | -0.0660 | 1.00e+00 | -2.37 | [-0.0763, -0.0562] |
| UNSW-NB15 | Co-Teaching | tail_macro_f1 | -0.1178 | 1.00e+00 | -1.84 | [-0.1408, -0.0954] |
| UNSW-NB15 | Co-Teaching | tail_recall | -0.1091 | 1.00e+00 | -1.08 | [-0.1447, -0.0735] |
