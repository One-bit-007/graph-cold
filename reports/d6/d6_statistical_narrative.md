# Statistical Narrative

## Technical summary

The paper artifacts aggregate the verified real-data evaluation matrix from `results/table_main_expanded.csv`. Across matched dataset, noise, beta, and seed cells, Graph-CoLD improves Macro-F1 over CoLD by 1.83 percentage points. The paired grouped t-test reports p=5.60e-06, effect size dz=0.458, and n=102 pairs. This supports a claim of consistent improvement, not a claim beyond the tested data scope.

## Dataset scope

The formal result scope is CICIDS-2017 postfilter11 and CESNET-TLS-Year22 postfilter25. The sample policy is explicit in every row: CICIDS-2017 uses the full postfilter11 protocol after minimum-count filtering and dominant-class downsampling, while CESNET-TLS-Year22 uses a deterministic audit-window subset followed by postfilter25 stratified splitting.

## Method scope

The matrix includes Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE.

## Excluded baselines

The following methods are outside the formal two-dataset label-noise matrix:

- Argus: excluded: provenance case-study method; no formal two-dataset label-noise implementation
- Co-Teaching+: not independently implemented and verified on real data
- Flash: excluded: provenance case-study method; no formal two-dataset label-noise implementation
- cleanlab: legacy audit key; official cleanlab is represented by the Confident-Learning method row

No generated stand-in rows are reported for these methods.

## Graph-CoLD vs CoLD

The paired grouped test controls for scenario difficulty by matching dataset, noise type, noise rate, graph beta, and seed. The observed 1.83 percentage-point lift is statistically reliable at p=5.60e-06. The effect is modest in absolute terms because both methods are strong on clean and easy settings, especially CESNET-TLS-Year22.

## Graph-CoLD vs noise-learning baselines

Relative to Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE, Graph-CoLD has higher average Macro-F1 in the expanded matrix. These comparisons should be read as robustness under noisy labels within the implemented baselines, not as an exhaustive benchmark against every published variant.

## ERR interpretation

Graph-CoLD's mean ERR_final is 1.0000, compared with 0.8953 for ablation_hard. The 10.47 percentage-point gap supports the evidence retention claim: soft weights preserve clean informative evidence better than hard deletion in the evaluated scenarios.

## CESNET ceiling effect

CESNET-TLS-Year22 Macro-F1 is high for several methods, so small improvements should not be over-read. The high-noise Graph-CoLD vs CoLD lift on CESNET is 0.00 percentage points, while CICIDS-2017 shows a larger high-noise lift of 5.19 percentage points. The C&S-ready wording is: Graph-CoLD improves robustness and evidence retention under noisy labels, with the clearest margins on CICIDS-2017.

## Operational meaning

Compression ratio is an operational alert reduction proxy. Combined with ERR, it asks whether fewer reviewed alerts still retain clean informative evidence. This is the SOC-facing interpretation: the method is useful when it shortens a review queue without discarding the evidence analysts need.

## Caution against overclaiming

The results are traceable to two verified real datasets and the implemented baselines. The paper should avoid universal superiority language and should state that omitted provenance systems require separate future evaluation before formal comparison.

## Conclusion-ready insight block

- Graph-CoLD shows consistent improvement over CoLD in paired scenario-level testing while remaining close to CoLD on clean labels.
- The largest practical gains occur under CICIDS-2017 noisy settings, where structured label-space consistency helps absorb corrupted training labels.
- Evidence retention improves over hard deletion, supporting the use of soft weights for preserving clean informative alerts.
- Compression ratio is reported as an operational alert reduction proxy rather than a direct SOC labor measurement.
- CESNET-TLS-Year22 should be interpreted as a high-ceiling, verified TLS application-classification subset.
