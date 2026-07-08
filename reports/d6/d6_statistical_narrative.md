# Statistical Narrative

## Technical summary

The paper artifacts aggregate the verified real-data evaluation matrix from `results/table_main_expanded.csv`. Across matched dataset, noise, beta, and seed cells, Graph-CoLD improves Macro-F1 over CoLD by 2.17 percentage points. The paired grouped t-test reports p=1.27e-04, effect size dz=0.552, and n=51 pairs. This supports a claim of consistent improvement, not a claim beyond the tested data scope.

## Dataset scope

    The formal result scope is CICIDS-2017, CESNET-TLS-Year22, UNSW-NB15. The sample policy is explicit in every row: CICIDS-2017 uses the full postfilter11 protocol after minimum-count filtering and dominant-class downsampling, CESNET-TLS-Year22 uses a deterministic audit-window subset followed by postfilter25 stratified splitting, and UNSW-NB15 uses the verified local partition layout with postfilter class policy.

## Method scope

The matrix includes Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE.

## Excluded baselines

    The following methods are outside the formal real-data label-noise matrix:

- Co-Teaching+: not independently implemented as a formal real-data row; represented by Co-Teaching-lite where applicable
- cleanlab: not independently implemented as a formal real-data row; represented by Confident-Learning where applicable

No generated stand-in rows are reported for these methods.

## Graph-CoLD vs CoLD

The paired grouped test controls for scenario difficulty by matching dataset, noise type, noise rate, graph beta, and seed. The observed 2.17 percentage-point lift is statistically reliable at p=1.27e-04. The effect is modest in absolute terms because both methods are strong on clean and easy settings, especially CESNET-TLS-Year22.

## Graph-CoLD vs noise-learning baselines

Relative to Noisy-Supervised, Confident-Learning, Co-Teaching, Decoupling, FINE, MCRe, and MORSE, Graph-CoLD has higher average Macro-F1 in the expanded matrix. These comparisons should be read as robustness under noisy labels within the implemented baselines, not as an exhaustive benchmark against every published variant.

## ERR interpretation

Graph-CoLD's mean ERR_final is 0.7894, matching 0.7894 for ablation_hard. The 0.00 percentage-point gap means the clean rerun does not show an ERR-based retention gain over hard deletion.

## CESNET ceiling effect

    CESNET-TLS-Year22 Macro-F1 is high for several methods, so small improvements should not be over-read. The high-noise Graph-CoLD vs CoLD lift on CESNET is 0.60 percentage points, while CICIDS-2017 shows a larger high-noise lift of 2.24 percentage points. The C&S-ready wording is: Graph-CoLD improves Macro-F1 under noisy labels, while ERR retention is tied with hard deletion in the clean rerun.

    ## UNSW-NB15 extension

    UNSW-NB15 contributes a verified third dataset using temporal and process/feature-block views. Its high-noise Graph-CoLD vs CoLD lift is 0.23 percentage points; this should be described as an additional robustness check, not as a provenance-graph SOC case study.

## Operational meaning

Compression ratio is an operational alert reduction proxy. Combined with ERR, it asks whether fewer reviewed alerts still retain clean informative evidence. This is the SOC-facing interpretation: the method is useful when it shortens a review queue without discarding the evidence analysts need.

## Caution against overclaiming

    The results are traceable to the verified real datasets in scope and the implemented baselines. The paper should avoid universal superiority language and should state that omitted provenance systems require separate future evaluation before formal comparison.

## Conclusion-ready insight block

- Graph-CoLD shows consistent improvement over CoLD in paired scenario-level testing while remaining close to CoLD on clean labels.
- The largest practical gains occur under CICIDS-2017 noisy settings, where structured label-space consistency helps absorb corrupted training labels.
- Evidence retention matches hard deletion in this clean rerun, so the evidence-preservation claim must be scoped as unresolved rather than positive.
- Compression ratio is reported as an operational alert reduction proxy rather than a direct SOC labor measurement.
- CESNET-TLS-Year22 should be interpreted as a high-ceiling, verified TLS application-classification subset.
- UNSW-NB15 adds a third verified real-data partition with temporal and process/feature-block views.
