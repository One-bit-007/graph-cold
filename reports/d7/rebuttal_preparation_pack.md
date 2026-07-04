# Rebuttal Preparation Pack

## Baseline limitation
Likely concern: important named baselines are absent.
Response strategy: emphasize that the result matrix excludes methods without
faithful, independently implemented, smoke-passed real-data rows. This prevents
unfair or approximate comparisons.

## CESNET subset
Likely concern: CESNET-TLS-Year22 is large, but the paper uses a deterministic
audit-window subset.
Response strategy: point to Table 1, Experimental Setup, Limitations, and the
reproducibility README. Do not expand the claim beyond the evaluated subset.

## Co-Teaching-lite
Likely concern: the baseline may not represent full Co-Teaching.
Response strategy: agree and state that it is intentionally named lite. The
manuscript does not use it as a substitute for a full deep reproduction.

## ERR definition
Likely concern: evidence retention could be gamed by retaining too much.
Response strategy: ERR is paired with compression ratio and Tail-ERR. The
ablation_hard comparison isolates hard deletion versus evidence preserving
weights.

## OpTC and MALTLS-22
Likely concern: the introduction motivates SOC settings but no enterprise OpTC
case is reported.
Response strategy: state that the manuscript is a real-data robustness study on
two verified network datasets. OpTC requires verified provenance event tables
before formal enterprise evaluation.

## Graph construction
Likely concern: five-view design is not fully active for every dataset.
Response strategy: active views are dataset-dependent. The paper does not claim
process or threat-intelligence evidence where the underlying fields are absent.
