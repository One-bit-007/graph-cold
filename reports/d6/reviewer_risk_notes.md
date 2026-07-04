# Reviewer Risk Notes

## Risk 1 - Baseline exclusion

FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full Co-Teaching are not included in the formal D6 tables because they do not have independently implemented and smoke-passed real-data rows in this repository. There is no fake implementation in the result matrix. These methods are suitable future extensions once faithful implementations are available.

## Risk 2 - CESNET subset

CESNET-TLS-Year22 uses a deterministic evaluation subset and postfilter25 class policy. Every result row records the sample_policy field, and the manuscript must not claim a full archive evaluation.

## Risk 3 - Co-Teaching-lite

Co-Teaching-lite is not a complete deep Co-Teaching reproduction. It is named lite to avoid over-comparison and to make its implementation scope clear.

## Risk 4 - Ceiling effect

CESNET-TLS-Year22 Macro-F1 is close to 0.995 for the strongest methods. The paper should not overstate Macro-F1 margins on this dataset; the safer emphasis is stability and ERR.
