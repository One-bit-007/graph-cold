# D7 Final Reviewer Simulation

## Overall decision risk

The package is stronger than the earlier draft because it uses verified
CICIDS-2017 and CESNET-TLS-Year22 rows only, identifies the CESNET subset policy,
and avoids reporting unsupported datasets or baselines. Submission readiness
remains false because reference metadata and broader baseline coverage still need
author review.

## Major likely concerns

1. Baseline limitation: FINE, MCRe, MORSE, Flash, Argus, Decoupling, and full
Co-Teaching are excluded. Defense: the paper reports only implemented and
smoke-passed real-data methods.
2. CESNET subset: CESNET-TLS-Year22 is not reported as full archive. Defense:
the subset policy is stated in the abstract-adjacent setup, tables, limitations,
and reproducibility notes.
3. Co-Teaching-lite: the method is named as lightweight approximation. Defense:
the manuscript does not equate it with the original full deep method.
4. ERR definition: the manuscript links ERR to retained clean informative
evidence and reports hard-deletion comparison.
5. OpTC absence: the manuscript states that real provenance events are required
before an enterprise case can be formal.
6. MALTLS-22 absence: the manuscript states the source did not pass the local
verification gate.
7. Graph construction: process and threat-intelligence views are explicitly
disabled where data fields are missing.

## Key numbers

- Graph-CoLD vs CoLD mean Macro-F1 difference: 1.83 pp
- p-value: 5.60e-06
- Effect size dz: 0.458
- Graph-CoLD ERR_final: 1.0000
- ablation_hard ERR_final: 0.8953
