# Reviewer 2 - Security / SOC

R2 summary
The SOC motivation is credible, especially evidence retention, but the paper must avoid implying deployment validation without analyst studies or verified enterprise provenance.

Likely recommendation: minor-to-major revision
Score 0-10: 7
Fatal risk: no

## Major Concerns
- Compression ratio and ERR are operational proxies, not analyst-time measurements.
- ERR=1.0 can look suspicious unless explicitly defined as retention over clean informative samples.
- No OpTC formal case means enterprise realism remains limited.
- CESNET ceiling effect makes classifier margin less informative.

## Minor Concerns
- Active view masks should be stated as contract-driven.
- Ranking should be framed as an alert-priority proxy.

## Questions to Authors
- What does a SOC analyst gain from retained informative samples?
- How would compression be used in a live queue?
- What evidence is missing for an enterprise case?
