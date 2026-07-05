# Reviewer 3 - Data / Experimental Rigor

R3 summary
The artifact is unusually transparent about scope, hashes, and exclusions, but the reviewer will scrutinize deterministic subsets, omitted datasets, scenario dependence, and baseline fidelity.

Likely recommendation: major revision possible, send-to-review likely
Score 0-10: 6.5
Fatal risk: no

## Major Concerns
- CESNET postfilter25 is not a full archive evaluation.
- CICIDS postfilter11 can bias class coverage and must remain declared.
- MALTLS-22 and OpTC omissions need direct justification.
- Paired p-values may be correlated across scenario settings.
- FINE-style failed smoke and is excluded; this should be transparent.

## Minor Concerns
- Reference the result table and audit hashes near claims.
- Avoid interpreting p-values as independent operational trials.

## Questions to Authors
- Are split/noise/model seeds paired for every comparison?
- Are active views fixed by dataset contracts?
- Are raw datasets redistributed?
