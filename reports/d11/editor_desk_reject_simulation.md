# D11 Editor Desk Reject Simulation

- Desk reject probability estimate: 24% before D11 patch; expected 14% after patch, internal heuristic estimate
- Decision simulation: send_to_review
- Probability note: internal heuristic estimate only, not a real acceptance prediction.

## Top Acceptance Signals
- Clear C&S fit: noisy-label intrusion detection and SOC alert prioritization.
- Real-data-only reinforced matrix with explicit dataset and baseline scope.
- Graph-CDM is stated as label-space consistency rather than embedding-distance filtering.
- Evidence-retention framing addresses a security-operations failure mode.
- Reproducibility gates, hashes, and candidate package are present.

## Top Desk Reject Risks
- CESNET-TLS-Year22 uses a deterministic postfilter25 subset rather than full archive coverage.
- SOC benefit is supported by retention and compression proxies, not by an analyst study.
- Baseline coverage is bounded to implemented and smoke-passed methods.
- Human author, funding, competing-interest, and upload confirmations remain required.

## Mandatory Pre-Submission Fixes
- Add explicit FINE-style smoke-gate exclusion language.
- Clarify that ERR=1.0 is a retained-mask evidence metric, not a classification score.
- Explain Decoupling under structured SOC noise.
- State the CESNET ceiling-effect interpretation in Discussion and Limitations.
- Remove broad benchmark wording from the final candidate.

## Optional Polish Items
- Human author metadata and institutional statements.
- Final visual pass in the journal upload PDF.
- Reference spot-check by the corresponding author.
