# D11 Cross-Reviewer Consensus Risk Map

- Estimated desk reject risk: 14% after D11 patch, internal heuristic estimate
- Estimated major revision risk: 45% internal heuristic estimate
- Estimated minor revision or accept risk: 41% internal heuristic estimate
- Safe to submit after patches: False

## Risk Clusters
### baseline coverage
- Mentioned by: R1, R3
- Severity: high
- Current defense: Formal methods are limited to implemented and smoke-passed baselines.
- Remaining gap: Broader faithful baselines remain future work.
- Recommended patch: Use bounded-comparison wording and avoid broad benchmark claims.
- Rebuttal strategy: Stress artifact honesty and smoke-passed criterion.

### FINE-style exclusion
- Mentioned by: R1, R3
- Severity: medium
- Current defense: D9.5 smoke report records the failure.
- Remaining gap: Formal manuscript needed a direct explanation.
- Recommended patch: Add Discussion/Limitation paragraph explaining exclusion.
- Rebuttal strategy: State that unstable numbers are not reported.

### CESNET subset / ceiling effect
- Mentioned by: R2, R3
- Severity: high
- Current defense: Manuscript states postfilter25 subset.
- Remaining gap: Ceiling effect needed stronger interpretation.
- Recommended patch: Add explicit ceiling-effect paragraph.
- Rebuttal strategy: Frame CESNET as cross-domain stability and retention check.

### SOC operational validity
- Mentioned by: R2
- Severity: high
- Current defense: Compression and ERR are already described as proxies.
- Remaining gap: No analyst study or enterprise case.
- Recommended patch: Keep proxy language and state future analyst validation.
- Rebuttal strategy: Acknowledge limitation directly.

### ERR interpretation
- Mentioned by: R2, R3
- Severity: medium
- Current defense: ERR is defined as evidence retention.
- Remaining gap: ERR=1.0 can be misread.
- Recommended patch: Add explicit statement that ERR is not detection accuracy.
- Rebuttal strategy: Explain retained-mask threshold and clean informative subset.

### Graph-CDM novelty
- Mentioned by: R1
- Severity: medium
- Current defense: Method section uses label-space components.
- Remaining gap: Novelty can look incremental.
- Recommended patch: Emphasize label-space graph consistency and evidence preservation.
- Rebuttal strategy: Contrast with embedding-distance and hard deletion.

### Co-Teaching-lite naming
- Mentioned by: R1, R3
- Severity: low
- Current defense: Already named lite.
- Remaining gap: None if wording remains precise.
- Recommended patch: Keep lite wording.
- Rebuttal strategy: State no full Co-Teaching claim.

### Decoupling weakness
- Mentioned by: R1, R2
- Severity: medium
- Current defense: Decoupling entered D9.5 matrix.
- Remaining gap: Underperformance may look unfair without mechanism explanation.
- Recommended patch: Add structured-noise disagreement limitation.
- Rebuttal strategy: Explain disagreement is vulnerable to correlated SOC labels.

### references / related work
- Mentioned by: R1, R3
- Severity: medium
- Current defense: References are limited and verified.
- Remaining gap: Reviewer may request wider literature.
- Recommended patch: Do not add unverified citations in D11; note bounded related work.
- Rebuttal strategy: Offer future revision literature expansion if requested.

### submission declarations
- Mentioned by: AE, R3
- Severity: medium
- Current defense: Data availability exists and D11 adds AI statement.
- Remaining gap: Human author/funding/COI confirmation remains.
- Recommended patch: Keep submission_ready false.
- Rebuttal strategy: Flag as pre-upload task, not science blocker.
