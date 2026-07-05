# D11 Desk-Reject Patch Plan

- Trigger: pre_patch_desk_reject_probability_above_20_percent_internal_heuristic
- Source estimate: 24% before D11 patch; expected 14% after patch, internal heuristic estimate
- Probability note: internal heuristic estimate only.

## Planned And Applied Patches
### E1: FINE-style exclusion explanation
- Action: Explain that FINE-style was implemented but excluded because it failed the pre-registered CICIDS symmetric smoke gate.
- Status after generation: applied

### E2: CESNET ceiling-effect clarification
- Action: Clarify that CESNET postfilter25 is interpreted as cross-domain stability and evidence-retention evidence under a Macro-F1 ceiling.
- Status after generation: applied

### E3: Decoupling limitation
- Action: Explain why disagreement-only filtering can underperform under correlated SOC noise.
- Status after generation: applied

### E4: ERR=1.0 clarification
- Action: State that ERR is a retained-mask evidence-retention measure, not detection accuracy.
- Status after generation: applied

### E5_E6: Baseline coverage and overclaiming wording
- Action: Use implemented smoke-passed baseline language and remove broad overclaiming terms.
- Status after generation: applied
