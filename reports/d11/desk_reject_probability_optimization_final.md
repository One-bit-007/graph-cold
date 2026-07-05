# D11 Desk Reject Probability Optimization Final

All probabilities below are internal heuristic estimates, not real acceptance predictions.

## Before Patch
- Desk reject risk: 24% internal heuristic estimate
- FINE-style exclusion not yet directly explained in the manuscript.
- ERR=1.0 could be misread as perfect detection.
- CESNET ceiling-effect interpretation needed stronger wording.
- Decoupling underperformance needed mechanism-level explanation.
- Broad benchmark wording remained in D9.5 text.

## After Patch
- Desk reject risk: 14% internal heuristic estimate
- Human author, funding, competing-interest, and upload confirmations remain.
- Evaluation remains bounded to two verified datasets and implemented baselines.
- No analyst-in-the-loop SOC deployment study is included.

## Target Acceptance Probability 80plus
- Status: heuristic_target_only
- Estimated probability range: 60-75% internal heuristic estimate after D11, conditional on human confirmation
- Condition: human declarations completed
- Condition: reference spot-check completed
- Condition: final PDF visual review completed
- Condition: journal upload metadata completed
