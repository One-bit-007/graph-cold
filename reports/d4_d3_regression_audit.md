# D4 D3 Regression Audit

## Summary

The D3 Stage-2 implementation was audited before adding D4 functionality. Graph-CDM remains in label / prediction space and does not use embedding distance.

## Checks

- D_pred compares per-view labels with observed training labels.
- D_neigh is KL divergence over label probabilities, normalized to [0, 1], with empty-neighbor fallback.
- D_view is mode disagreement across view labels.
- Evidence scores support `log` and `inverse` frequency protection and min-max normalization.
- Positive rho keeps soft weights finite and strictly positive.
- ERR now follows `sum(w*e)/sum(e)`, Tail-ERR uses the tail subset, and final ERR averages both.

## Issues

None.
