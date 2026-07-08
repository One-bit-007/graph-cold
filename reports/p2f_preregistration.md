# P2f Pre-registration

This file is written before running or reading the final P2f powered comparison
numbers.

## Frozen Hypothesis

P2e's rare-recovery metric mixed recovery with retention and therefore made
soft retention win mechanically. P2f removes that tautology. A method recovers a
rare suspicious sample only when its trained classifier predicts that sample's
clean class correctly.

## Scope

- Datasets: CICIDS-2017, CESNET-TLS-Year22, UNSW-NB15 real local audit windows.
- Noise: tail-concentrated asymmetric label noise at rates 40%, 60%, and 80%.
- Seeds: 0 through 9.
- Test split target: 5000 rows per dataset/seed, stratified when available.
- Train split target: 10000 rows per dataset/seed.
- Methods: CoLD, ablation_hard, Graph-CoLD-soft, Graph-CoLD-semisup.

## Corrected Rare-recovery Definition

For a training node v:

```
eligible(v) := clean(v) AND rare_tail_class(v) AND GraphCDM(v) > theta
recovered(v) := eligible(v) AND classifier(v) == y_true(v)
rare_recovery_rate := sum(recovered) / sum(eligible)
```

Hard deletion is evaluated on the same eligible samples as soft retention. A
deleted sample still exists and can be classified correctly by the resulting
model. Soft retention is not automatically correct.

## Success Criterion

For each dataset and candidate method (`Graph-CoLD-soft` or
`Graph-CoLD-semisup`) compare against `ablation_hard` on:

- tail Macro-F1; and
- corrected rare-recovery.

P2f counts a dataset as PASS if at least one candidate method beats hard deletion
on either metric with paired p < 0.05 after Holm correction and Cohen dz >= 0.3,
under asymmetric tail noise >= 40%, with no significant aggregate Macro-F1
regression.

Global verdict:

- `robust`: PASS on at least two datasets.
- `narrow`: PASS on exactly one dataset.
- `null`: PASS on zero datasets.

The final report must use this verdict as written and must not relabel a narrow
or null result as salvaged.
