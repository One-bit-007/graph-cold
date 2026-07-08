# P2e Pre-registration

This file is written after the A1 tension gate and before running any redesigned
Part B comparison matrix.

## Frozen hypothesis

Suspicious-but-clean rare/tail training samples can be harmed by hard deletion.
An evidence-rescue weight or evidence-aware semi-supervised mode may recover
tail performance by keeping high-evidence/high-CDM samples active without
letting their observed noisy labels dominate supervised cross-entropy.

## Primary comparison

Compare the following methods under identical real-data splits, graph/CDM,
observed noisy labels, and seeds:

- `CoLD`
- `ablation_hard`
- `Graph-CoLD-soft`
- `Graph-CoLD-semisup`
- `Graph-CoLD`

## Evaluation scope

- Datasets: CICIDS-2017, CESNET-TLS-Year22, UNSW-NB15.
- Noise: tail-concentrated asymmetric label noise at rates 40%, 60%, and 80%
  within tail malicious classes.
- Seeds: 0, 1, 2.
- Data policy: real data only; deterministic tail-preserving audit windows
  larger than P2d's 1500-row window.

## Success criterion

P2e is `salvaged` if `Graph-CoLD-semisup` or `Graph-CoLD-soft` beats
`ablation_hard` on either:

- tail-class Macro-F1, or
- rare-evidence recovery rate,

with paired p < 0.05 and Cohen dz >= 0.3 on at least one real dataset under
asymmetric tail noise >= 40%, while showing no significant regression on
aggregate Macro-F1.

P2e is `partial` if the direction is positive but only one of p < 0.05 or
dz >= 0.3 holds, or if the benefit appears only in one narrow dataset/noise
slice with no aggregate Macro-F1 regression.

P2e is `null` if neither redesigned method beats hard deletion under the above
criteria, or if gains are accompanied by a significant aggregate Macro-F1
regression.

## Integrity constraints

- `flip_mask` and clean labels may be used only for offline measurement.
- Graph-CDM, evidence, graph construction, weights, and training inputs must use
  observed noisy labels plus unsupervised feature anomaly only.
- If the result is `null`, report fallback A honestly and do not tune thresholds
  after reading final numbers.
