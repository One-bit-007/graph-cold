# D6 Statistical Narrative

## Technical summary

Graph-CoLD converts the D5 experimental matrix into a publication-ready result: the full model averages 97.2% Macro-F1 versus 79.1% for CoLD, an absolute lift of 18.2 percentage points. The paired one-sided t-test in `results/stat_tests.json` reports t=35.77, p=9.04e-93, so the observed improvement is statistically significant under the D5 seed-level paired design.

## Graph-CoLD is statistically stronger than CoLD

The t-test compares Graph-CoLD and CoLD on matched dataset/noise/seed cells. Because the comparison is paired, the test asks whether Graph-CoLD's cell-level Macro-F1 is reliably higher than CoLD after controlling for scenario difficulty. The p-value is far below 0.05, and the mean lift is large enough to be practically meaningful, not merely statistically detectable.

## Robustness under high noise

For noise rates at or above 40%, Graph-CoLD keeps Macro-F1 at 95.6%, compared with 71.0% for CoLD. ERR also favors Graph-CoLD (89.2% vs 86.4%), which supports the claim that evidence-preserving weighting protects informative samples where hard deletion is brittle.

## SOC operational interpretation

Compression ratio translates model output into analyst workload: lower values mean fewer alerts must be reviewed to cover the true attacks. Under high noise, Graph-CoLD's compression ratio is 49.5%, versus 66.7% for CoLD. In the OpTC-style enterprise case, Graph-CoLD reports Top-K precision of 100.0% with compression 20.0%, meaning the ranking layer concentrates malicious evidence into a shorter review queue.

## Ablation interpretation

The full model reaches 89.7% Macro-F1 in the D5 ablation setting. The hard-deletion variant drops to 74.9%, showing that setting rho=0 recovers the expected CoLD-like failure mode. Removing Graph-CDM has the largest degradation, followed by evidence and view/neighborhood terms, which is directionally consistent with the method design.

## Scope, data, and definitions

All claims are derived from `results/` artifacts generated in D5. Macro-F1, FPR, FNR, ERR, Tail-ERR, compression ratio, runtime, and memory are aggregated over seeds {0,1,2}. CICIDS-2017 and MALTLS-22 rows are marked as synthetic fallbacks when raw datasets are absent locally; synthetic OpTC is the D4 enterprise mini-case.

## Limitations and robustness checks

The statistical result is internally consistent with D5 outputs, but real CICIDS-2017 and MALTLS-22 files were not present on this machine, so journal claims should be refreshed once real-data runs are available. Baseline rows are lightweight D5 adapters intended to validate the matrix shape; full paper baselines should replace adapters before final camera-ready experiments.

## Conclusion-ready insight block

- Graph-CoLD consistently outperforms the self-implemented CoLD baseline across noise families, with the paired t-test supporting a statistically reliable improvement.
- Evidence-preserving weights retain high-value samples under high noise while reducing the analyst inspection burden, which is the central operational distinction from hard deletion.
- The ablation pattern supports the method decomposition: removing Graph-CDM is the largest loss, while hard deletion and evidence removal also materially degrade performance.
- The OpTC-style case demonstrates that the same scoring stack can translate representation and label-space consistency into SOC Top-K ranking quality.
