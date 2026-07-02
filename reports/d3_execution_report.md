# D3 Execution Report

## 1. Implementation summary

D3 implements Stage-2 Graph-CoLD without changing the Stage-1 encoder. The new implementation provides label-space Graph-CDM, evidence-preserving weights, weighted classification loss, and deterministic SOC alert ranking. Public APIs are available as `graph_cdm.forward()`, `evidence.compute()`, `ranking.topk()`, and `loss.compute()`.

## 2. CK-4 Graph-CDM correctness analysis

`D_pred` is computed against the observed label `y_v` for each view prediction. `D_neigh` is KL divergence between node soft labels and neighbor mean soft labels. `D_view` is mode disagreement across view-predicted labels. `D_chain` uses temporal soft-label cosine similarity over temporal pairs.

## 3. CK-5 CoLD degeneracy test

When `rho=0`, `soft_weights()` returns the hard keep mask `GraphCDM(v) <= theta`, matching the CoLD-style hard deletion ablation behavior.

## 4. CK-6 metric consistency notes

ERR is implemented as retention of clean informative samples, favoring low-frequency classes by default. Ranking is stable because ties are resolved by node index after descending priority score. Priority scores combine malicious probability, Graph-CDM, and evidence.

## 5. Known issues / TODOs

`D_chain` currently uses temporal adjacent-pair consistency from `MultiViewGraph.temporal_pairs`; OpTC-specific attack-chain semantics remain D4 work. Real-data convergence and ranking calibration should be rechecked once CICIDS-2017 and MALTLS-22 files are available locally.
