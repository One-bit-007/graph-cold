"""Graph-CDM and evidence-preserving soft reweighting (core contribution).

Graph Causal Divergence Metric
------------------------------
For each node v:

    GraphCDM(v) = lambda_pred * D_pred(v)
                + lambda_neigh * D_neigh(v)
                + lambda_view  * D_view(v)
                + lambda_chain * D_chain(v)      # OpTC only; 0 on flow data

    D_pred(v)  : P(soft-label(v) != observed-label(v))   [inherits CoLD CDM]
    D_neigh(v) : KL( y_hat(v) || Agg_{u in N(v)} y_hat(u) )   graph-neighborhood
    D_view(v)  : variance/dispersion of per-view predicted labels
    D_chain(v) : attack-chain context inconsistency (provenance graphs)

Evidence-preserving soft reweighting (NOT deletion)
---------------------------------------------------
Evidence score (protect low-frequency classes, early-APT, boundary anomalies):

    e(v)  = freq_protect(n_{y_v}) * (1 + gamma * anomaly(v))
    ~e(v) = normalize(e(v)) in [0, 1]

Soft weight (high divergence -> down-weight, never zero):

    w(v) = sigmoid(-kappa * (GraphCDM(v) - theta)) * (1 - rho) + rho * ~e(v)

Downstream weighted robust loss:  L = sum_v w(v) * CE(f(z_v), y_v)

Setting rho=0 and hard-thresholding recovers CoLD's deletion -> ablation baseline.

Contract
--------
graph_cdm(view_preds, neigh_agg, soft_labels, observed, cfg) -> np.ndarray [N]
evidence_score(y, anomaly, cfg) -> np.ndarray [N]
soft_weights(cdm, evidence, cfg) -> np.ndarray [N]
"""
from __future__ import annotations

import numpy as np


def graph_cdm(view_preds, neigh_agg, soft_labels, observed, cfg):
    """Return per-node Graph-CDM in [0, 1]."""
    raise NotImplementedError("TODO(Codex): weighted sum of D_pred/D_neigh/D_view(/D_chain).")


def evidence_score(y, anomaly, cfg):
    """Low-frequency-class protection * anomaly boost, normalized to [0,1]."""
    raise NotImplementedError("TODO(Codex)")


def soft_weights(cdm, evidence, cfg):
    """w = sigmoid(-kappa*(cdm-theta))*(1-rho) + rho*evidence  (no zeros)."""
    raise NotImplementedError("TODO(Codex)")
