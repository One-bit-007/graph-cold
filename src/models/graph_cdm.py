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

from dataclasses import dataclass
from typing import Any

import numpy as np


EPS = 1e-12


@dataclass
class GraphCDMResult:
    score: np.ndarray
    d_pred: np.ndarray
    d_neigh: np.ndarray
    d_view: np.ndarray
    d_chain: np.ndarray


def forward(view_preds, soft_labels, observed, cfg, graph=None, neigh_agg=None, temporal_pairs=None, return_components=False):
    """Compute label-space Graph-CDM.

    Parameters
    ----------
    view_preds:
        Per-view label predictions, shaped [M, N] or dict(view -> [N]).
    soft_labels:
        Label probabilities y_hat, shaped [N, K].
    observed:
        Observed / ground-truth labels y_v, shaped [N].
    cfg:
        Config dict containing graph_cdm lambda weights.
    graph:
        Optional MultiViewGraph used to aggregate neighbor soft labels.
    neigh_agg:
        Optional precomputed neighbor label mean [N, K].
    temporal_pairs:
        Optional [2, E_t] temporal predecessor/successor pairs. Defaults to
        graph.temporal_pairs when present.
    return_components:
        If true, return GraphCDMResult. Otherwise return score [N].
    """
    view_matrix = _as_view_label_matrix(view_preds)
    probs = _normalize_rows(np.asarray(soft_labels, dtype=np.float64))
    observed = np.asarray(observed, dtype=np.int64)
    if view_matrix.shape[1] != observed.shape[0] or probs.shape[0] != observed.shape[0]:
        raise ValueError("view_preds, soft_labels, and observed must agree on N.")

    cfg_cdm = _section(cfg, "graph_cdm")
    d_pred = _d_pred(view_matrix, observed)
    d_neigh = _d_neigh(probs, neigh_agg if neigh_agg is not None else _neighbor_soft_mean(probs, graph))
    d_view = _d_view(view_matrix)
    if temporal_pairs is None and graph is not None:
        temporal_pairs = getattr(graph, "temporal_pairs", None)
    d_chain = _d_chain(probs, temporal_pairs)

    score = (
        float(cfg_cdm.get("lambda_pred", 0.4)) * d_pred
        + float(cfg_cdm.get("lambda_neigh", 0.3)) * d_neigh
        + float(cfg_cdm.get("lambda_view", 0.3)) * d_view
        + float(cfg_cdm.get("lambda_chain", 0.0)) * d_chain
    )
    score = np.clip(score, 0.0, 1.0)
    result = GraphCDMResult(score=score, d_pred=d_pred, d_neigh=d_neigh, d_view=d_view, d_chain=d_chain)
    return result if return_components else result.score


def graph_cdm(view_preds, neigh_agg, soft_labels, observed, cfg):
    """Return per-node Graph-CDM in [0, 1]."""
    return forward(view_preds, soft_labels, observed, cfg, neigh_agg=neigh_agg)


def evidence_score(y, anomaly, cfg):
    """Low-frequency-class protection * anomaly boost, normalized to [0,1]."""
    from src.models.evidence import compute

    return compute(y, cfg, anomaly=anomaly)


def soft_weights(cdm, evidence, cfg):
    """w = sigmoid(-kappa*(cdm-theta))*(1-rho) + rho*evidence  (no zeros)."""
    cfg_ev = _section(cfg, "evidence_preserving")
    cdm = np.asarray(cdm, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64)
    if cdm.shape != evidence.shape:
        raise ValueError("cdm and evidence must have the same shape.")
    theta = float(cfg_ev.get("theta", 0.5))
    rho = float(cfg_ev.get("rho", 0.2))
    if bool(cfg_ev.get("ablation_hard", False)) or np.isclose(rho, 0.0):
        return (cdm <= theta).astype(np.float64)
    kappa = float(cfg_ev.get("kappa", 4.0))
    sigmoid = 1.0 / (1.0 + np.exp(kappa * (cdm - theta)))
    weights = sigmoid * (1.0 - rho) + rho * evidence
    return np.clip(weights, 0.0, 1.0)


def _d_pred(view_matrix: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.mean(view_matrix != observed[None, :], axis=0).astype(np.float64)


def _d_view(view_matrix: np.ndarray) -> np.ndarray:
    out = np.zeros(view_matrix.shape[1], dtype=np.float64)
    for idx in range(view_matrix.shape[1]):
        _, counts = np.unique(view_matrix[:, idx], return_counts=True)
        out[idx] = 1.0 - float(counts.max()) / float(view_matrix.shape[0])
    return out


def _d_neigh(soft_labels: np.ndarray, neigh_agg: np.ndarray | None) -> np.ndarray:
    if neigh_agg is None:
        neigh_agg = soft_labels
    neigh = _normalize_rows(np.asarray(neigh_agg, dtype=np.float64))
    kl = np.sum(soft_labels * (np.log(soft_labels + EPS) - np.log(neigh + EPS)), axis=1)
    return _minmax(kl)


def _d_chain(soft_labels: np.ndarray, temporal_pairs) -> np.ndarray:
    out = np.zeros(soft_labels.shape[0], dtype=np.float64)
    counts = np.zeros(soft_labels.shape[0], dtype=np.float64)
    if temporal_pairs is None:
        return out
    pairs = np.asarray(temporal_pairs, dtype=np.int64)
    if pairs.size == 0:
        return out
    if pairs.shape[0] != 2:
        pairs = pairs.T
    src, dst = pairs
    src_prob = soft_labels[src]
    dst_prob = soft_labels[dst]
    denom = np.linalg.norm(src_prob, axis=1) * np.linalg.norm(dst_prob, axis=1)
    sim = np.sum(src_prob * dst_prob, axis=1) / np.maximum(denom, EPS)
    div = 1.0 - np.clip(sim, 0.0, 1.0)
    np.add.at(out, src, div)
    np.add.at(out, dst, div)
    np.add.at(counts, src, 1.0)
    np.add.at(counts, dst, 1.0)
    return out / np.maximum(counts, 1.0)


def _neighbor_soft_mean(soft_labels: np.ndarray, graph: Any | None) -> np.ndarray | None:
    if graph is None:
        return None
    acc = np.zeros_like(soft_labels, dtype=np.float64)
    degree = np.zeros(soft_labels.shape[0], dtype=np.float64)
    for edge in getattr(graph, "views", {}).values():
        edge_index = np.asarray(edge.edge_index, dtype=np.int64)
        if edge_index.size == 0:
            continue
        weights = np.asarray(edge.edge_weight, dtype=np.float64)
        src, dst = edge_index
        np.add.at(acc, dst, soft_labels[src] * weights[:, None])
        np.add.at(degree, dst, weights)
    no_neigh = degree <= EPS
    out = acc / np.maximum(degree[:, None], EPS)
    out[no_neigh] = soft_labels[no_neigh]
    return out


def _as_view_label_matrix(view_preds) -> np.ndarray:
    if isinstance(view_preds, dict):
        arrays = [_labels_from_view_pred(value) for value in view_preds.values()]
        return np.vstack(arrays).astype(np.int64)
    arr = np.asarray(view_preds)
    if arr.ndim == 1:
        return arr[None, :].astype(np.int64)
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim == 3:
        return np.argmax(arr, axis=2).astype(np.int64)
    raise ValueError("view_preds must be [N], [M,N], [M,N,K], or dict(view -> predictions).")


def _labels_from_view_pred(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        return arr.astype(np.int64)
    if arr.ndim == 2:
        return np.argmax(arr, axis=1).astype(np.int64)
    raise ValueError("Each view prediction must be labels [N] or probabilities/logits [N,K].")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPS, None)
    return values / np.maximum(values.sum(axis=1, keepdims=True), EPS)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = values.max(initial=0.0) - values.min(initial=0.0)
    if span <= EPS:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _section(cfg: dict, name: str) -> dict:
    if not isinstance(cfg, dict):
        return {}
    return cfg.get(name, cfg)
