"""SOC alert prioritization (Top-K).

Priority score combines risk, false-positive likelihood, and false-negative risk,
all derived from Graph-CDM components and node attributes:

    P(v) = risk(v) * (1 - FP_prob(v)) + eta * FN_risk(v)

Outputs a ranking and the Top-K highest-risk alerts for the analyst. Enables the
alert-compression-ratio metric (how few alerts an analyst must inspect to cover
the true attacks).

Contract
--------
priority_scores(cdm_components, node_attrs, cfg) -> np.ndarray [N]
top_k(scores, k) -> indices
alert_compression_ratio(scores, y_true) -> float
"""
from __future__ import annotations

import numpy as np


_MISSING = object()


def priority_scores(cdm_components, node_attrs, cfg):
    cfg_rank = cfg.get("ranking", cfg) if isinstance(cfg, dict) else {}
    alpha1 = float(cfg_rank.get("alpha_mal", cfg_rank.get("alpha1", 1.0)))
    alpha2 = float(cfg_rank.get("alpha_cdm", cfg_rank.get("alpha2", 1.0)))
    alpha3 = float(cfg_rank.get("alpha_evidence", cfg_rank.get("alpha3", 1.0)))

    y_mal = _get_value(cdm_components, node_attrs, "y_mal", default=None)
    if y_mal is None:
        soft = _get_value(cdm_components, node_attrs, "soft_labels", default=None)
        benign_class = int(cfg_rank.get("benign_class", 0))
        if soft is None:
            y_mal = np.zeros_like(_as_array(_get_value(cdm_components, node_attrs, "graph_cdm", "cdm")))
        else:
            soft = np.asarray(soft, dtype=np.float64)
            y_mal = 1.0 - soft[:, benign_class]
    cdm = _as_array(_get_value(cdm_components, node_attrs, "graph_cdm", "cdm"))
    evidence = _as_array(_get_value(cdm_components, node_attrs, "evidence"))
    return alpha1 * _as_array(y_mal) + alpha2 * cdm + alpha3 * evidence


def top_k(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    k = max(0, min(int(k), scores.shape[0]))
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    return order[:k].astype(np.int64)


def topk(scores, k):
    return top_k(scores, k)


def alert_compression_ratio(scores, y_true):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true)
    malicious = np.flatnonzero(y_true != 0)
    if malicious.size == 0:
        return 0.0
    ranked = top_k(scores, scores.shape[0])
    malicious_rank_positions = np.flatnonzero(np.isin(ranked, malicious))
    inspected = int(malicious_rank_positions.max()) + 1 if malicious_rank_positions.size else scores.shape[0]
    return float(inspected) / float(scores.shape[0])


def _get_value(primary, secondary, *keys, default=_MISSING):
    for container in (primary, secondary):
        if isinstance(container, dict):
            for key in keys:
                if key in container:
                    return container[key]
    if default is not _MISSING:
        return default
    raise KeyError(f"Missing required ranking field; tried {keys}.")


def _as_array(value):
    return np.asarray(value, dtype=np.float64)
