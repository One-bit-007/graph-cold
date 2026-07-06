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
ranking_metrics(scores, y_true, cfg) -> dict
queue_load_curve(scores, y_true, budgets, benign_class=0) -> list[dict]
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


def ranking_metrics(scores, y_true, cfg=None):
    cfg_rank = cfg.get("ranking", cfg) if isinstance(cfg, dict) else {}
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true)
    benign_class = int(cfg_rank.get("benign_class", 0))
    k = int(cfg_rank.get("top_k", min(100, scores.shape[0])))
    budget = float(cfg_rank.get("review_budget", 0.1))
    recall_targets = tuple(cfg_rank.get("recall_targets", (0.90, 0.95)))
    malicious = y_true != benign_class
    n_malicious = int(malicious.sum())
    if scores.size == 0:
        base = {
            "topk_precision": 0.0,
            "topk_recall": 0.0,
            "precision_at_budget": 0.0,
            "alert_compression_at_full_recall": 0.0,
        }
        for target in recall_targets:
            base[f"compression_at_recall_{int(round(float(target) * 100))}"] = 0.0
        return base

    ranked = top_k(scores, scores.shape[0])
    top = ranked[: max(0, min(k, scores.shape[0]))]
    top_hits = int(malicious[top].sum()) if top.size else 0
    budget_k = max(1, int(np.ceil(np.clip(budget, 0.0, 1.0) * scores.shape[0])))
    budget_idx = ranked[:budget_k]
    budget_hits = int(malicious[budget_idx].sum())
    out = {
        "topk_precision": float(top_hits / top.size) if top.size else 0.0,
        "topk_recall": float(top_hits / n_malicious) if n_malicious else 0.0,
        "precision_at_budget": float(budget_hits / budget_idx.size) if budget_idx.size else 0.0,
        "alert_compression_at_full_recall": alert_compression_ratio(scores, y_true),
    }
    for target in recall_targets:
        label = f"compression_at_recall_{int(round(float(target) * 100))}"
        out[label] = compression_at_fixed_recall(scores, y_true, target, benign_class=benign_class)
    return out


def compression_at_fixed_recall(scores, y_true, recall_target: float, benign_class: int = 0) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true)
    if scores.size == 0:
        return 0.0
    malicious = y_true != benign_class
    total = int(malicious.sum())
    if total == 0:
        return 0.0
    target_hits = int(np.ceil(np.clip(float(recall_target), 0.0, 1.0) * total))
    ranked = top_k(scores, scores.shape[0])
    cumulative = np.cumsum(malicious[ranked].astype(np.int64))
    positions = np.flatnonzero(cumulative >= target_hits)
    inspected = int(positions[0]) + 1 if positions.size else scores.shape[0]
    return float(inspected / scores.shape[0])


def queue_load_curve(scores, y_true, budgets=None, benign_class: int = 0):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true)
    if budgets is None:
        budgets = np.linspace(0.01, 1.0, 100)
    budgets = np.asarray(budgets, dtype=np.float64)
    malicious = y_true != benign_class
    total = int(malicious.sum())
    if scores.size == 0:
        return [{"review_budget": float(budget), "topk_recall": 0.0, "precision": 0.0} for budget in budgets]
    ranked = top_k(scores, scores.shape[0])
    out = []
    for budget in budgets:
        k = max(1, int(np.ceil(np.clip(float(budget), 0.0, 1.0) * scores.shape[0])))
        idx = ranked[:k]
        hits = int(malicious[idx].sum())
        out.append(
            {
                "review_budget": float(budget),
                "topk_recall": float(hits / total) if total else 0.0,
                "precision": float(hits / idx.size) if idx.size else 0.0,
            }
        )
    return out


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
