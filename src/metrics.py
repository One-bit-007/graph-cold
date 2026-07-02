"""Evaluation metrics.

Standard + Graph-CoLD-specific metrics.

Standard
--------
macro_f1(y_true, y_pred) -> float
false_positive_rate(y_true, y_pred, benign_class) -> float
false_negative_rate(y_true, y_pred, benign_class) -> float

Graph-CoLD specific
-------------------
evidence_retention_rate(keep_or_weight, clean_mask, y, cfg) -> float
    Fraction of *clean* informative samples (esp. low-frequency / boundary /
    early-APT) preserved after denoising. For CoLD (hard deletion) this uses the
    keep_mask; for Graph-CoLD (soft weights) a sample counts as retained if its
    weight exceeds a retention threshold. This is the metric that quantifies the
    "evidence preserved, not discarded" claim.

alert_compression_ratio(scores, y_true) -> float   # re-exported from ranking

noise_detection_prf(flip_mask, pred_noisy_mask) -> (P, R, F1)
    How well the method identifies the injected noisy labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def false_positive_rate(y_true, y_pred, benign_class):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    benign = y_true == benign_class
    if benign.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[benign] != benign_class))


def false_negative_rate(y_true, y_pred, benign_class):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    malicious = y_true != benign_class
    if malicious.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[malicious] == benign_class))


def evidence_retention_rate(keep_or_weight, clean_mask, y, cfg):
    values = np.asarray(keep_or_weight, dtype=np.float64)
    clean = np.asarray(clean_mask, dtype=bool)
    y = np.asarray(y)
    cfg_ev = cfg.get("evidence_preserving", cfg) if isinstance(cfg, dict) else {}
    evidence = cfg_ev.get("evidence_scores")
    if evidence is None:
        evidence = _frequency_evidence(y, mode=str(cfg_ev.get("freq_protect", "log")))
    evidence = np.asarray(evidence, dtype=np.float64)
    if values.shape[0] != y.shape[0] or evidence.shape[0] != y.shape[0] or clean.shape[0] != y.shape[0]:
        raise ValueError("keep_or_weight, evidence_scores, clean_mask, and y must have the same length.")
    weights = np.clip(values, 0.0, 1.0)
    evidence = np.clip(evidence, 0.0, None)
    components = evidence_retention_components(weights, evidence, clean, y)
    return components["err_final"]


def evidence_retention_components(weights, evidence, clean_mask, y):
    weights = np.asarray(weights, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64)
    clean = np.asarray(clean_mask, dtype=bool)
    y = np.asarray(y)
    base_mask = clean if clean.any() else np.ones_like(clean, dtype=bool)
    err = _weighted_retention(weights, evidence, base_mask)

    labels, counts = np.unique(y[base_mask], return_counts=True)
    if labels.size == 0:
        tail_mask = base_mask
    else:
        tail_cut = np.median(counts)
        tail_labels = labels[counts <= tail_cut]
        tail_mask = base_mask & np.isin(y, tail_labels)
        if not tail_mask.any():
            tail_mask = base_mask
    err_tail = _weighted_retention(weights, evidence, tail_mask)
    err_final = float(np.clip(0.5 * (err + err_tail), 0.0, 1.0))
    return {
        "err": float(np.clip(err, 0.0, 1.0)),
        "err_tail": float(np.clip(err_tail, 0.0, 1.0)),
        "err_final": err_final,
        "tail_count": int(tail_mask.sum()),
    }


def noise_detection_prf(flip_mask, pred_noisy_mask):
    flip = np.asarray(flip_mask, dtype=bool)
    pred = np.asarray(pred_noisy_mask, dtype=bool)
    tp = float(np.sum(flip & pred))
    fp = float(np.sum(~flip & pred))
    fn = float(np.sum(flip & ~pred))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _weighted_retention(weights, evidence, mask):
    denom = float(np.sum(evidence[mask]))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(weights[mask] * evidence[mask]) / denom)


def _frequency_evidence(y, mode="log"):
    labels, counts = np.unique(y, return_counts=True)
    count_map = {label: count for label, count in zip(labels, counts)}
    class_counts = np.asarray([count_map[label] for label in y], dtype=np.float64)
    if mode in {"inverse", "1/n"}:
        raw = 1.0 / np.maximum(class_counts, 1.0)
    else:
        raw = 1.0 / np.maximum(np.log1p(class_counts), 1e-12)
    if raw.size == 0:
        return raw
    span = raw.max() - raw.min()
    if span <= 1e-12:
        return np.ones_like(raw, dtype=np.float64)
    return (raw - raw.min()) / span
