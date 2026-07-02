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
    threshold = float(cfg_ev.get("retention_threshold", 0.5))
    retained = values.astype(bool) if values.dtype == bool else values >= threshold
    labels, counts = np.unique(y, return_counts=True)
    if labels.size == 0:
        return 0.0
    rare_cut = np.median(counts)
    rare_labels = labels[counts <= rare_cut]
    informative = clean & np.isin(y, rare_labels)
    if informative.sum() == 0:
        informative = clean
    if informative.sum() == 0:
        return 0.0
    return float(np.mean(retained[informative]))


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
