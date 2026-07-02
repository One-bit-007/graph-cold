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
    Evidence retention on clean informative samples. A sample is retained by the
    binary rule retained(v)=1[w(v) >= tau_ret], not by continuous w*e.

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
    flip_mask = cfg_ev.get("flip_mask")
    _validate_clean_mask(clean, y, flip_mask)
    evidence = cfg_ev.get("evidence_scores")
    if evidence is None:
        evidence = _frequency_evidence(y, mode=str(cfg_ev.get("freq_protect", "log")))
    evidence = np.asarray(evidence, dtype=np.float64)
    if values.shape[0] != y.shape[0] or evidence.shape[0] != y.shape[0] or clean.shape[0] != y.shape[0]:
        raise ValueError("keep_or_weight, evidence_scores, clean_mask, and y must have the same length.")
    weights = np.clip(values, 0.0, 1.0)
    evidence = np.clip(evidence, 0.0, None)
    tau_ret = float(cfg_ev.get("retention_threshold", 0.1))
    components = evidence_retention_components(weights, evidence, clean, y, retention_threshold=tau_ret)
    return components["err_final"]


def evidence_retention_components(weights, evidence, clean_mask, y, retention_threshold=0.1):
    weights = np.asarray(weights, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64)
    clean = np.asarray(clean_mask, dtype=bool)
    y = np.asarray(y)
    if weights.shape[0] != y.shape[0] or evidence.shape[0] != y.shape[0] or clean.shape[0] != y.shape[0]:
        raise ValueError("weights, evidence, clean_mask, and y must have the same length.")
    if not clean.any():
        return {"err": 0.0, "err_tail": 0.0, "err_final": 0.0, "tail_count": 0, "informative_count": 0}
    retained = np.clip(weights, 0.0, 1.0) >= float(retention_threshold)
    evidence = np.clip(evidence, 0.0, None)

    labels, counts = np.unique(y[clean], return_counts=True)
    if labels.size == 0:
        tail_mask = clean.copy()
    else:
        tail_cut = np.median(counts)
        tail_labels = labels[counts <= tail_cut]
        tail_mask = clean & np.isin(y, tail_labels)
        if not tail_mask.any():
            tail_mask = clean.copy()
    anomaly_cut = _safe_quantile(evidence[clean], 0.75)
    anomaly_mask = clean & (evidence >= anomaly_cut)
    informative_mask = clean & (tail_mask | anomaly_mask)
    if not informative_mask.any():
        informative_mask = clean.copy()
    err = _binary_retention(retained, evidence, informative_mask)
    err_tail = _binary_retention(retained, evidence, tail_mask)
    err_final = float(np.clip(0.5 * (err + err_tail), 0.0, 1.0))
    return {
        "err": float(np.clip(err, 0.0, 1.0)),
        "err_tail": float(np.clip(err_tail, 0.0, 1.0)),
        "err_final": err_final,
        "tail_count": int(tail_mask.sum()),
        "informative_count": int(informative_mask.sum()),
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


def _binary_retention(retained, evidence, mask):
    denom = float(np.sum(evidence[mask]))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(retained[mask].astype(np.float64) * evidence[mask]) / denom)


def _safe_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _validate_clean_mask(clean, y, flip_mask):
    if clean.shape[0] != y.shape[0]:
        raise ValueError("clean_mask and y must have the same length.")
    if flip_mask is None:
        if clean.all():
            raise ValueError("clean_mask must be derived from a real flip_mask; all-True masks are not allowed for ERR.")
        return
    flip = np.asarray(flip_mask, dtype=bool)
    if flip.shape[0] != y.shape[0]:
        raise ValueError("flip_mask and y must have the same length.")
    expected = ~flip
    if not np.array_equal(clean, expected):
        raise ValueError("clean_mask must be exactly the complement of flip_mask.")
    if not flip.any():
        raise ValueError("ERR requires an injected-noise flip_mask with at least one flipped sample.")


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
