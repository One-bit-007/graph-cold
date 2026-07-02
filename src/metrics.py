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


def macro_f1(y_true, y_pred):
    raise NotImplementedError("TODO(Codex)")


def false_positive_rate(y_true, y_pred, benign_class):
    raise NotImplementedError("TODO(Codex)")


def false_negative_rate(y_true, y_pred, benign_class):
    raise NotImplementedError("TODO(Codex)")


def evidence_retention_rate(keep_or_weight, clean_mask, y, cfg):
    raise NotImplementedError("TODO(Codex): core new metric (ERR).")


def noise_detection_prf(flip_mask, pred_noisy_mask):
    raise NotImplementedError("TODO(Codex)")
