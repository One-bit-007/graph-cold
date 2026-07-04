"""Shared baseline helpers for real-data label-noise experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np


@dataclass
class BaselineResult:
    method: str
    method_family: str
    implementation_status: str
    y_pred: np.ndarray
    proba: np.ndarray
    weights: np.ndarray
    retained_mask: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def aligned_proba(model, X, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    for col_idx, label in enumerate(getattr(model, "classes_", np.arange(raw.shape[1]))):
        label_idx = int(label)
        if 0 <= label_idx < num_classes:
            out[:, label_idx] = raw[:, col_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 1e-12
    if np.any(zero):
        out[zero, :] = 1.0 / max(num_classes, 1)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    labels, counts = np.unique(y, return_counts=True)
    weights = {int(label): y.shape[0] / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[int(label)] for label in y], dtype=np.float64)


def ensure_class_coverage(retained: np.ndarray, confidence: np.ndarray, y: np.ndarray) -> np.ndarray:
    retained = np.asarray(retained, dtype=bool).copy()
    confidence = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        if members.size and not retained[members].any():
            retained[members[np.argmax(confidence[members])]] = True
    if np.unique(y[retained]).size < 2:
        retained[:] = True
    return retained
