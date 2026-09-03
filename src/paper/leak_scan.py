"""Feature/target association scanner; association is NOT proof of leakage.

For low-cardinality columns MI is discrete/exact on the audit sample. Other
columns use at most 32 quantile bins to avoid unique-value memorization.
Report raw MI in nats AND target-normalized MI = I(X;Y)/H(Y). A deterministic
binary target projection has normalized MI 1, not necessarily raw MI 1 nat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def _entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _discretize(x, bins):
    values, inverse = np.unique(x, return_inverse=True)
    if len(values) <= bins:
        return inverse, "exact_discrete"
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins+1)))
    return np.searchsorted(edges[1:-1], x, side="right"), "quantile_binned"


def _association(x, discrete, y):
    entropy = _entropy(y)
    if entropy <= 0:
        return {"mi_nats": 0., "normalized_mi": None, "max_abs_ovr_correlation": None}
    mi = float(mutual_info_score(discrete, y))
    centered = x-x.mean()
    norm = float(np.linalg.norm(centered))
    correlations = []
    for label in np.unique(y):
        indicator = (y == label).astype(float)
        indicator -= indicator.mean()
        denom = norm * np.linalg.norm(indicator)
        correlations.append(abs(float(centered @ indicator / denom)) if denom > 0 else 0.)
    return {"mi_nats": mi, "normalized_mi": float(np.clip(mi/entropy, 0, 1)),
            "max_abs_ovr_correlation": float(max(correlations))}


def scan_features(X, y, feature_names, *, benign_class=None, max_rows=20000,
                  bins=32, mi_threshold=0.95, correlation_threshold=0.995, seed=0):
    """Return every feature, no silent removal. CESNET has no binary IDS target."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or X.shape != (len(y), len(feature_names)) or not len(y):
        raise ValueError("X, labels and feature names must align")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite feature values must be investigated before MI")
    if max_rows and len(y) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(len(y), max_rows, replace=False))
        X, y = X[idx], y[idx]
    rows = []
    for j, name in enumerate(feature_names):
        x = X[:, j]
        discrete, mode = _discretize(x, bins)
        multi = _association(x, discrete, y)
        binary = _association(x, discrete, (y != benign_class).astype(int)) if benign_class is not None else None
        checks = [multi] + ([binary] if binary is not None else [])
        flagged = any((m["normalized_mi"] is not None and m["normalized_mi"] >= mi_threshold)
                      or (m["max_abs_ovr_correlation"] is not None and m["max_abs_ovr_correlation"] >= correlation_threshold)
                      for m in checks)
        rows.append({"feature": str(name), "n_audit_rows": len(y), "n_unique": len(np.unique(x)),
                     "discretization": mode, "multiclass_mi_nats": multi["mi_nats"],
                     "multiclass_normalized_mi": multi["normalized_mi"],
                     "multiclass_max_abs_ovr_correlation": multi["max_abs_ovr_correlation"],
                     "binary_mi_nats": binary["mi_nats"] if binary else None,
                     "binary_normalized_mi": binary["normalized_mi"] if binary else None,
                     "binary_abs_correlation": binary["max_abs_ovr_correlation"] if binary else None,
                     "flagged": bool(flagged)})
    return pd.DataFrame(rows)
