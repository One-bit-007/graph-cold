"""F-5 compliant noise-rate estimation: only the noisy training set is used.

Both Co-Teaching (forget-rate schedule) and FINE (retain fraction) previously
accepted the *true injected* noise rate, which violates the frozen F-5 tuning
protocol (no clean validation set, no true injected rate). This helper estimates
the noise rate from ``(X_train, y_noisy)`` alone via 3-fold stratified
cross-fit out-of-fold probabilities: estimate = mean(1 - P(y_noisy)), i.e. the
mean self-doubt of a shared-budget ExtraTrees model on held-out folds.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold


def estimate_noise_rate(
    X_train: np.ndarray,
    y_noisy: np.ndarray,
    seed: int = 0,
    n_estimators: int = 24,
) -> float:
    """Estimate the label-noise rate using only the noisy training data.

    Returns mean(1 - P_out_of_fold(y_noisy)), clipped to [0.0, 0.95].
    No clean labels, no injected-rate input - F-5 compliant by construction.
    """
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_noisy, dtype=np.int64)
    n = y.shape[0]
    if n == 0 or np.unique(y).size < 2:
        return 0.0
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    if nonzero.size < 2:
        return 0.0
    n_splits = int(min(3, int(nonzero.min())))
    if n_splits < 2:
        # degenerate class sizes: fit on all data, use in-sample self-doubt
        model = ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=seed, class_weight="balanced", n_jobs=-1
        )
        model.fit(X, y)
        proba = _aligned_proba(model, X, int(np.max(y)) + 1)
        return float(np.clip(np.mean(1.0 - proba[np.arange(n), y]), 0.0, 0.95))

    proba = np.zeros((n, int(np.max(y)) + 1), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1901)
    for fold, (fit_idx, eval_idx) in enumerate(splitter.split(X, y)):
        model = ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=seed + 37 + fold, class_weight="balanced", n_jobs=-1
        )
        model.fit(X[fit_idx], y[fit_idx])
        proba[eval_idx] = _aligned_proba(model, X[eval_idx], proba.shape[1])
    estimate = float(np.mean(1.0 - proba[np.arange(n), y]))
    return float(np.clip(estimate, 0.0, 0.95))


def _aligned_proba(model: ExtraTreesClassifier, X: np.ndarray, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    for col, label in enumerate(getattr(model, "classes_", np.arange(raw.shape[1]))):
        label_idx = int(label)
        if 0 <= label_idx < num_classes:
            out[:, label_idx] = raw[:, col]
    return out
