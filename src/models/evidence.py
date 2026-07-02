"""Evidence preservation scores for Stage-2 Graph-CoLD."""
from __future__ import annotations

import numpy as np


EPS = 1e-12


def compute(y, cfg, anomaly=None, recon_error=None, entropy=None, class_counts=None):
    """Compute min-max normalized evidence scores e(v) in [0, 1].

    `freq_protect="log"` uses 1/log(1+n_y) to protect low-frequency classes.
    `freq_protect="inverse"` uses 1/n_y.
    """
    y = np.asarray(y, dtype=np.int64)
    cfg_ev = cfg.get("evidence_preserving", cfg) if isinstance(cfg, dict) else {}
    gamma = float(cfg_ev.get("gamma_anomaly", 1.0))
    freq_mode = str(cfg_ev.get("freq_protect", "log")).lower()
    counts = _class_counts(y, class_counts)
    class_count = np.asarray([counts[int(label)] for label in y], dtype=np.float64)

    if freq_mode in {"inverse", "1/n"}:
        freq = 1.0 / np.maximum(class_count, 1.0)
    elif freq_mode in {"log", "log_inverse", "1/log"}:
        freq = 1.0 / np.maximum(np.log1p(class_count), EPS)
    else:
        raise ValueError("freq_protect must be one of {'log', 'inverse'}.")

    anom = _anomaly(anomaly=anomaly, recon_error=recon_error, entropy=entropy)
    raw = freq * (1.0 + gamma * anom)
    return _minmax(raw)


def _class_counts(y: np.ndarray, class_counts=None) -> dict[int, int]:
    if class_counts is not None:
        if isinstance(class_counts, dict):
            return {int(k): int(v) for k, v in class_counts.items()}
        counts_arr = np.asarray(class_counts)
        return {idx: int(count) for idx, count in enumerate(counts_arr)}
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _anomaly(anomaly=None, recon_error=None, entropy=None) -> np.ndarray:
    source = anomaly
    if source is None:
        source = recon_error
    if source is None:
        source = entropy
    if source is None:
        raise ValueError("Evidence score requires anomaly, recon_error, or entropy.")
    return _minmax(np.asarray(source, dtype=np.float64))


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    span = values.max() - values.min()
    if span <= EPS:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.min()) / span
