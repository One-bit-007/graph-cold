"""Label-noise injection.

Injects noise ONLY into training labels. The test set stays clean.
Three schemes:

1. symmetric   : uniform corruption across all classes.
2. asymmetric  : corruption confined to malicious classes (-> benign).
3. graph_consistency (NEW): flips preferentially along locally-consistent edges.
   See docs/spec_graph_noise.md for the formal definition.

Contract
--------
inject_symmetric(y, ratio, num_classes, rng) -> y_noisy, flip_mask
inject_asymmetric(y, ratio, benign_class, rng) -> y_noisy, flip_mask
inject_graph_consistency(y, ratio, graph, cfg, rng) -> y_noisy, flip_mask

`flip_mask` (bool [N]) marks corrupted positions; used to compute noise-detection
metrics and the Evidence Retention Rate.
"""
from __future__ import annotations

import numpy as np


def inject_symmetric(y, ratio, num_classes, rng):
    rng = _as_generator(rng)
    y = np.asarray(y)
    y_noisy = y.copy()
    flip_mask = np.zeros(y.shape[0], dtype=bool)
    if ratio <= 0 or y.shape[0] == 0:
        return y_noisy, flip_mask
    if num_classes < 2:
        raise ValueError("Symmetric noise requires at least two classes.")

    n_flips = _num_flips(y.shape[0], ratio)
    if n_flips == 0:
        return y_noisy, flip_mask

    flip_idx = rng.choice(y.shape[0], size=n_flips, replace=False)
    offsets = rng.integers(1, num_classes, size=n_flips)
    y_noisy[flip_idx] = (y_noisy[flip_idx] + offsets) % num_classes
    flip_mask[flip_idx] = True
    return y_noisy, flip_mask


def inject_asymmetric(y, ratio, benign_class, rng):
    rng = _as_generator(rng)
    y = np.asarray(y)
    y_noisy = y.copy()
    flip_mask = np.zeros(y.shape[0], dtype=bool)
    if ratio <= 0 or y.shape[0] == 0:
        return y_noisy, flip_mask

    candidates = np.flatnonzero(y != benign_class)
    if candidates.size == 0:
        return y_noisy, flip_mask

    n_flips = min(_num_flips(y.shape[0], ratio), candidates.size)
    if n_flips == 0:
        return y_noisy, flip_mask

    flip_idx = rng.choice(candidates, size=n_flips, replace=False)
    y_noisy[flip_idx] = benign_class
    flip_mask[flip_idx] = True
    return y_noisy, flip_mask


def inject_graph_consistency(y, ratio, graph, cfg, rng):
    """Flip labels biased toward high graph-consistency edges.

    See docs/spec_graph_noise.md — must expose `consistency_bias` so that a
    fraction of flips land on locally-consistent edges and the rest are random.
    """
    raise NotImplementedError("TODO(D4): implement graph-consistency label noise.")


def _as_generator(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _num_flips(n_samples: int, ratio: float) -> int:
    ratio = float(ratio)
    if ratio < 0 or ratio > 1:
        raise ValueError("Noise ratio must be in [0, 1].")
    return int(np.floor(ratio * n_samples))
