"""Guard tests for P7 (P1-2 + P1-3): F-5 compliant noise-rate estimation.

These tests pin three properties:
1. estimate_noise_rate can only consume (X, y_noisy, seed) - no clean labels and
   no true injected rate can flow into it (F-5 by signature).
2. FINE and Co-Teaching constructed with noise_rate=None must record
   noise_rate_source="estimated_from_noisy_train" and a non-trivial estimated
   target on noisy data.
3. The formal P7 grid artifact, when present, is complete (180 unique
   scenario-method rows), F-5 compliant, and free of missing metrics.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.baselines.coteaching_full import CoTeachingFullBaseline
from src.baselines.fine_style import FINEBaseline
from src.baselines.noise_estimate import estimate_noise_rate

RESULTS = Path("results/p7_baseline_expansion.csv")
GRID_KEYS = ["dataset", "tail_flip_rate", "seed", "method"]


def _synthetic_noisy(n: int = 900, n_features: int = 12, n_classes: int = 4,
                     flip: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 4.0, size=(n_classes, n_features))
    y = np.repeat(np.arange(n_classes), n // n_classes)
    X = centers[y] + rng.normal(0.0, 1.0, size=(n, n_features))
    y_noisy = y.copy()
    k = int(flip * n)
    idx = rng.choice(n, size=k, replace=False)
    y_noisy[idx] = (y_noisy[idx] + 1) % n_classes
    return X.astype(np.float32), y.astype(np.int64), y_noisy.astype(np.int64)


def test_estimator_signature_cannot_receive_oracle_info():
    params = set(inspect.signature(estimate_noise_rate).parameters)
    assert params <= {"X_train", "y_noisy", "seed", "n_estimators"}
    forbidden = {"y_clean", "clean_y", "true_rate", "injected_rate", "noise_rate"}
    assert forbidden.isdisjoint(params)


def test_estimator_recovers_synthetic_noise_and_is_deterministic():
    X, _, y_noisy = _synthetic_noisy(flip=0.3)
    est1 = estimate_noise_rate(X, y_noisy, seed=0, n_estimators=8)
    est2 = estimate_noise_rate(X, y_noisy, seed=0, n_estimators=8)
    assert est1 == pytest.approx(est2)
    # loose bounds: separable clusters with 30% flips must be clearly > 0
    assert 0.05 < est1 <= 0.95
    clean_est = estimate_noise_rate(X, _synthetic_noisy(flip=0.0)[1], seed=0, n_estimators=8)
    assert est1 > clean_est


def test_estimator_degenerates_safely_on_tiny_class():
    X = np.zeros((6, 3), dtype=np.float32)
    y = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    est = estimate_noise_rate(X, y, seed=0, n_estimators=4)
    assert 0.0 <= est <= 0.95


def test_coteaching_none_estimates_and_records_source():
    X, _, y_noisy = _synthetic_noisy(n=600, flip=0.25)
    baseline = CoTeachingFullBaseline(seed=0, noise_rate=None, epochs=2, batch_size=128)
    result = baseline.fit_predict(X, y_noisy, X, 4)
    assert result.details["noise_rate_source"] == "estimated_from_noisy_train"
    target = result.details["forget_rate_target"]
    expected = estimate_noise_rate(X, y_noisy, seed=0)
    assert target == pytest.approx(min(expected, 0.5))
    assert target > 0.0


def test_fine_none_estimates_and_records_source():
    X, _, y_noisy = _synthetic_noisy(n=600, flip=0.25)
    baseline = FINEBaseline(seed=0, noise_rate=None, classifier_epochs=2, n_estimators=4)
    result = baseline.fit_predict(X, y_noisy, X, 4)
    assert result.details["noise_rate_source"] == "estimated_from_noisy_train"
    est = result.details["estimated_noise_rate"]
    assert est > 0.0
    assert est == pytest.approx(estimate_noise_rate(X, y_noisy, seed=0))


def test_runner_never_passes_true_rate_to_baselines():
    source = Path("src/paper/p7_baseline_expansion.py").read_text(encoding="utf-8")
    assert "noise_rate=None" in source
    assert "noise_rate=rate" not in source
    assert "noise_rate=flip" not in source


@pytest.mark.skipif(not RESULTS.exists(), reason="P7 grid artifact not generated on this machine")
def test_formal_grid_completeness_and_f5_compliance():
    import pandas as pd

    df = __import__("pandas").read_csv(RESULTS, keep_default_na=False)
    assert len(df) == 180
    assert not df.duplicated(subset=GRID_KEYS).any()
    expected = {
        (d, r, s, m)
        for d in ("cicids2017", "unsw_nb15", "cesnet_tls_year22")
        for r in (0.4, 0.6, 0.8)
        for s in range(10)
        for m in ("FINE", "Co-Teaching")
    }
    got = {(str(a), float(b), int(c), str(d)) for a, b, c, d in
           zip(df["dataset"], df["tail_flip_rate"], df["seed"], df["method"])}
    assert got == expected
    assert set(df["noise_rate_source"].unique()) == {"estimated_from_noisy_train"}
    assert df[["macro_f1", "tail_macro_f1", "fpr", "fnr"]].notna().all().all()
