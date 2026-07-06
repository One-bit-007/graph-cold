import numpy as np

from src.baselines.base import array_hash
from src.baselines.mcre import MCReBaseline
from src.baselines.morse import MORSEBaseline


def _toy():
    rng = np.random.default_rng(314)
    parts = []
    labels = []
    for label, loc in enumerate((-2.0, 0.0, 2.0)):
        parts.append(rng.normal(loc=loc, scale=0.35, size=(45, 7)))
        labels.extend([label] * 45)
    X = np.vstack(parts).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    y_noisy = y.copy()
    y_noisy[:12] = 1
    y_noisy[50:62] = 2
    return X, y, y_noisy


def test_mcre_uses_noisy_labels_and_returns_baseline_result():
    X, y_clean, y_noisy = _toy()

    result = MCReBaseline(seed=4, noise_rate=0.2, n_components=4).fit_predict(X, y_noisy, X, num_classes=3)

    assert result.method == "MCRe"
    assert result.implementation_status == "verified_implementation"
    assert result.details["training_label_hash"] == array_hash(y_noisy)
    assert result.details["training_label_hash"] != array_hash(y_clean)
    assert result.retained_mask.shape == y_noisy.shape
    assert 0.0 < result.retained_mask.mean() <= 1.0
    assert result.proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(result.proba.sum(axis=1), 1.0, atol=1e-6)


def test_morse_is_deterministic_and_records_unlabeled_split():
    X, _, y_noisy = _toy()

    a = MORSEBaseline(seed=8, noise_rate=0.25, n_components=4).fit_predict(X, y_noisy, X, num_classes=3)
    b = MORSEBaseline(seed=8, noise_rate=0.25, n_components=4).fit_predict(X, y_noisy, X, num_classes=3)

    assert a.method == "MORSE"
    assert a.details["split_rule"] == "actual_noise_rate_as_unlabeled_split_ratio"
    np.testing.assert_array_equal(a.retained_mask, b.retained_mask)
    np.testing.assert_array_equal(a.y_pred, b.y_pred)
