import numpy as np

from src.baselines.base import array_hash
from src.baselines.confident_learning import ConfidentLearningBaseline


def _toy():
    rng = np.random.default_rng(11)
    parts = []
    labels = []
    for label, loc in enumerate((-3.0, 0.0, 3.0)):
        parts.append(rng.normal(loc=loc, scale=0.35, size=(40, 5)))
        labels.extend([label] * 40)
    X = np.vstack(parts).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    y_noisy = y.copy()
    y_noisy[0:12] = 1
    y_noisy[45:57] = 2
    return X, y, y_noisy


def test_cl_filtering_uses_noisy_labels_and_returns_valid_mask():
    X, y_clean, y_noisy = _toy()
    baseline = ConfidentLearningBaseline(seed=42, noise_rate=0.2, n_estimators=8)

    result = baseline.fit_predict(X, y_noisy, X, num_classes=3)

    assert result.method in {"Confident-Learning", "CL-filtering"}
    assert result.details["training_label_hash"] == array_hash(y_noisy)
    assert result.details["training_label_hash"] != array_hash(y_clean)
    assert result.retained_mask.shape == y_noisy.shape
    assert 0.0 < result.retained_mask.mean() <= 1.0
    assert result.proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(result.proba.sum(axis=1), 1.0)


def test_cl_filtering_is_deterministic_for_fixed_seed():
    X, _, y_noisy = _toy()

    a = ConfidentLearningBaseline(seed=5, noise_rate=0.2, n_estimators=8).fit_predict(X, y_noisy, X, num_classes=3)
    b = ConfidentLearningBaseline(seed=5, noise_rate=0.2, n_estimators=8).fit_predict(X, y_noisy, X, num_classes=3)

    np.testing.assert_array_equal(a.y_pred, b.y_pred)
    np.testing.assert_array_equal(a.retained_mask, b.retained_mask)
