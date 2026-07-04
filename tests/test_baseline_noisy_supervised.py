import numpy as np
from sklearn.metrics import accuracy_score

from src.baselines.base import array_hash
from src.baselines.noisy_supervised import NoisySupervisedBaseline


def _toy():
    rng = np.random.default_rng(7)
    X0 = rng.normal(loc=-2.0, scale=0.2, size=(30, 4))
    X1 = rng.normal(loc=2.0, scale=0.2, size=(30, 4))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    y_noisy = y.copy()
    y_noisy[:10] = 1
    return X, y, y_noisy


def test_noisy_supervised_uses_noisy_y_train_and_retains_all_samples():
    X, y_clean, y_noisy = _toy()
    baseline = NoisySupervisedBaseline(seed=42, n_estimators=8)

    result = baseline.fit_predict(X, y_noisy, X, num_classes=2)

    assert result.details["training_label_hash"] == array_hash(y_noisy)
    assert result.details["training_label_hash"] != array_hash(y_clean)
    assert result.retained_mask.all()
    assert np.allclose(result.weights, 1.0)
    assert result.y_pred.shape == y_clean.shape
    assert accuracy_score(y_clean, result.y_pred) > 0.5


def test_noisy_supervised_is_deterministic_for_fixed_seed():
    X, _, y_noisy = _toy()

    a = NoisySupervisedBaseline(seed=3, n_estimators=8).fit_predict(X, y_noisy, X, num_classes=2)
    b = NoisySupervisedBaseline(seed=3, n_estimators=8).fit_predict(X, y_noisy, X, num_classes=2)

    np.testing.assert_array_equal(a.y_pred, b.y_pred)
    np.testing.assert_allclose(a.proba, b.proba)
