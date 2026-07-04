import numpy as np

from src.baselines.base import array_hash
from src.baselines.coteaching import CoTeachingLiteBaseline


def _toy():
    rng = np.random.default_rng(19)
    X0 = rng.normal(loc=-2.5, scale=0.4, size=(50, 6))
    X1 = rng.normal(loc=0.0, scale=0.4, size=(50, 6))
    X2 = rng.normal(loc=2.5, scale=0.4, size=(50, 6))
    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int64)
    y_noisy = y.copy()
    y_noisy[:15] = 1
    y_noisy[50:65] = 2
    return X, y, y_noisy


def test_coteaching_lite_uses_two_models_and_noisy_y_train():
    X, y_clean, y_noisy = _toy()
    baseline = CoTeachingLiteBaseline(seed=42, noise_rate=0.2, epochs=3)

    result = baseline.fit_predict(X, y_noisy, X, num_classes=3)

    assert result.method == "Co-Teaching-lite"
    assert result.details["small_loss_exchange"] is True
    assert result.details["training_label_hash"] == array_hash(y_noisy)
    assert result.details["training_label_hash"] != array_hash(y_clean)
    assert result.retained_mask.shape == y_noisy.shape
    assert 0.0 < result.retained_mask.mean() <= 1.0
    assert result.proba.shape == (X.shape[0], 3)


def test_coteaching_lite_is_deterministic_for_fixed_seed():
    X, _, y_noisy = _toy()

    a = CoTeachingLiteBaseline(seed=9, noise_rate=0.2, epochs=3).fit_predict(X, y_noisy, X, num_classes=3)
    b = CoTeachingLiteBaseline(seed=9, noise_rate=0.2, epochs=3).fit_predict(X, y_noisy, X, num_classes=3)

    np.testing.assert_array_equal(a.y_pred, b.y_pred)
    np.testing.assert_array_equal(a.retained_mask, b.retained_mask)
