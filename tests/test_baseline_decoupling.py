import numpy as np

from src.baselines.base import array_hash
from src.baselines.decoupling import DecouplingBaseline


def _toy():
    rng = np.random.default_rng(123)
    parts = []
    labels = []
    for label, loc in enumerate((-2.0, 0.0, 2.0)):
        parts.append(rng.normal(loc=loc, scale=0.45, size=(45, 6)))
        labels.extend([label] * 45)
    X = np.vstack(parts).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    y_noisy = y.copy()
    y_noisy[:12] = 1
    y_noisy[52:64] = 2
    return X, y, y_noisy


def test_decoupling_uses_noisy_y_train_and_records_disagreement_update():
    X, y_clean, y_noisy = _toy()
    result = DecouplingBaseline(seed=42, epochs=4, batch_size=24).fit_predict(X, y_noisy, X, num_classes=3)

    assert result.method == "Decoupling"
    assert result.details["training_label_hash"] == array_hash(y_noisy)
    assert result.details["training_label_hash"] != array_hash(y_clean)
    assert result.details["train_label_source"] == "noisy_y_train"
    assert result.details["eval_label_source"] == "clean_y_test"
    assert "disagreement_fraction" in result.details
    assert "update_fraction" in result.details
    assert result.retained_mask.dtype == bool
    assert result.retained_mask.shape == y_noisy.shape
    np.testing.assert_allclose(result.proba.sum(axis=1), 1.0, atol=1e-6)


def test_decoupling_is_deterministic_for_fixed_seed():
    X, _, y_noisy = _toy()

    a = DecouplingBaseline(seed=7, epochs=4, batch_size=30).fit_predict(X, y_noisy, X, num_classes=3)
    b = DecouplingBaseline(seed=7, epochs=4, batch_size=30).fit_predict(X, y_noisy, X, num_classes=3)

    np.testing.assert_array_equal(a.y_pred, b.y_pred)
    np.testing.assert_array_equal(a.retained_mask, b.retained_mask)
    np.testing.assert_allclose(a.proba, b.proba)
