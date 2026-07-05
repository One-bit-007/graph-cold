import numpy as np

from src.baselines.base import array_hash
from src.baselines.fine_style import FINEStyleBaseline


def _toy():
    rng = np.random.default_rng(456)
    parts = []
    labels = []
    for label, loc in enumerate((-3.0, -1.0, 1.0, 3.0)):
        parts.append(rng.normal(loc=loc, scale=0.35, size=(40, 8)))
        labels.extend([label] * 40)
    X = np.vstack(parts).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    y_noisy = y.copy()
    y_noisy[:10] = 1
    y_noisy[45:55] = 2
    return X, y, y_noisy


def test_fine_style_name_metadata_and_retained_mask_are_honest():
    X, y_clean, y_noisy = _toy()
    result = FINEStyleBaseline(seed=42, noise_rate=0.2, n_components=4, classifier_epochs=3).fit_predict(
        X, y_noisy, X, num_classes=4
    )

    assert result.method == "FINE-style"
    assert result.method != "FINE"
    assert result.details["training_label_hash"] == array_hash(y_noisy)
    assert result.details["training_label_hash"] != array_hash(y_clean)
    assert result.details["representation_source"] == "standardized_features_pca"
    assert "not full original implementation" in result.details["faithfulness_level"]
    assert result.retained_mask.shape == y_noisy.shape
    assert 0.0 < result.retained_mask.mean() <= 1.0
    assert result.proba.shape == (X.shape[0], 4)
    np.testing.assert_allclose(result.proba.sum(axis=1), 1.0, atol=1e-6)


def test_fine_style_retains_all_when_noise_rate_zero():
    X, _, y_noisy = _toy()
    result = FINEStyleBaseline(seed=5, noise_rate=0.0, n_components=4, classifier_epochs=2).fit_predict(
        X, y_noisy, X, num_classes=4
    )

    assert result.retained_mask.all()
    assert result.details["retain_fraction"] == 1.0


def test_fine_style_is_deterministic_for_fixed_seed():
    X, _, y_noisy = _toy()

    a = FINEStyleBaseline(seed=9, noise_rate=0.3, n_components=4, classifier_epochs=3).fit_predict(X, y_noisy, X, 4)
    b = FINEStyleBaseline(seed=9, noise_rate=0.3, n_components=4, classifier_epochs=3).fit_predict(X, y_noisy, X, 4)

    np.testing.assert_array_equal(a.retained_mask, b.retained_mask)
    np.testing.assert_array_equal(a.y_pred, b.y_pred)
