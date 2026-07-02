import numpy as np

from src.data.noise import inject_asymmetric, inject_symmetric


def test_symmetric_noise_is_seed_reproducible_and_ratio_is_exact_floor():
    y = np.arange(100) % 5

    noisy_a, mask_a = inject_symmetric(y, ratio=0.2, num_classes=5, rng=np.random.default_rng(7))
    noisy_b, mask_b = inject_symmetric(y, ratio=0.2, num_classes=5, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(noisy_a, noisy_b)
    np.testing.assert_array_equal(mask_a, mask_b)
    assert mask_a.sum() == 20
    assert np.all(noisy_a[mask_a] != y[mask_a])
    np.testing.assert_array_equal(noisy_a[~mask_a], y[~mask_a])


def test_asymmetric_noise_flips_only_malicious_to_benign():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)

    noisy, mask = inject_asymmetric(y, ratio=0.3, benign_class=0, rng=np.random.default_rng(3))

    assert mask.sum() == 9
    assert np.all(y[mask] != 0)
    assert np.all(noisy[mask] == 0)
    np.testing.assert_array_equal(noisy[~mask], y[~mask])


def test_noise_functions_do_not_touch_test_labels_when_only_train_is_injected():
    y_train = np.arange(50) % 4
    y_test = np.arange(20) % 4
    y_test_before = y_test.copy()

    inject_symmetric(y_train, ratio=0.4, num_classes=4, rng=np.random.default_rng(11))

    np.testing.assert_array_equal(y_test, y_test_before)
