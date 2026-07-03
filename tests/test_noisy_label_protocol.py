import numpy as np

from src.data.noise import inject_symmetric


def test_symmetric_noise_is_reused_by_all_methods_and_keeps_test_clean():
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_test = np.array([0, 1, 2, 1])
    clean_test = y_test.copy()

    y_noisy, flip = inject_symmetric(y_train, 0.3, 3, np.random.default_rng(42))
    graph_y_noisy = y_noisy.copy()
    cold_y_noisy = y_noisy.copy()
    graph_flip = flip.copy()
    cold_flip = flip.copy()

    np.testing.assert_array_equal(graph_y_noisy, cold_y_noisy)
    np.testing.assert_array_equal(graph_flip, cold_flip)
    np.testing.assert_array_equal(y_test, clean_test)
    assert flip.sum() == 3
    assert np.all(y_noisy[flip] != y_train[flip])


def test_noise_reproducibility_with_same_seed():
    y_train = np.arange(30) % 4

    y_a, flip_a = inject_symmetric(y_train, 0.2, 4, np.random.default_rng(7))
    y_b, flip_b = inject_symmetric(y_train, 0.2, 4, np.random.default_rng(7))

    np.testing.assert_array_equal(y_a, y_b)
    np.testing.assert_array_equal(flip_a, flip_b)
