from types import SimpleNamespace

import numpy as np

from src.data.noise import inject_graph_consistency, inject_symmetric


def _graph(edge_index, edge_weight=None):
    edge = SimpleNamespace(
        edge_index=np.asarray(edge_index, dtype=np.int64),
        edge_weight=np.ones(np.asarray(edge_index).shape[1], dtype=np.float32)
        if edge_weight is None
        else np.asarray(edge_weight, dtype=np.float32),
    )
    return SimpleNamespace(views={"host": edge})


def test_beta_zero_degenerates_exactly_to_symmetric_same_seed():
    y = np.arange(120) % 4
    cfg = {"num_classes": 4, "graph_consistency": {"consistency_bias": 0.0}}

    noisy_graph, mask_graph = inject_graph_consistency(y, 0.25, _graph([[0, 1], [1, 0]]), cfg, np.random.default_rng(42))
    noisy_sym, mask_sym = inject_symmetric(y, 0.25, 4, np.random.default_rng(42))

    np.testing.assert_array_equal(mask_graph, mask_sym)
    np.testing.assert_array_equal(noisy_graph, noisy_sym)
    assert mask_graph.sum() == int(np.floor(0.25 * len(y)))


def test_beta_zero_transition_statistics_are_symmetric_like():
    y = np.arange(600) % 3
    cfg = {"num_classes": 3, "consistency_bias": 0.0}

    noisy, mask = inject_graph_consistency(y, 0.3, _graph([[0, 1], [1, 0]]), cfg, np.random.default_rng(7))
    transitions = np.zeros((3, 3), dtype=int)
    for src, dst in zip(y[mask], noisy[mask]):
        transitions[src, dst] += 1

    assert mask.sum() == 180
    assert np.all(np.diag(transitions) == 0)
    off_diag = transitions[transitions > 0]
    assert off_diag.max() - off_diag.min() < 25


def test_beta_one_prefers_graph_consistent_majority_targets():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    edge_index = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 3, 3, 0, 0, 3, 3, 3],
        ]
    )
    cfg = {"num_classes": 3, "graph_consistency": {"consistency_bias": 1.0}}

    noisy, mask = inject_graph_consistency(y, 0.5, _graph(edge_index), cfg, np.random.default_rng(42))

    assert mask.sum() == 4
    graph_target_flips = np.sum((np.flatnonzero(mask) != 3) & (noisy[mask] == 1))
    assert graph_target_flips >= 3


def test_graph_consistency_noise_supports_multiclass_and_seed_reproducibility():
    y = np.arange(50) % 5
    edge_index = np.vstack([np.arange(49), np.arange(1, 50)])
    cfg = {"num_classes": 5, "graph_consistency": {"consistency_bias": 0.7}}

    noisy_a, mask_a = inject_graph_consistency(y, 0.4, _graph(edge_index), cfg, np.random.default_rng(42))
    noisy_b, mask_b = inject_graph_consistency(y, 0.4, _graph(edge_index), cfg, np.random.default_rng(42))
    noisy_c, mask_c = inject_graph_consistency(y, 0.4, _graph(edge_index), cfg, np.random.default_rng(43))

    np.testing.assert_array_equal(noisy_a, noisy_b)
    np.testing.assert_array_equal(mask_a, mask_b)
    assert mask_a.sum() == 20
    assert np.all(noisy_a[mask_a] != y[mask_a])
    assert not (np.array_equal(noisy_a, noisy_c) and np.array_equal(mask_a, mask_c))


def test_empty_graph_and_isolated_nodes_fall_back_to_random_flips():
    y = np.arange(30) % 3
    empty_graph = SimpleNamespace(views={"host": SimpleNamespace(edge_index=np.zeros((2, 0), dtype=int), edge_weight=np.zeros(0))})
    cfg = {"num_classes": 3, "graph_consistency": {"consistency_bias": 1.0}}

    noisy, mask = inject_graph_consistency(y, 0.2, empty_graph, cfg, np.random.default_rng(42))

    assert mask.sum() == 6
    assert np.all(noisy[mask] != y[mask])
