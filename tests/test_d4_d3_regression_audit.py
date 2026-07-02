from types import SimpleNamespace

import numpy as np

from src.graph.build import EdgeIndex
from src.metrics import evidence_retention_components, evidence_retention_rate
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence


def _cfg(rho=0.2, freq="log"):
    return {
        "graph_cdm": {
            "lambda_pred": 0.4,
            "lambda_neigh": 0.3,
            "lambda_view": 0.3,
            "lambda_chain": 0.0,
        },
        "evidence_preserving": {
            "kappa": 4.0,
            "theta": 0.5,
            "rho": rho,
            "gamma_anomaly": 1.0,
            "freq_protect": freq,
        },
    }


def test_d_pred_uses_observed_labels_not_fused_prediction():
    y = np.array([0, 1, 2, 1])
    view_labels = np.array([[0, 1, 2, 1], [0, 2, 2, 0], [1, 1, 0, 0]])
    fused_label_that_disagrees = np.array([1, 2, 0, 0])
    soft = np.eye(3)[fused_label_that_disagrees] * 0.8 + 0.2 / 3

    result = graph_cdm.forward(view_labels, soft, y, _cfg(), neigh_agg=soft, return_components=True)

    expected = np.mean(view_labels != y[None, :], axis=0)
    np.testing.assert_allclose(result.d_pred, expected)
    assert not np.array_equal(result.d_pred, (fused_label_that_disagrees != y).astype(float))


def test_d_neigh_empty_neighbors_are_finite_normalized_default():
    y = np.array([0, 1, 1])
    soft = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    empty_edge = EdgeIndex(
        edge_index=np.zeros((2, 0), dtype=int),
        edge_weight=np.zeros(0),
        feature_mask=np.ones(2, dtype=bool),
        node_mask=np.ones(3, dtype=bool),
        batches=[np.arange(3)],
    )
    graph = SimpleNamespace(views={"host": empty_edge}, temporal_pairs=np.zeros((2, 0), dtype=int))

    result = graph_cdm.forward(np.array([[0, 1, 1]]), soft, y, _cfg(), graph=graph, return_components=True)

    assert np.all(np.isfinite(result.d_neigh))
    assert np.all((result.d_neigh >= 0.0) & (result.d_neigh <= 1.0))


def test_d_view_is_mode_disagreement_not_embedding_distance():
    y = np.array([0, 1, 2])
    view_labels = np.array([[0, 1, 2], [1, 1, 2], [1, 2, 0], [1, 2, 0]])
    soft = np.eye(3)[y] * 0.9 + 0.1 / 3

    result = graph_cdm.forward(view_labels, soft, y, _cfg(), neigh_agg=soft, return_components=True)

    np.testing.assert_allclose(result.d_view, np.array([0.25, 0.5, 0.5]))


def test_rho_positive_weights_are_finite_and_strictly_positive():
    cdm = np.array([0.0, 0.5, 1.0])
    evidence = np.array([0.0, 0.5, 1.0])

    weights = graph_cdm.soft_weights(cdm, evidence, _cfg(rho=0.2))

    assert weights.min() > 0.0
    assert np.all(np.isfinite(weights))


def test_evidence_supports_log_and_inverse_and_minmax_normalizes():
    y = np.array([0, 0, 0, 1, 2, 2])
    anomaly = np.array([0.0, 0.1, 0.2, 1.0, 0.3, 0.4])

    evidence_log = compute_evidence(y, _cfg(freq="log"), anomaly=anomaly)
    evidence_inv = compute_evidence(y, _cfg(freq="inverse"), anomaly=anomaly)

    assert np.isclose(evidence_log.min(), 0.0)
    assert np.isclose(evidence_log.max(), 1.0)
    assert np.isclose(evidence_inv.min(), 0.0)
    assert np.isclose(evidence_inv.max(), 1.0)
    assert not np.allclose(evidence_log, evidence_inv)


def test_err_main_tail_and_final_formula_with_zero_denominator_guard():
    weights = np.array([1.0, 0.2, 0.8, 0.1, 0.4, 0.9])
    evidence = np.array([0.1, 0.1, 0.8, 0.7, 0.6, 0.6])
    flip = np.array([False, True, False, False, False, True])
    clean = ~flip
    y = np.array([0, 0, 1, 2, 2, 2])

    components = evidence_retention_components(weights, evidence, clean, y, retention_threshold=0.5)
    cfg = {"evidence_preserving": {"evidence_scores": evidence, "flip_mask": flip, "retention_threshold": 0.5}}
    err = evidence_retention_rate(weights, clean, y, cfg)

    retained = weights >= 0.5
    tail_mask = clean & np.isin(y, [0, 1])
    anomaly_mask = clean & (evidence >= np.quantile(evidence[clean], 0.75))
    informative = clean & (tail_mask | anomaly_mask)
    main = np.sum(retained[informative] * evidence[informative]) / np.sum(evidence[informative])
    tail = np.sum(retained[tail_mask] * evidence[tail_mask]) / np.sum(evidence[tail_mask])
    np.testing.assert_allclose(components["err"], main)
    np.testing.assert_allclose(components["err_tail"], tail)
    np.testing.assert_allclose(err, 0.5 * (main + tail))
    assert components["informative_count"] >= components["tail_count"]
    assert evidence_retention_rate(weights, clean, y, {"evidence_preserving": {"evidence_scores": np.zeros(6), "flip_mask": flip}}) == 0.0
