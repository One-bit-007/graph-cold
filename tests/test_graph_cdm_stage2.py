from types import SimpleNamespace

import numpy as np
import torch

from src.metrics import evidence_retention_rate
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.models.loss import compute as compute_loss
from src.ranking.prioritize import priority_scores, topk


def _cfg(rho=0.2):
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
            "freq_protect": "log",
            "retention_threshold": 0.5,
        },
        "ranking": {
            "alpha1": 1.0,
            "alpha2": 0.5,
            "alpha3": 0.25,
            "benign_class": 0,
        },
    }


def test_ck4_graph_cdm_is_label_space_ground_truth_and_mode_based():
    view_preds = np.array([[0, 1, 1, 2], [1, 1, 2, 2]])
    observed = np.array([0, 1, 2, 2])
    soft = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.1, 0.8],
        ]
    )
    neigh = soft.copy()

    result = graph_cdm.forward(view_preds, soft, observed, _cfg(), neigh_agg=neigh, return_components=True)

    np.testing.assert_allclose(result.d_pred, np.array([0.5, 0.0, 0.5, 0.0]))
    np.testing.assert_allclose(result.d_view, np.array([0.5, 0.0, 0.5, 0.0]))
    np.testing.assert_allclose(result.d_neigh, np.zeros(4))
    assert np.all((result.score >= 0.0) & (result.score <= 1.0))


def test_d_neigh_label_kl_and_d_chain_temporal_similarity():
    view_preds = np.array([[0, 1, 1], [0, 1, 1]])
    observed = np.array([0, 1, 1])
    soft = np.array([[0.95, 0.05], [0.5, 0.5], [0.05, 0.95]])
    neigh = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    cfg = _cfg()
    cfg["graph_cdm"]["lambda_chain"] = 0.4

    result = graph_cdm.forward(
        view_preds,
        soft,
        observed,
        cfg,
        neigh_agg=neigh,
        temporal_pairs=np.array([[0, 1], [1, 2]]),
        return_components=True,
    )

    assert result.d_neigh[0] > result.d_neigh[1]
    assert result.d_chain[1] > 0.0
    assert result.d_chain[0] > 0.0


def test_evidence_score_uses_frequency_anomaly_and_normalizes():
    y = np.array([0, 0, 0, 1, 2, 2])
    recon_error = np.array([0.0, 0.1, 0.0, 1.0, 0.2, 0.3])

    evidence = compute_evidence(y, _cfg(), recon_error=recon_error)

    assert np.isclose(evidence.min(), 0.0)
    assert np.isclose(evidence.max(), 1.0)
    assert evidence[3] == evidence.max()


def test_ck5_rho_zero_degenerates_to_cold_hard_keep_mask():
    cdm = np.array([0.1, 0.5, 0.7])
    evidence = np.array([0.0, 0.4, 1.0])

    weights = graph_cdm.soft_weights(cdm, evidence, _cfg(rho=0.0))

    np.testing.assert_array_equal(weights, np.array([1.0, 1.0, 0.0]))


def test_ck6_ranking_is_stable_and_err_tracks_evidence_retention():
    cdm = np.array([0.2, 0.8, 0.8, 0.1])
    evidence = np.array([0.1, 0.9, 0.9, 0.2])
    soft = np.array([[0.9, 0.1], [0.2, 0.8], [0.2, 0.8], [0.7, 0.3]])
    scores = priority_scores({"cdm": cdm, "evidence": evidence, "soft_labels": soft}, {}, _cfg())

    np.testing.assert_array_equal(topk(scores, 2), np.array([1, 2]))
    np.testing.assert_array_equal(topk(scores, 2), topk(scores.copy(), 2))
    flip = np.array([False, True, False, False])
    clean = ~flip
    cfg = _cfg()
    cfg["evidence_preserving"]["flip_mask"] = flip
    cfg["evidence_preserving"]["evidence_scores"] = np.array([0.1, 0.9, 0.9, 0.2])
    err_low = evidence_retention_rate(np.array([1.0, 0.0, 0.0, 1.0]), clean, np.array([0, 1, 1, 0]), cfg)
    err_high = evidence_retention_rate(np.array([1.0, 1.0, 1.0, 1.0]), clean, np.array([0, 1, 1, 0]), cfg)
    assert err_high > err_low


def test_ck7_stage2_end_to_end_and_weighted_loss_backprop_is_deterministic():
    torch.manual_seed(42)
    np.random.seed(42)
    view_preds = {
        "host": np.array([0, 1, 1, 0]),
        "ip": np.array([0, 1, 0, 0]),
        "temporal": np.array([0, 1, 1, 1]),
    }
    observed = np.array([0, 1, 1, 0])
    soft = np.array([[0.9, 0.1], [0.2, 0.8], [0.45, 0.55], [0.7, 0.3]])
    graph = SimpleNamespace(views={}, temporal_pairs=np.array([[0, 1], [1, 2]]))

    result_a = graph_cdm.forward(view_preds, soft, observed, _cfg(), graph=graph, return_components=True)
    result_b = graph_cdm.forward(view_preds, soft, observed, _cfg(), graph=graph, return_components=True)
    np.testing.assert_allclose(result_a.score, result_b.score)
    evidence = compute_evidence(observed, _cfg(), entropy=np.array([0.1, 0.2, 0.8, 0.3]))
    weights = graph_cdm.soft_weights(result_a.score, evidence, _cfg())
    scores = priority_scores({"graph_cdm": result_a.score, "evidence": evidence, "soft_labels": soft}, {}, _cfg())
    assert topk(scores, 2).shape == (2,)

    logits = torch.tensor([[2.0, 0.1], [0.2, 1.4], [0.3, 0.7], [1.3, 0.4]], requires_grad=True)
    loss = compute_loss(logits, observed, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_weighted_cls_loss_converges_on_fixed_labels():
    torch.manual_seed(42)
    y = torch.tensor([0, 1, 1, 0])
    weights = torch.tensor([1.0, 0.8, 0.6, 1.0])
    logits = torch.zeros((4, 2), requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.5)

    first = None
    last = None
    for step in range(20):
        optimizer.zero_grad()
        loss = compute_loss(logits, y, weights)
        if step == 0:
            first = float(loss.detach())
        loss.backward()
        optimizer.step()
        last = float(loss.detach())

    assert last < first
