from types import SimpleNamespace

import numpy as np

from src.graph.build import VIEW_NAMES, build_multiview_graph
from src.models.encoders import MultiViewEncoder
from src.models.graph_cdm import forward as graph_cdm_forward


def _cfg():
    return {
        "graph": {"views": list(VIEW_NAMES), "knn_k": 2, "temporal_window": 60},
        "encoder": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
        "representation": {"mask_prob": 0.0},
        "train": {"batch_size": 8},
    }


def test_cicids_contract_disables_process_and_threat_intel_views():
    dataset = SimpleNamespace(
        X_train=np.arange(48, dtype=np.float32).reshape(8, 6),
        meta={
            "feature_names": ["src_port", "flow_bytes", "duration", "iat", "pkt_len", "rate"],
            "timestamps": {"train": np.arange(8)},
            "expected_view_support": {
                "host": True,
                "ip": True,
                "temporal": True,
                "process": False,
                "threat_intel": False,
            },
        },
    )

    graph = build_multiview_graph(dataset, _cfg())

    assert set(graph.views) == {"host", "ip", "temporal"}
    assert graph.active_views == ["host", "ip", "temporal"]
    assert graph.inactive_views == ["process", "threat_intel"]
    assert "process" not in graph.view_masks
    assert "threat_intel" not in graph.view_masks


def test_encoder_mean_fusion_only_uses_active_graph_views():
    dataset = SimpleNamespace(
        X_train=np.random.default_rng(42).normal(size=(10, 6)).astype(np.float32),
        meta={
            "feature_names": ["src_port", "flow_bytes", "duration", "iat", "pkt_len", "rate"],
            "expected_view_support": {
                "host": True,
                "ip": True,
                "temporal": True,
                "process": False,
                "threat_intel": False,
            },
        },
    )
    graph = build_multiview_graph(dataset, _cfg())
    model = MultiViewEncoder(_cfg())

    view_embeddings = model(graph)
    fused = model.aggregate(view_embeddings)

    assert set(view_embeddings) == {"host", "ip", "temporal"}
    assert fused.shape == (dataset.X_train.shape[0], 128)


def test_graph_cdm_d_pred_and_d_view_use_only_active_view_predictions():
    observed = np.array([0, 1, 1, 2])
    active_view_preds = {
        "host": np.array([0, 1, 0, 2]),
        "ip": np.array([0, 2, 1, 2]),
        "temporal": np.array([1, 1, 1, 0]),
    }
    inactive_view_preds = {
        **active_view_preds,
        "process": np.array([2, 2, 2, 2]),
        "threat_intel": np.array([2, 2, 2, 2]),
    }
    soft_labels = np.eye(3, dtype=float)[observed]

    active = graph_cdm_forward(active_view_preds, soft_labels, observed, {}, return_components=True)
    contaminated = graph_cdm_forward(inactive_view_preds, soft_labels, observed, {}, return_components=True)

    expected_d_pred = np.mean(np.vstack(list(active_view_preds.values())) != observed[None, :], axis=0)
    expected_d_view = np.array([1.0 - 2.0 / 3.0, 1.0 - 2.0 / 3.0, 1.0 - 2.0 / 3.0, 1.0 - 2.0 / 3.0])

    np.testing.assert_allclose(active.d_pred, expected_d_pred)
    np.testing.assert_allclose(active.d_view, expected_d_view)
    assert not np.allclose(active.d_pred, contaminated.d_pred)
    assert not np.allclose(active.d_view, contaminated.d_view)
