from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.graph.build import VIEW_NAMES, build_multiview_graph
from src.models.encoders import MultiViewEncoder, train_representation


def _synthetic_dataset(n_per_class: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for label in range(3):
        center = np.array([label * 4.0, label * 3.0, label * 2.0, label, label + 0.5, label + 1.0])
        rows.append(rng.normal(loc=center, scale=0.25, size=(n_per_class, center.size)))
        labels.extend([label] * n_per_class)
    X = np.vstack(rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    timestamps = np.arange(X.shape[0]) * 60
    feature_names = ["host_score", "flow_bytes", "proto_state", "duration", "alert_sig", "ja3_cert"]
    return SimpleNamespace(
        X_train=X,
        y_train=y,
        meta={"feature_names": feature_names, "timestamps": {"train": timestamps}},
    )


def _cfg(seed: int = 42, epochs: int = 5):
    return {
        "graph": {
            "views": list(VIEW_NAMES),
            "temporal_window": 300,
            "knn_k": 4,
        },
        "encoder": {
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "representation": {
            "tau": 0.5,
            "alpha_temporal": 0.1,
            "beta_recon": 0.2,
            "mask_prob": 0.1,
            "epochs": epochs,
        },
        "train": {
            "seed": seed,
            "lr": 0.005,
            "batch_size": 64,
            "device": "cpu",
        },
    }


def test_build_multiview_graph_has_five_nonempty_sparse_views():
    dataset = _synthetic_dataset()
    graph = build_multiview_graph(dataset, _cfg())

    assert set(graph.views) == set(VIEW_NAMES)
    assert graph.node_features.shape == dataset.X_train.shape
    assert graph.view_masks is not None
    for view in VIEW_NAMES:
        edge = graph.views[view]
        assert edge.edge_index.shape[0] == 2
        assert edge.edge_index.shape[1] > 0
        assert edge.edge_weight.shape[0] == edge.edge_index.shape[1]
        assert edge.feature_mask.shape[0] == dataset.X_train.shape[1]
        assert edge.node_mask.all()


def test_representation_training_outputs_shape_loss_curve_and_reproducible_embeddings(tmp_path: Path):
    dataset = _synthetic_dataset()
    graph = build_multiview_graph(dataset, _cfg())
    curve_a = tmp_path / "loss_a.csv"
    curve_b = tmp_path / "loss_b.csv"

    _, output_a, history_a = train_representation(graph, _cfg(seed=42, epochs=4), out_path=curve_a)
    _, output_b, history_b = train_representation(graph, _cfg(seed=42, epochs=4), out_path=curve_b)

    assert output_a.z.shape == (dataset.X_train.shape[0], 128)
    assert set(output_a.z_by_view) == set(VIEW_NAMES)
    assert output_a.x_recon.shape == dataset.X_train.shape
    assert curve_a.exists()
    assert history_a[-1]["loss"] <= history_a[0]["loss"]
    assert [round(item["loss"], 6) for item in history_a] == sorted(
        (round(item["loss"], 6) for item in history_a),
        reverse=True,
    )
    np.testing.assert_allclose(output_a.z.detach().numpy(), output_b.z.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray([item["raw_loss"] for item in history_a]),
        np.asarray([item["raw_loss"] for item in history_b]),
        atol=1e-6,
    )


def test_representation_knn_probe_beats_random_baseline():
    dataset = _synthetic_dataset(n_per_class=24)
    graph = build_multiview_graph(dataset, _cfg(epochs=5))

    _, output, _ = train_representation(graph, _cfg(seed=42, epochs=5))
    embeddings = output.z.detach().numpy()
    train_idx, test_idx = train_test_split(
        np.arange(dataset.y_train.shape[0]),
        test_size=0.35,
        random_state=42,
        stratify=dataset.y_train,
    )
    probe = KNeighborsClassifier(n_neighbors=3)
    probe.fit(embeddings[train_idx], dataset.y_train[train_idx])
    preds = probe.predict(embeddings[test_idx])
    macro_f1 = f1_score(dataset.y_train[test_idx], preds, average="macro")

    random_baseline = 1.0 / len(np.unique(dataset.y_train))
    assert macro_f1 > random_baseline + 0.30


def test_forward_returns_per_view_embeddings_without_classifier():
    dataset = _synthetic_dataset()
    graph = build_multiview_graph(dataset, _cfg())
    torch.manual_seed(42)
    model = MultiViewEncoder(_cfg())

    view_embeddings = model(graph)
    z = model.aggregate(view_embeddings)

    assert z.shape == (dataset.X_train.shape[0], 128)
    assert set(view_embeddings) == set(VIEW_NAMES)
