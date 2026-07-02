"""CoLD baseline re-implementation (no official code exists).

Serves as (a) a comparison baseline and (b) the object of extension. Faithful to
the NDSS'26 slides:

  1. Feature Reordering: correlation matrix -> Maximum Spanning Tree -> DFS order.
  2. Local Joint Learning: partition reordered features into M subsets, feature
     obfuscation (Bernoulli mask), shared encoder, local-alignment contrastive
     loss L_la + global-reconstruction loss L_gr.
  3. Causal Collaborative Denoising: GMM over projected subset representations ->
     per-subset cluster labels -> CDM (fraction of subsets disagreeing with the
     observed label) -> HARD filter with epsilon = 0 -> train downstream classifier.

Contract
--------
class CoLD:
    fit_representation(X)           # stage 1 (self-supervised)
    purify(X, y) -> keep_mask       # stage 2: CDM > 0 removed (hard deletion)
    fit_classifier(X, y, keep_mask)
    predict(X) -> y_hat

Reference libs: sklearn.mixture.GaussianMixture, networkx (MST), torch encoder.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.mixture import GaussianMixture

from src.graph.build import build_multiview_graph
from src.models.encoders import train_representation


def feature_reordering(X):
    """Correlation -> MST (networkx) -> DFS traversal order. Returns index perm."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    n_features = X.shape[1]
    if n_features == 0:
        return np.array([], dtype=np.int64)
    if n_features == 1:
        return np.array([0], dtype=np.int64)

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            graph.add_edge(i, j, weight=float(corr[i, j]))

    tree = nx.maximum_spanning_tree(graph, weight="weight")
    order = list(nx.dfs_preorder_nodes(tree, source=0))
    if len(order) < n_features:
        order.extend(idx for idx in range(n_features) if idx not in order)
    return np.asarray(order, dtype=np.int64)


class CoLD:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        graph_cfg = cfg.get("graph", {}) if isinstance(cfg, dict) else {}
        cold_cfg = cfg.get("cold", {}) if isinstance(cfg, dict) else {}
        self.seed = int(cold_cfg.get("seed", train_cfg.get("seeds", [0])[0] if train_cfg else 0))
        self.max_subsets = int(cold_cfg.get("max_subsets", graph_cfg.get("max_subsets", 4)))
        self.latent_dim = int(cold_cfg.get("latent_dim", 8))
        self.epsilon = float(cold_cfg.get("epsilon", 0.0))
        self.gmm_max_iter = int(cold_cfg.get("gmm_max_iter", 100))
        self.classifier_estimators = int(cold_cfg.get("classifier_estimators", 200))
        self.feature_order_: np.ndarray | None = None
        self.subsets_: list[np.ndarray] = []
        self.gmms_: list[GaussianMixture] = []
        self.cluster_label_maps_: list[dict[int, int]] = []
        self.num_classes_: int | None = None
        self.classifier_: ExtraTreesClassifier | None = None
        self.used_full_training_recovery_: bool = False
        self.rep_encoder_ = None
        self.rep_history_ = None

    def fit_representation(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        self.feature_order_ = feature_reordering(X)
        dataset = _ArrayDataset(X)
        graph = build_multiview_graph(dataset, self.cfg)
        cfg = _representation_cfg(self.cfg, self.seed)
        self.rep_encoder_, _, self.rep_history_ = train_representation(graph, cfg)
        self.subsets_ = [np.flatnonzero(graph.views[view].feature_mask) for view in graph.views]
        return self

    def purify(self, X, y):
        """Return keep_mask; CoLD removes samples with CDM > epsilon (=0)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if not self.subsets_:
            self.fit_representation(X)

        self.num_classes_ = int(np.max(y)) + 1 if y.size else 0
        reps = self._transform_subsets(X)
        disagreements = []
        self.gmms_ = []
        self.cluster_label_maps_ = []

        for subset_idx, rep in enumerate(reps):
            n_components = min(self.num_classes_, rep.shape[0])
            if n_components < 2:
                pred_labels = np.full_like(y, fill_value=y[0] if y.size else 0)
                self.gmms_.append(None)  # type: ignore[arg-type]
                self.cluster_label_maps_.append({0: int(pred_labels[0]) if pred_labels.size else 0})
            else:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",
                    reg_covar=1e-6,
                    max_iter=self.gmm_max_iter,
                    random_state=self.seed + subset_idx,
                )
                clusters = gmm.fit_predict(rep)
                label_map = _majority_label_map(clusters, y)
                pred_labels = np.array([label_map[int(cluster)] for cluster in clusters], dtype=np.int64)
                self.gmms_.append(gmm)
                self.cluster_label_maps_.append(label_map)
            disagreements.append(pred_labels != y)

        cdm = np.mean(np.vstack(disagreements), axis=0) if disagreements else np.zeros_like(y, dtype=float)
        self.cdm_ = cdm
        keep_mask = cdm <= self.epsilon
        self.keep_mask_ = keep_mask
        return keep_mask

    def fit_classifier(self, X, y, keep_mask):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.shape[0] != y.shape[0]:
            raise ValueError("keep_mask must have the same length as y.")

        train_mask = keep_mask.copy()
        if np.unique(y[train_mask]).size < 2:
            train_mask = np.ones_like(keep_mask, dtype=bool)
            self.used_full_training_recovery_ = True

        self.classifier_ = ExtraTreesClassifier(
            n_estimators=self.classifier_estimators,
            random_state=self.seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.classifier_.fit(X[train_mask], y[train_mask])
        return self

    def predict(self, X):
        if self.classifier_ is None:
            raise RuntimeError("CoLD classifier is not fitted. Call fit_classifier first.")
        X = np.asarray(X, dtype=np.float32)
        return self.classifier_.predict(X)

    def _transform_subsets(self, X: np.ndarray) -> list[np.ndarray]:
        if self.rep_encoder_ is None:
            raise RuntimeError("Representation encoder is not fitted.")
        dataset = _ArrayDataset(X)
        graph = build_multiview_graph(dataset, self.cfg)
        self.rep_encoder_.eval()
        import torch

        with torch.no_grad():
            output = self.rep_encoder_.encode(graph)
        return [tensor.detach().cpu().numpy().astype(np.float32) for tensor in output.z_by_view.values()]


def _majority_label_map(clusters: np.ndarray, y: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    default_label = int(np.bincount(y).argmax()) if y.size else 0
    for cluster in np.unique(clusters):
        members = y[clusters == cluster]
        if members.size == 0:
            mapping[int(cluster)] = default_label
        else:
            mapping[int(cluster)] = int(np.bincount(members).argmax())
    return mapping


class _ArrayDataset:
    def __init__(self, X: np.ndarray):
        self.X_train = X
        self.meta = {"feature_names": [f"f_{idx}" for idx in range(X.shape[1])]}


def _representation_cfg(cfg: dict, seed: int) -> dict:
    out = dict(cfg)
    out.setdefault("encoder", {})
    out.setdefault("representation", {})
    out.setdefault("train", {})
    out["encoder"] = dict(out["encoder"])
    out["representation"] = dict(out["representation"])
    out["train"] = dict(out["train"])
    out["encoder"]["hidden_dim"] = 128
    out["representation"].setdefault("epochs", cfg.get("cold", {}).get("rep_epochs", 10))
    out["train"]["seed"] = seed
    out["train"]["device"] = "cuda" if out["train"].get("device") == "cuda" else "cpu"
    return out
