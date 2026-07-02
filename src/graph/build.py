"""Five-view heterogeneous temporal graph construction.

On flow datasets (CICIDS/MALTLS) the "views" are derived from feature semantics
so that CoLD's feature-subset idea is lifted to a graph over samples:

    host          : nodes linked by shared host identifiers
    ip            : nodes linked by IP-communication features
    process       : nodes linked by process/behavioral features (if available)
    temporal      : nodes linked within the same temporal window / adjacency
    threat_intel  : nodes linked by shared IOC / threat-intel attributes

On OpTC the views come from the provenance graph (host/process/flow) directly.

Contract
--------
build_multiview_graph(dataset, cfg) -> MultiViewGraph
    MultiViewGraph:
        views: dict[str, EdgeIndex]     # per-view adjacency (torch_geometric)
        node_features: Tensor [N, d]
        node_index: mapping sample_id -> node
        snapshots: optional list for temporal modeling

local_consistency(graph, view, features) -> np.ndarray [N]
    Per-node local-consistency score used by graph-consistency noise and by
    Graph-CDM's D_neigh term. Reuse a KS-test / distribution-overlap measure
    consistent with CoLD's empirical analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


VIEW_NAMES = ("host", "ip", "process", "temporal", "threat_intel")


@dataclass
class EdgeIndex:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    feature_mask: np.ndarray
    node_mask: np.ndarray
    batches: list[np.ndarray]


@dataclass
class MultiViewGraph:
    views: dict[str, Any]
    node_features: Any
    node_index: dict
    snapshots: list | None = None
    view_masks: dict[str, Any] | None = None
    temporal_pairs: Any | None = None


def build_multiview_graph(dataset, cfg) -> MultiViewGraph:
    graph_cfg = cfg.get("graph", cfg) if isinstance(cfg, dict) else {}
    X = _dataset_features(dataset, graph_cfg.get("split", "train"))
    feature_names = list(getattr(dataset, "meta", {}).get("feature_names", []))
    if not feature_names:
        feature_names = [f"f_{idx}" for idx in range(X.shape[1])]

    n_nodes, n_features = X.shape
    if n_nodes == 0 or n_features == 0:
        raise ValueError("Cannot build a graph from an empty feature matrix.")

    batch_size = int(_nested_get(cfg, ("train", "batch_size"), graph_cfg.get("batch_size", 128)))
    batches = _make_batches(n_nodes, batch_size)
    timestamps = _dataset_timestamps(dataset, graph_cfg.get("split", "train"))

    views: dict[str, EdgeIndex] = {}
    for view in VIEW_NAMES:
        feature_mask = _select_feature_mask(view, feature_names, X)
        if view == "temporal":
            edge_index, edge_weight, snapshots, temporal_pairs = _temporal_edges(
                n_nodes,
                timestamps,
                int(graph_cfg.get("temporal_window", 3600)),
                batch_size,
            )
        else:
            k = int(graph_cfg.get("knn_k", 10))
            if view == "threat_intel":
                k = max(1, min(k, int(graph_cfg.get("threat_intel_k", max(1, k // 2)))))
            edge_index, edge_weight = _knn_edges(X[:, feature_mask], k=k)
            snapshots = None
            temporal_pairs = None

        if edge_index.shape[1] == 0 and n_nodes > 1:
            edge_index, edge_weight = _chain_edges(n_nodes)

        views[view] = EdgeIndex(
            edge_index=edge_index.astype(np.int64),
            edge_weight=edge_weight.astype(np.float32),
            feature_mask=feature_mask.astype(bool),
            node_mask=np.ones(n_nodes, dtype=bool),
            batches=batches,
        )

    _, _, temporal_snapshots, temporal_pairs_final = _temporal_edges(
        n_nodes,
        timestamps,
        int(graph_cfg.get("temporal_window", 3600)),
        batch_size,
    )

    return MultiViewGraph(
        views=views,
        node_features=X.astype(np.float32),
        node_index={idx: idx for idx in range(n_nodes)},
        snapshots=temporal_snapshots,
        view_masks={view: edge.feature_mask for view, edge in views.items()},
        temporal_pairs=temporal_pairs_final,
    )


def local_consistency(graph: MultiViewGraph, view: str, features) -> Any:
    """Per-node local-consistency score (higher = more overlap across classes)."""
    if view not in graph.views:
        raise KeyError(f"Unknown view '{view}'. Available views: {sorted(graph.views)}")
    X = np.asarray(features, dtype=np.float32)
    edge = graph.views[view]
    n_nodes = X.shape[0]
    scores = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    src, dst = edge.edge_index
    if src.size == 0:
        return scores

    diff = X[src] - X[dst]
    sim = 1.0 / (1.0 + np.linalg.norm(diff, axis=1))
    np.add.at(scores, src, sim)
    np.add.at(counts, src, 1.0)
    out = scores / np.maximum(counts, 1.0)
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out.astype(np.float32)


def _dataset_features(dataset, split: str) -> np.ndarray:
    if split == "test" and hasattr(dataset, "X_test"):
        return np.asarray(dataset.X_test, dtype=np.float32)
    if hasattr(dataset, "X_train"):
        return np.asarray(dataset.X_train, dtype=np.float32)
    if hasattr(dataset, "X"):
        return np.asarray(dataset.X, dtype=np.float32)
    return np.asarray(dataset, dtype=np.float32)


def _dataset_timestamps(dataset, split: str) -> np.ndarray | None:
    meta = getattr(dataset, "meta", {}) or {}
    timestamps = meta.get("timestamps")
    if isinstance(timestamps, dict):
        value = timestamps.get(split)
    else:
        value = timestamps
    if value is None:
        return None
    return np.asarray(value)


def _nested_get(cfg: dict, path: tuple[str, ...], default):
    current = cfg
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _make_batches(n_nodes: int, batch_size: int) -> list[np.ndarray]:
    batch_size = max(1, batch_size)
    return [np.arange(start, min(start + batch_size, n_nodes), dtype=np.int64) for start in range(0, n_nodes, batch_size)]


def _select_feature_mask(view: str, feature_names: list[str], X: np.ndarray) -> np.ndarray:
    lowered = [name.lower() for name in feature_names]
    keyword_map = {
        "host": ("host", "src", "dst", "endpoint", "sport", "dport", "port"),
        "ip": ("ip", "byte", "pkt", "packet", "flow", "duration", "iat", "rate", "len"),
        "process": ("proc", "service", "proto", "flag", "state", "behavior", "tls", "ssl"),
        "temporal": ("time", "iat", "duration", "flow", "active", "idle"),
        "threat_intel": ("ioc", "sig", "alert", "threat", "attack", "ja3", "sni", "cert", "fingerprint"),
    }
    keywords = keyword_map[view]
    mask = np.array([any(keyword in name for keyword in keywords) for name in lowered], dtype=bool)
    if mask.any():
        return mask

    n_features = len(feature_names)
    if n_features == 1:
        return np.ones(1, dtype=bool)

    variances = np.var(X, axis=0)
    ranked = np.argsort(variances)[::-1]
    block = max(1, int(np.ceil(n_features * 0.35)))
    fallback_offsets = {
        "host": 0,
        "ip": max(0, block // 2),
        "process": block,
        "temporal": max(0, n_features - block),
        "threat_intel": max(0, n_features - max(1, block // 2)),
    }
    start = min(fallback_offsets[view], max(0, n_features - 1))
    selected = ranked[start : start + block]
    if selected.size == 0:
        selected = ranked[:1]
    mask = np.zeros(n_features, dtype=bool)
    mask[selected] = True
    return mask


def _knn_edges(X_view: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = X_view.shape[0]
    if n_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    k = max(1, min(k, n_nodes - 1))
    X_view = np.nan_to_num(X_view, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(np.var(X_view, axis=0), 0.0):
        return _chain_edges(n_nodes)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_view)
    distances, indices = nn.kneighbors(X_view)
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), k)
    dst = indices[:, 1 : k + 1].reshape(-1).astype(np.int64)
    dist = distances[:, 1 : k + 1].reshape(-1)
    edge_index = np.vstack([src, dst])
    edge_weight = 1.0 / (1.0 + dist)
    return _dedupe_edges(edge_index, edge_weight)


def _temporal_edges(
    n_nodes: int,
    timestamps: np.ndarray | None,
    temporal_window: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    if n_nodes <= 1:
        empty = np.zeros((2, 0), dtype=np.int64)
        return empty, np.zeros(0, dtype=np.float32), [np.arange(n_nodes, dtype=np.int64)], empty

    if timestamps is None:
        snapshots = _make_batches(n_nodes, max(2, batch_size))
    else:
        values = _timestamp_to_seconds(timestamps)
        bins = np.floor((values - np.nanmin(values)) / max(1, temporal_window)).astype(np.int64)
        snapshots = [np.flatnonzero(bins == bin_id).astype(np.int64) for bin_id in np.unique(bins)]
        snapshots = [snapshot for snapshot in snapshots if snapshot.size > 0]

    edges = []
    weights = []
    for snapshot in snapshots:
        if snapshot.size == 1:
            continue
        ordered = np.sort(snapshot)
        src = ordered[:-1]
        dst = ordered[1:]
        edges.append(np.vstack([src, dst]))
        edges.append(np.vstack([dst, src]))
        weights.extend([1.0] * (2 * src.size))

    temporal_pairs = []
    for left, right in zip(snapshots[:-1], snapshots[1:]):
        width = min(left.size, right.size)
        if width:
            temporal_pairs.append(np.vstack([np.sort(left)[:width], np.sort(right)[:width]]))

    if edges:
        edge_index = np.hstack(edges).astype(np.int64)
        edge_weight = np.asarray(weights, dtype=np.float32)
    else:
        edge_index, edge_weight = _chain_edges(n_nodes)

    if temporal_pairs:
        pairs = np.hstack(temporal_pairs).astype(np.int64)
    else:
        pairs = np.zeros((2, 0), dtype=np.int64)
    return edge_index, edge_weight, snapshots, pairs


def _timestamp_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    import pandas as pd

    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    if parsed.notna().any():
        seconds = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.float64) / 1e9
        finite = np.isfinite(seconds)
        fill = np.nanmin(seconds[finite]) if finite.any() else 0.0
        seconds[~finite] = fill
        return seconds
    numeric = pd.to_numeric(timestamps, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(numeric).all():
        return np.arange(timestamps.shape[0], dtype=np.float64)
    return np.nan_to_num(numeric, nan=np.nanmin(numeric))


def _chain_edges(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    src = np.arange(n_nodes - 1, dtype=np.int64)
    dst = src + 1
    edge_index = np.hstack([np.vstack([src, dst]), np.vstack([dst, src])])
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weight


def _dedupe_edges(edge_index: np.ndarray, edge_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] == 0:
        return edge_index, edge_weight
    pairs: dict[tuple[int, int], float] = {}
    for idx in range(edge_index.shape[1]):
        key = (int(edge_index[0, idx]), int(edge_index[1, idx]))
        pairs[key] = max(pairs.get(key, 0.0), float(edge_weight[idx]))
    items = sorted(pairs.items())
    deduped_index = np.asarray([[src, dst] for (src, dst), _ in items], dtype=np.int64).T
    deduped_weight = np.asarray([weight for _, weight in items], dtype=np.float32)
    return deduped_index, deduped_weight
