"""Label-noise injection.

Injects noise ONLY into training labels. The test set stays clean.
Three schemes:

1. symmetric   : uniform corruption across all classes.
2. asymmetric  : corruption confined to malicious classes (-> benign).
3. graph_consistency (NEW): flips preferentially along locally-consistent edges.
   See docs/spec_graph_noise.md for the formal definition.

Contract
--------
inject_symmetric(y, ratio, num_classes, rng) -> y_noisy, flip_mask
inject_asymmetric(y, ratio, benign_class, rng) -> y_noisy, flip_mask
inject_graph_consistency(y, ratio, graph, cfg, rng) -> y_noisy, flip_mask

`flip_mask` (bool [N]) marks corrupted positions; used to compute noise-detection
metrics and the Evidence Retention Rate.
"""
from __future__ import annotations

import numpy as np


def inject_symmetric(y, ratio, num_classes, rng):
    rng = _as_generator(rng)
    y = np.asarray(y)
    y_noisy = y.copy()
    flip_mask = np.zeros(y.shape[0], dtype=bool)
    if ratio <= 0 or y.shape[0] == 0:
        return y_noisy, flip_mask
    if num_classes < 2:
        raise ValueError("Symmetric noise requires at least two classes.")

    n_flips = _num_flips(y.shape[0], ratio)
    if n_flips == 0:
        return y_noisy, flip_mask

    flip_idx = rng.choice(y.shape[0], size=n_flips, replace=False)
    offsets = rng.integers(1, num_classes, size=n_flips)
    y_noisy[flip_idx] = (y_noisy[flip_idx] + offsets) % num_classes
    flip_mask[flip_idx] = True
    return y_noisy, flip_mask


def inject_asymmetric(y, ratio, benign_class, rng):
    rng = _as_generator(rng)
    y = np.asarray(y)
    y_noisy = y.copy()
    flip_mask = np.zeros(y.shape[0], dtype=bool)
    if ratio <= 0 or y.shape[0] == 0:
        return y_noisy, flip_mask

    candidates = np.flatnonzero(y != benign_class)
    if candidates.size == 0:
        return y_noisy, flip_mask

    n_flips = min(_num_flips(y.shape[0], ratio), candidates.size)
    if n_flips == 0:
        return y_noisy, flip_mask

    flip_idx = rng.choice(candidates, size=n_flips, replace=False)
    y_noisy[flip_idx] = benign_class
    flip_mask[flip_idx] = True
    return y_noisy, flip_mask


def inject_graph_consistency(y, ratio, graph, cfg, rng):
    """Flip labels biased toward high graph-consistency edges.

    See docs/spec_graph_noise.md — must expose `consistency_bias` so that a
    fraction of flips land on locally-consistent edges and the rest are random.
    """
    rng = _as_generator(rng)
    y = np.asarray(y)
    y_noisy = y.copy()
    flip_mask = np.zeros(y.shape[0], dtype=bool)
    num_classes = _num_classes(y, cfg)
    n_flips = _num_flips(y.shape[0], ratio)
    if n_flips == 0 or y.shape[0] == 0:
        return y_noisy, flip_mask
    if num_classes < 2:
        raise ValueError("Graph-consistency noise requires at least two classes.")

    beta = _consistency_bias(cfg)
    if np.isclose(beta, 0.0):
        return inject_symmetric(y, ratio, num_classes, rng)

    n_graph = int(np.floor(beta * n_flips))
    n_rand = n_flips - n_graph
    if n_graph > 0:
        candidates = _graph_flip_candidates(y, graph, num_classes)
        if candidates:
            selected = _sample_graph_candidates(candidates, min(n_graph, len(candidates)), rng)
            for node, target, _score in selected:
                y_noisy[node] = target
                flip_mask[node] = True
        n_rand += n_graph - int(flip_mask.sum())

    if n_rand > 0:
        remaining = np.flatnonzero(~flip_mask)
        if remaining.size:
            chosen = rng.choice(remaining, size=min(n_rand, remaining.size), replace=False)
            offsets = rng.integers(1, num_classes, size=chosen.size)
            y_noisy[chosen] = (y_noisy[chosen] + offsets) % num_classes
            flip_mask[chosen] = True
    return y_noisy, flip_mask


def _as_generator(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _num_flips(n_samples: int, ratio: float) -> int:
    ratio = float(ratio)
    if ratio < 0 or ratio > 1:
        raise ValueError("Noise ratio must be in [0, 1].")
    return int(np.floor(ratio * n_samples))


def _num_classes(y: np.ndarray, cfg) -> int:
    if isinstance(cfg, dict):
        for key in ("num_classes", "n_classes"):
            if key in cfg and cfg[key] is not None:
                return int(cfg[key])
        nested = cfg.get("graph_consistency", {})
        if isinstance(nested, dict) and nested.get("num_classes") is not None:
            return int(nested["num_classes"])
    return int(np.max(y)) + 1 if y.size else 0


def _consistency_bias(cfg) -> float:
    if not isinstance(cfg, dict):
        return 0.8
    if "consistency_bias" in cfg:
        beta = float(cfg["consistency_bias"])
    else:
        beta = float(cfg.get("graph_consistency", {}).get("consistency_bias", 0.8))
    if beta < 0.0 or beta > 1.0:
        raise ValueError("consistency_bias must be in [0, 1].")
    return beta


def _graph_flip_candidates(y: np.ndarray, graph, num_classes: int) -> list[tuple[int, int, float]]:
    neighbors = _neighbors_from_graph(graph, y.shape[0])
    degree = np.asarray([sum(weight for _dst, weight in node_neighbors) for node_neighbors in neighbors], dtype=np.float64)
    max_degree = max(float(degree.max(initial=0.0)), 1e-12)
    candidates: list[tuple[int, int, float]] = []
    for node, node_neighbors in enumerate(neighbors):
        if not node_neighbors:
            continue
        counts = np.zeros(num_classes, dtype=np.float64)
        for dst, weight in node_neighbors:
            label = int(y[dst])
            if 0 <= label < num_classes:
                counts[label] += float(weight)
        total = counts.sum()
        if total <= 0.0:
            continue
        counts[int(y[node])] = -1.0
        target = int(np.argmax(counts))
        target_mass = counts[target]
        if target_mass <= 0.0 or target == int(y[node]):
            continue
        q = target_mass / total
        centrality = degree[node] / max_degree
        score = float(q * centrality)
        if score > 0.0:
            candidates.append((node, target, score))
    return candidates


def _neighbors_from_graph(graph, n_nodes: int) -> list[list[tuple[int, float]]]:
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    if graph is None:
        return neighbors
    views = getattr(graph, "views", None)
    if views is None and isinstance(graph, dict):
        views = graph
    if views is None:
        edge_index = getattr(graph, "edge_index", None)
        edge_weight = getattr(graph, "edge_weight", None)
        _add_edges(neighbors, edge_index, edge_weight)
        return neighbors
    for edge in views.values():
        if isinstance(edge, dict):
            edge_index = edge.get("edge_index")
            edge_weight = edge.get("edge_weight")
        else:
            edge_index = getattr(edge, "edge_index", None)
            edge_weight = getattr(edge, "edge_weight", None)
        _add_edges(neighbors, edge_index, edge_weight)
    return neighbors


def _add_edges(neighbors: list[list[tuple[int, float]]], edge_index, edge_weight) -> None:
    if edge_index is None:
        return
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.size == 0:
        return
    if edges.shape[0] != 2:
        edges = edges.T
    if edge_weight is None:
        weights = np.ones(edges.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(edge_weight, dtype=np.float64)
    n_nodes = len(neighbors)
    for idx in range(edges.shape[1]):
        src = int(edges[0, idx])
        dst = int(edges[1, idx])
        if 0 <= src < n_nodes and 0 <= dst < n_nodes and src != dst:
            neighbors[src].append((dst, float(max(weights[idx], 0.0))))


def _sample_graph_candidates(candidates: list[tuple[int, int, float]], n_samples: int, rng) -> list[tuple[int, int, float]]:
    scores = np.asarray([candidate[2] for candidate in candidates], dtype=np.float64)
    if scores.sum() <= 0.0:
        probs = None
    else:
        probs = scores / scores.sum()
    indices = rng.choice(len(candidates), size=n_samples, replace=False, p=probs)
    return [candidates[int(idx)] for idx in indices]
