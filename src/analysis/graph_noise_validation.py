"""Graph-consistency noise validation artifacts."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.data.noise import inject_graph_consistency, inject_symmetric


def transition_matrix(y_true, y_noisy, flip_mask, num_classes: int, normalize: bool = True) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    flip = np.asarray(flip_mask, dtype=bool)
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.float64)
    for src, dst in zip(y_true[flip], y_noisy[flip]):
        if 0 <= int(src) < num_classes and 0 <= int(dst) < num_classes:
            matrix[int(src), int(dst)] += 1.0
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(row_sum, 1.0), out=np.zeros_like(matrix), where=row_sum > 0)
    return matrix


def beta0_matches_symmetric(y, ratio: float, num_classes: int, seed: int = 42) -> dict[str, Any]:
    graph = _validation_graph(np.asarray(y), num_classes)
    cfg = {"num_classes": num_classes, "graph_consistency": {"consistency_bias": 0.0}}
    noisy_graph, mask_graph = inject_graph_consistency(y, ratio, graph, cfg, np.random.default_rng(seed))
    noisy_sym, mask_sym = inject_symmetric(y, ratio, num_classes, np.random.default_rng(seed))
    return {
        "mask_equal": bool(np.array_equal(mask_graph, mask_sym)),
        "labels_equal": bool(np.array_equal(noisy_graph, noisy_sym)),
        "flip_count": int(mask_graph.sum()),
        "target_flip_count": int(np.floor(float(ratio) * len(y))),
        "transition_l1": float(np.abs(
            transition_matrix(y, noisy_graph, mask_graph, num_classes)
            - transition_matrix(y, noisy_sym, mask_sym, num_classes)
        ).sum()),
    }


def beta_sweep_report(
    ratio: float = 0.3,
    betas: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0),
    num_classes: int = 4,
    n_per_class: int = 150,
    seed: int = 42,
    figure_path: str | Path | None = "figures/fig_p1_graph_noise_beta_sweep.pdf",
) -> dict[str, Any]:
    y = np.repeat(np.arange(num_classes, dtype=np.int64), int(n_per_class))
    graph = _validation_graph(y, num_classes)
    matrices: dict[str, np.ndarray] = {}
    concentration: dict[str, float] = {}
    for beta in betas:
        cfg = {"num_classes": num_classes, "graph_consistency": {"consistency_bias": float(beta)}}
        noisy, mask = inject_graph_consistency(y, ratio, graph, cfg, np.random.default_rng(seed))
        mat = transition_matrix(y, noisy, mask, num_classes)
        matrices[f"{beta:.1f}"] = mat
        concentration[f"{beta:.1f}"] = _transition_concentration(mat)
    if figure_path is not None:
        write_beta_sweep_figure(matrices, figure_path)
    beta0 = beta0_matches_symmetric(y, ratio, num_classes, seed)
    return {
        "ratio": float(ratio),
        "betas": [float(beta) for beta in betas],
        "num_classes": int(num_classes),
        "n": int(y.shape[0]),
        "beta0_matches_symmetric": bool(beta0["mask_equal"] and beta0["labels_equal"]),
        "beta0_transition_l1": beta0["transition_l1"],
        "transition_concentration": concentration,
        "concentration_increases": bool(concentration[f"{betas[-1]:.1f}"] >= concentration[f"{betas[0]:.1f}"]),
        "figure": str(figure_path) if figure_path is not None else None,
    }


def write_beta_sweep_figure(matrices: dict[str, np.ndarray], figure_path: str | Path) -> None:
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(matrices)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.1 * len(labels), 3.0), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    vmax = max(float(mat.max()) for mat in matrices.values()) or 1.0
    for ax, label in zip(axes, labels):
        im = ax.imshow(matrices[label], vmin=0.0, vmax=vmax, cmap="magma")
        ax.set_title(f"beta={label}")
        ax.set_xlabel("Target class")
        ax.set_ylabel("Source class")
        ax.set_xticks(range(matrices[label].shape[1]))
        ax.set_yticks(range(matrices[label].shape[0]))
    fig.colorbar(im, ax=axes, shrink=0.78)
    fig.savefig(path)
    plt.close(fig)


def _transition_concentration(matrix: np.ndarray) -> float:
    off = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(off, 0.0)
    row_max = off.max(axis=1)
    row_sum = off.sum(axis=1)
    valid = row_sum > 0
    if not valid.any():
        return 0.0
    return float(np.mean(row_max[valid] / row_sum[valid]))


def _validation_graph(y: np.ndarray, num_classes: int):
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    by_class = {label: np.flatnonzero(y == label) for label in range(num_classes)}
    for label, nodes in by_class.items():
        target_label = (label + 1) % num_classes
        target_nodes = by_class[target_label]
        for offset, node in enumerate(nodes):
            target = int(target_nodes[offset % target_nodes.size])
            src.append(int(node))
            dst.append(target)
            weights.append(3.0)
            alt_label = (label + 2) % num_classes
            alt_nodes = by_class[alt_label]
            src.append(int(node))
            dst.append(int(alt_nodes[offset % alt_nodes.size]))
            weights.append(0.3)
    edge = SimpleNamespace(
        edge_index=np.asarray([src, dst], dtype=np.int64),
        edge_weight=np.asarray(weights, dtype=np.float32),
    )
    return SimpleNamespace(views={"ip": edge})
