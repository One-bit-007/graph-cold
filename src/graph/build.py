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


@dataclass
class MultiViewGraph:
    views: dict[str, Any]
    node_features: Any
    node_index: dict
    snapshots: list | None = None


def build_multiview_graph(dataset, cfg) -> MultiViewGraph:
    raise NotImplementedError("TODO(Codex): implement five-view graph builder.")


def local_consistency(graph: MultiViewGraph, view: str, features) -> Any:
    """Per-node local-consistency score (higher = more overlap across classes)."""
    raise NotImplementedError("TODO(Codex)")
