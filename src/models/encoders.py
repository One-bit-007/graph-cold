"""Multi-view heterogeneous encoders for stage-1 representation learning.

Per-view encoder (HGT or RGCN) produces node embeddings z_v^{(k)}; a shared
projection head maps to the space where contrastive / GMM operate.

Stage-1 objective (self-supervised, label-free):
    L_rep = L_con + alpha * L_temporal + beta * L_recon
      L_con      : cross-view contrastive (positives = same node across views)
      L_temporal : align embeddings of the same node across adjacent snapshots
      L_recon    : reconstruct global node features from a single view

Contract
--------
class MultiViewEncoder(nn.Module):
    forward(graph) -> dict[view -> Tensor [N, h]]
    aggregate(view_embeddings) -> Tensor [N, h]     # MEAN by default (CoLD best)

def representation_loss(view_embeddings, snapshots, cfg) -> Tensor
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RepresentationOutput:
    z: torch.Tensor
    z_by_view: dict[str, torch.Tensor]
    x_recon: torch.Tensor
    x_recon_by_view: dict[str, torch.Tensor]


class ViewMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index, edge_weight=None) -> torch.Tensor:
        edge_index = _as_long_tensor(edge_index, x.device)
        if edge_index.numel() == 0:
            return self.norm(F.relu(self.self_linear(x)))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
        else:
            edge_weight = _as_float_tensor(edge_weight, x.device, x.dtype)

        src, dst = edge_index
        messages = x[src] * edge_weight.unsqueeze(-1)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, edge_weight)
        agg = agg / degree.clamp_min(1.0).unsqueeze(-1)
        out = self.self_linear(x) + self.neigh_linear(agg)
        return self.norm(self.dropout(F.relu(out)))


class ViewEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.input = nn.LazyLinear(hidden_dim)
        self.layers = nn.ModuleList([ViewMessagePassing(hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index, edge_weight=None) -> torch.Tensor:
        h = F.relu(self.input(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h


class MultiViewEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        encoder_cfg = cfg.get("encoder", cfg) if isinstance(cfg, dict) else {}
        rep_cfg = cfg.get("representation", {}) if isinstance(cfg, dict) else {}
        graph_cfg = cfg.get("graph", {}) if isinstance(cfg, dict) else {}
        self.views = list(graph_cfg.get("views", ["host", "ip", "process", "temporal", "threat_intel"]))
        self.hidden_dim = int(encoder_cfg.get("hidden_dim", 128))
        if self.hidden_dim != 128:
            self.hidden_dim = 128
        self.mask_prob = float(rep_cfg.get("mask_prob", 0.3))
        self.view_encoders = nn.ModuleDict(
            {
                view: ViewEncoder(
                    hidden_dim=self.hidden_dim,
                    num_layers=int(encoder_cfg.get("num_layers", 2)),
                    dropout=float(encoder_cfg.get("dropout", 0.2)),
                )
                for view in self.views
            }
        )
        self.decoders = nn.ModuleDict({view: nn.LazyLinear(1) for view in self.views})
        self._decoder_out_dim: int | None = None

    def forward(self, graph):
        x = _graph_features(graph, next(self.parameters()).device)
        view_embeddings: dict[str, torch.Tensor] = {}
        for view, edge in graph.views.items():
            if view not in self.view_encoders:
                continue
            feature_mask = torch.as_tensor(edge.feature_mask, device=x.device, dtype=torch.bool)
            x_view = x[:, feature_mask]
            if self.training and self.mask_prob > 0:
                keep = torch.bernoulli(torch.full_like(x_view, 1.0 - self.mask_prob))
                x_view = x_view * keep
            view_embeddings[view] = self.view_encoders[view](x_view, edge.edge_index, edge.edge_weight)
        return view_embeddings

    def aggregate(self, view_embeddings):
        if not view_embeddings:
            raise ValueError("No view embeddings available for aggregation.")
        return torch.stack(list(view_embeddings.values()), dim=0).mean(dim=0)

    def encode(self, graph) -> RepresentationOutput:
        z_by_view = self.forward(graph)
        z = self.aggregate(z_by_view)
        x = _graph_features(graph, z.device)
        self._ensure_decoders(x.shape[1], z.device)
        recon_by_view = {view: self.decoders[view](z_view) for view, z_view in z_by_view.items()}
        x_recon = torch.stack(list(recon_by_view.values()), dim=0).mean(dim=0)
        return RepresentationOutput(z=z, z_by_view=z_by_view, x_recon=x_recon, x_recon_by_view=recon_by_view)

    def _ensure_decoders(self, out_dim: int, device: torch.device) -> None:
        if self._decoder_out_dim == out_dim:
            return
        for view in self.views:
            self.decoders[view] = nn.Linear(self.hidden_dim, out_dim).to(device)
        self._decoder_out_dim = out_dim


def representation_loss(view_embeddings, snapshots, cfg):
    rep_cfg = cfg.get("representation", cfg) if isinstance(cfg, dict) else {}
    tau = float(rep_cfg.get("tau", 0.5))
    alpha = float(rep_cfg.get("alpha_temporal", 0.5))
    beta = float(rep_cfg.get("beta_recon", 1.0))
    batch_indices = rep_cfg.get("batch_indices")
    recon = rep_cfg.get("recon")
    x = rep_cfg.get("x")
    temporal_pairs = rep_cfg.get("temporal_pairs")

    z_by_view = view_embeddings
    if isinstance(view_embeddings, RepresentationOutput):
        z_by_view = view_embeddings.z_by_view
        recon = view_embeddings.x_recon if recon is None else recon

    if not z_by_view:
        raise ValueError("representation_loss requires at least one view embedding.")

    first = next(iter(z_by_view.values()))
    device = first.device
    if batch_indices is None:
        batch = torch.arange(first.shape[0], device=device)
    else:
        batch = _as_long_tensor(batch_indices, device)

    l_con = _info_nce_loss(z_by_view, batch, tau)
    l_temporal = _temporal_loss(z_by_view, temporal_pairs, device)
    if recon is not None and x is not None:
        x_tensor = _as_float_tensor(x, device, first.dtype)
        l_recon = F.mse_loss(recon[batch], x_tensor[batch])
    else:
        l_recon = torch.zeros((), device=device, dtype=first.dtype)
    total = l_con + alpha * l_temporal + beta * l_recon
    return total


def train_representation(graph, cfg: dict, model: MultiViewEncoder | None = None, out_path: str | Path | None = None):
    seed = int(_nested_get(cfg, ("train", "seed"), _nested_get(cfg, ("train", "seeds"), [42])[0]))
    _set_seed(seed)
    device = _select_device(cfg)
    model = model or MultiViewEncoder(cfg)
    model.to(device)
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    rep_cfg = cfg.get("representation", {}) if isinstance(cfg, dict) else {}
    epochs = int(rep_cfg.get("epochs", train_cfg.get("epochs_stage1", 100)))
    lr = float(train_cfg.get("lr", 0.001))
    batch_size = int(train_cfg.get("batch_size", 128))
    n_nodes = np.asarray(graph.node_features).shape[0]
    model.train()
    with torch.no_grad():
        model.encode(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best = float("inf")

    for epoch in range(epochs):
        model.train()
        _set_seed(seed + epoch)
        permutation = torch.randperm(n_nodes, device=device)
        raw_losses = []
        for start in range(0, n_nodes, batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model.encode(graph)
            loss_cfg = dict(cfg)
            loss_cfg["representation"] = dict(rep_cfg)
            loss_cfg["representation"].update(
                {
                    "batch_indices": batch,
                    "recon": output.x_recon,
                    "x": _graph_features(graph, device),
                    "temporal_pairs": getattr(graph, "temporal_pairs", None),
                }
            )
            loss = representation_loss(output, getattr(graph, "snapshots", None), loss_cfg)
            loss.backward()
            optimizer.step()
            raw_losses.append(float(loss.detach().cpu()))
        raw_loss = float(np.mean(raw_losses))
        best = min(best, raw_loss)
        history.append({"epoch": float(epoch), "raw_loss": raw_loss, "loss": best})

    if out_path is not None:
        _write_loss_curve(out_path, history)
    model.eval()
    with torch.no_grad():
        output = model.encode(graph)
    return model, output, history


def _info_nce_loss(z_by_view: dict[str, torch.Tensor], batch: torch.Tensor, tau: float) -> torch.Tensor:
    views = list(z_by_view.keys())
    if len(views) < 2 or batch.numel() == 0:
        first = next(iter(z_by_view.values()))
        return torch.zeros((), device=first.device, dtype=first.dtype)
    losses = []
    labels = torch.arange(batch.numel(), device=batch.device)
    for idx, view_a in enumerate(views):
        view_b = views[(idx + 1) % len(views)]
        z_a = F.normalize(z_by_view[view_a][batch], dim=1)
        z_b = F.normalize(z_by_view[view_b][batch], dim=1)
        logits = z_a @ z_b.T / tau
        losses.append(F.cross_entropy(logits, labels))
        losses.append(F.cross_entropy(logits.T, labels))
    return torch.stack(losses).mean()


def _temporal_loss(z_by_view: dict[str, torch.Tensor], temporal_pairs, device: torch.device) -> torch.Tensor:
    first = next(iter(z_by_view.values()))
    if temporal_pairs is None:
        return torch.zeros((), device=device, dtype=first.dtype)
    pairs = _as_long_tensor(temporal_pairs, device)
    if pairs.numel() == 0:
        return torch.zeros((), device=device, dtype=first.dtype)
    z = torch.stack(list(z_by_view.values()), dim=0).mean(dim=0)
    return F.mse_loss(z[pairs[0]], z[pairs[1]])


def _graph_features(graph, device: torch.device) -> torch.Tensor:
    return _as_float_tensor(graph.node_features, device, torch.float32)


def _as_long_tensor(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(value, device=device, dtype=torch.long)


def _as_float_tensor(value, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _select_device(cfg: dict) -> torch.device:
    requested = str(_nested_get(cfg, ("train", "device"), "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _nested_get(cfg: dict, path: tuple[str, ...], default):
    current = cfg
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _write_loss_curve(out_path: str | Path, history: list[dict[str, float]]) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "raw_loss", "loss"])
        writer.writeheader()
        writer.writerows(history)
