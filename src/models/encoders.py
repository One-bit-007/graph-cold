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


class MultiViewEncoder:  # subclass nn.Module in implementation
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def forward(self, graph):
        raise NotImplementedError("TODO(Codex): HGT/RGCN per view.")

    def aggregate(self, view_embeddings):
        raise NotImplementedError("TODO(Codex): MEAN aggregation default.")


def representation_loss(view_embeddings, snapshots, cfg):
    raise NotImplementedError("TODO(Codex): L_con + alpha*L_temporal + beta*L_recon")
