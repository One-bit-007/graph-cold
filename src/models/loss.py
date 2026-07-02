"""Stage-2 weighted classification loss integration."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute(logits, y, weights, reduction: str = "mean"):
    """Compute sum_v w(v) * CE(y_v, y_hat_v).

    Accepts torch tensors or numpy arrays. Returns a torch scalar so callers can
    backpropagate through classifier logits.
    """
    logits_t = _tensor(logits, dtype=torch.float32)
    y_t = _tensor(y, dtype=torch.long, device=logits_t.device)
    weights_t = _tensor(weights, dtype=torch.float32, device=logits_t.device)
    ce = F.cross_entropy(logits_t, y_t, reduction="none")
    weighted = ce * weights_t
    if reduction == "sum":
        return weighted.sum()
    if reduction == "none":
        return weighted
    if reduction != "mean":
        raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")
    return weighted.sum() / weights_t.sum().clamp_min(1e-12)


def _tensor(value, dtype, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.to(dtype=dtype)
        return out.to(device=device) if device is not None else out
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
