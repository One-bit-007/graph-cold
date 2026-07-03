import numpy as np
import torch
import torch.nn.functional as F

from src.models import loss
from src.models.evidence import compute as compute_evidence


def test_weighted_ce_mean_is_sum_over_weight_sum():
    logits = torch.tensor([[4.0, 0.0], [0.5, 1.5], [0.1, 2.0]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)

    actual = loss.compute(logits, y, weights, reduction="mean")
    ce = F.cross_entropy(logits, y, reduction="none")
    expected = (ce * weights).sum() / weights.sum()

    torch.testing.assert_close(actual, expected)


def test_uniform_weight_scaling_does_not_change_mean_loss():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.5]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1], dtype=torch.long)

    base = loss.compute(logits, y, torch.ones(3), reduction="mean")
    scaled = loss.compute(logits, y, torch.full((3,), 0.05), reduction="mean")

    torch.testing.assert_close(base, scaled)


def test_constant_evidence_normalization_retains_all_samples():
    y = np.array([0, 0, 1, 1])
    evidence = compute_evidence(y, {"evidence_preserving": {"freq_protect": "log"}}, anomaly=np.ones(4))

    np.testing.assert_allclose(evidence, np.ones(4))
