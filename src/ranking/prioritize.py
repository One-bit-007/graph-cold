"""SOC alert prioritization (Top-K).

Priority score combines risk, false-positive likelihood, and false-negative risk,
all derived from Graph-CDM components and node attributes:

    P(v) = risk(v) * (1 - FP_prob(v)) + eta * FN_risk(v)

Outputs a ranking and the Top-K highest-risk alerts for the analyst. Enables the
alert-compression-ratio metric (how few alerts an analyst must inspect to cover
the true attacks).

Contract
--------
priority_scores(cdm_components, node_attrs, cfg) -> np.ndarray [N]
top_k(scores, k) -> indices
alert_compression_ratio(scores, y_true) -> float
"""
from __future__ import annotations

import numpy as np


def priority_scores(cdm_components, node_attrs, cfg):
    raise NotImplementedError("TODO(Codex)")


def top_k(scores, k):
    raise NotImplementedError("TODO(Codex)")


def alert_compression_ratio(scores, y_true):
    raise NotImplementedError("TODO(Codex)")
