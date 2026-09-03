"""Canonical P9 reporting metrics; historical inferential helpers are untouched."""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np


def canonical_tail_f1_from_perclass(per_class_f1_json, tail_labels) -> float:
    """Average full-test-set per-class F1 over test-present tail classes.

    The caller must supply scores evaluated on the FULL test set, not on a
    true-tail-only row subset. Missing tail labels mean absent test classes,
    matching p3._tail_f1; callers must separately validate test-class coverage.
    No tail class in the test set yields NaN (never a fabricated zero).
    """
    values = json.loads(per_class_f1_json) if isinstance(per_class_f1_json, str) else per_class_f1_json
    if not isinstance(values, Mapping):
        raise ValueError("per-class F1 must be a JSON object or mapping")
    scores = {}
    for key, value in values.items():
        label = int(key)
        score = float(value)
        if label in scores or not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("duplicate class or invalid per-class F1")
        scores[label] = score
    selected = [scores[label] for label in sorted(set(map(int, tail_labels))) if label in scores]
    return float(np.mean(selected)) if selected else float("nan")
