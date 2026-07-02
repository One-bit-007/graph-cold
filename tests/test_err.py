import numpy as np
import pytest

from src.metrics import evidence_retention_rate


def _cfg(evidence, flip, tau=0.1):
    return {
        "evidence_preserving": {
            "evidence_scores": np.asarray(evidence, dtype=float),
            "flip_mask": np.asarray(flip, dtype=bool),
            "retention_threshold": tau,
        }
    }


def test_soft_weighting_err_exceeds_hard_deletion_on_same_clean_informative_set():
    y = np.array([0, 0, 1, 1, 2, 2])
    evidence = np.array([0.05, 0.1, 0.8, 0.7, 1.0, 0.9])
    flip = np.array([False, True, False, False, False, True])
    clean = ~flip

    soft_weights = np.array([0.9, 0.2, 0.35, 0.25, 0.4, 0.2])
    hard_delete = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    soft_err = evidence_retention_rate(soft_weights, clean, y, _cfg(evidence, flip, tau=0.1))
    hard_err = evidence_retention_rate(hard_delete, clean, y, _cfg(evidence, flip, tau=0.1))

    assert soft_err > hard_err


def test_all_soft_weights_above_threshold_retains_all_clean_informative_evidence():
    y = np.array([0, 0, 1, 1, 2])
    evidence = np.array([0.05, 0.1, 0.9, 0.8, 1.0])
    flip = np.array([False, True, False, False, False])
    clean = ~flip

    soft_weights = np.full_like(evidence, 0.2)
    hard_delete = np.array([1.0, 0.0, 1.0, 0.0, 0.0])

    soft_err = evidence_retention_rate(soft_weights, clean, y, _cfg(evidence, flip, tau=0.1))
    hard_err = evidence_retention_rate(hard_delete, clean, y, _cfg(evidence, flip, tau=0.1))

    assert soft_err == 1.0
    assert hard_err < 1.0


def test_clean_mask_must_be_derived_from_flip_mask_not_all_true_placeholder():
    y = np.array([0, 1, 1, 2])
    weights = np.ones(4)
    evidence = np.ones(4)
    flip = np.array([False, True, False, False])

    with pytest.raises(ValueError, match="complement of flip_mask"):
        evidence_retention_rate(weights, np.ones(4, dtype=bool), y, _cfg(evidence, flip))

    with pytest.raises(ValueError, match="real flip_mask"):
        evidence_retention_rate(weights, np.ones(4, dtype=bool), y, {"evidence_preserving": {"evidence_scores": evidence}})
