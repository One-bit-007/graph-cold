import numpy as np

from src.metrics import degree_evidence_retention_components, rare_evidence_recovery_rate
from src.models.graph_cdm import soft_weights


def test_evidence_rescue_soft_weights_keep_high_evidence_high_cdm_samples():
    cdm = np.array([0.9, 0.9, 0.2], dtype=float)
    evidence = np.array([1.0, 0.0, 0.5], dtype=float)

    hard = soft_weights(cdm, evidence, {"evidence_preserving": {"theta": 0.5, "rho": 0.0}})
    soft = soft_weights(
        cdm,
        evidence,
        {"evidence_preserving": {"theta": 0.5, "kappa": 8.0, "rho": 0.2, "lambda_rescue": 0.8}},
    )

    assert hard.tolist() == [0.0, 0.0, 1.0]
    assert soft[0] >= 0.1
    assert soft[1] < 0.1
    assert soft[0] > soft[1]
    assert soft[2] > soft[0]


def test_degree_err_rewards_soft_evidence_rescue_over_hard_deletion():
    y = np.array([1, 1, 2, 2], dtype=int)
    clean = np.array([True, True, True, False])
    evidence = np.array([1.0, 0.8, 0.4, 0.2], dtype=float)
    hard = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    soft = np.array([0.8, 0.6, 1.0, 0.0], dtype=float)

    hard_err = degree_evidence_retention_components(hard, evidence, clean, y)["err_final_degree"]
    soft_err = degree_evidence_retention_components(soft, evidence, clean, y)["err_final_degree"]

    assert soft_err > hard_err
    assert 0.0 <= hard_err < 1.0
    assert 0.0 < soft_err <= 1.0


def test_rare_evidence_recovery_rate_distinguishes_soft_from_hard():
    y_true = np.array([1, 1, 2, 0], dtype=int)
    y_pred = np.array([1, 0, 2, 0], dtype=int)
    clean = np.array([True, True, True, True])
    suspicious = np.array([True, True, False, False])
    tail_labels = np.array([1], dtype=int)
    hard = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    soft = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)

    hard_recovery = rare_evidence_recovery_rate(hard, y_true, y_pred, clean, suspicious, tail_labels)
    soft_recovery = rare_evidence_recovery_rate(soft, y_true, y_pred, clean, suspicious, tail_labels)

    assert hard_recovery["rare_recovery_rate"] == 0.0
    assert soft_recovery["rare_recovery_rate"] == 0.5
    assert soft_recovery["rare_retained_rate"] == 1.0
