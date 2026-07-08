from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import rare_evidence_recovery_rate


def test_rare_recovery_ignores_retention_in_numerator():
    y_true = np.array([1, 1, 1, 2], dtype=int)
    y_pred = np.array([1, 0, 1, 2], dtype=int)
    clean = np.array([True, True, True, True])
    suspicious = np.array([True, True, True, False])
    tail = np.array([1], dtype=int)
    hard_deleted = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    soft_retained = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    hard = rare_evidence_recovery_rate(hard_deleted, y_true, y_pred, clean, suspicious, tail)
    soft = rare_evidence_recovery_rate(soft_retained, y_true, y_pred, clean, suspicious, tail)

    assert hard["rare_recovery_rate"] == soft["rare_recovery_rate"] == 2 / 3
    assert hard["rare_retained_rate"] == 0.0
    assert soft["rare_retained_rate"] == 1.0


def test_p2f_real_results_show_nonconstant_rare_recovery_if_available():
    path = Path("results/p2f_tail_powered.csv")
    if not path.exists():
        return
    frame = pd.read_csv(path)
    hard = frame[frame["method"] == "ablation_hard"]["rare_recovery_rate"].to_numpy(dtype=float)
    soft = frame[frame["method"] == "Graph-CoLD-soft"]["rare_recovery_rate"].to_numpy(dtype=float)

    assert hard.size > 0 and soft.size > 0
    assert np.max(hard) > 0.0
    assert np.min(soft) < 1.0
    assert not np.allclose(hard, 0.0)
    assert not np.allclose(soft, 1.0)
