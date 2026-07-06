import numpy as np
import pandas as pd

from src.analysis.evidence_downstream import (
    counterfactual_soft_retention,
    pair_graphcold_vs_hard,
    tail_class_recall,
)


def test_tail_class_recall_uses_low_frequency_non_benign_classes():
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0])

    assert tail_class_recall(y_true, y_pred, benign_class=0) == 0.5


def test_pair_graphcold_vs_hard_outputs_high_noise_fnr_delta():
    frame = pd.DataFrame(
        [
            {"dataset": "cicids2017", "noise_type": "symmetric", "noise_rate": 0.4, "graph_beta": "none", "seed": 0, "method": "Graph-CoLD", "macro_f1": 0.8, "fnr": 0.2, "tail_recall": 0.7},
            {"dataset": "cicids2017", "noise_type": "symmetric", "noise_rate": 0.4, "graph_beta": "none", "seed": 0, "method": "ablation_hard", "macro_f1": 0.7, "fnr": 0.3, "tail_recall": 0.6},
        ]
    )

    paired = pair_graphcold_vs_hard(frame)

    assert len(paired) == 1
    assert paired.loc[0, "macro_f1_delta"] == 0.10000000000000009
    assert paired.loc[0, "fnr_delta_graphcold_minus_hard"] == -0.09999999999999998
    assert paired.loc[0, "tail_recall_delta"] == 0.09999999999999998


def test_counterfactual_soft_retention_reports_clean_and_correct_fractions():
    soft = np.array([0.2, 0.3, 0.05, 0.4])
    hard = np.array([1.0, 0.0, 0.0, 0.0])
    flip = np.array([False, True, False, False])
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 2])

    report = counterfactual_soft_retention(soft, hard, flip, y_true, y_pred_soft=y_pred)

    assert report["soft_retained_hard_deleted_n"] == 2.0
    assert report["soft_retained_hard_deleted_clean_fraction"] == 0.5
    assert report["soft_retained_hard_deleted_correct_fraction"] == 1.0
