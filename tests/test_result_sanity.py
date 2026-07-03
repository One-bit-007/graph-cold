import pandas as pd

from src.analysis.result_sanity import check_results


def test_result_sanity_passes_reasonable_real_dataset_rows():
    frame = pd.DataFrame(
        [
            {"dataset": "cicids2017", "method": "Graph-CoLD", "noise_type": "symmetric", "noise_rate": 0.2, "beta": None, "seed": 0, "macro_f1": 0.91, "fpr": 0.04, "fnr": 0.06, "err": 0.82, "active_views": "host|ip|temporal"},
            {"dataset": "cicids2017", "method": "CoLD", "noise_type": "symmetric", "noise_rate": 0.2, "beta": None, "seed": 0, "macro_f1": 0.86, "fpr": 0.06, "fnr": 0.09, "err": 0.70, "active_views": "host|ip|temporal"},
            {"dataset": "cicids2017", "method": "ablation_hard", "noise_type": "symmetric", "noise_rate": 0.2, "beta": None, "seed": 0, "macro_f1": 0.85, "fpr": 0.06, "fnr": 0.10, "err": 0.69, "active_views": "host|ip|temporal"},
        ]
    )

    report = check_results(frame)

    assert report["passed"] is True


def test_result_sanity_flags_perfect_metric_anomaly_and_dishonest_dataset():
    frame = pd.DataFrame(
        [
            {"dataset": "synthetic", "method": "Graph-CoLD", "macro_f1": 1.0, "fpr": 0.0, "fnr": 0.0, "err": 1.0},
            {"dataset": "synthetic", "method": "CoLD", "macro_f1": 0.9, "fpr": 0.1, "fnr": 0.1, "err": 0.5},
        ]
    )

    report = check_results(frame)

    assert report["passed"] is False
    assert "no_perfect_anomaly" in report["blocking_reasons"]
    assert "dataset_names_honest" in report["blocking_reasons"]
