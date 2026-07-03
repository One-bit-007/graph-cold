import pandas as pd
import pytest

from src.analysis.stat_tests import grouped_paired_summary, paired_summary


def test_paired_summary_uses_seed_matched_pairs():
    rows = []
    for seed in (0, 1, 2):
        rows.append({"dataset": "cicids2017", "noise_type": "symmetric", "noise_rate": 0.2, "beta": None, "seed": seed, "method": "Graph-CoLD", "macro_f1": 0.90 + seed * 0.01})
        rows.append({"dataset": "cicids2017", "noise_type": "symmetric", "noise_rate": 0.2, "beta": None, "seed": seed, "method": "CoLD", "macro_f1": 0.82 + seed * 0.01})
    frame = pd.DataFrame(rows)

    report = paired_summary(frame)

    assert report["n_pairs"] == 3
    assert report["naive_pooled_test_used"] is False
    assert report["mean_diff"] > 0


def test_grouped_paired_summary_reports_overall_and_groups():
    rows = []
    for rate in (0.1, 0.2):
        for seed in (0, 1, 2):
            rows.append({"dataset": "cicids2017", "noise_type": "symmetric", "noise_rate": rate, "seed": seed, "method": "Graph-CoLD", "macro_f1": 0.9})
            rows.append({"dataset": "cicids2017", "noise_type": "symmetric", "noise_rate": rate, "seed": seed, "method": "CoLD", "macro_f1": 0.8})
    report = grouped_paired_summary(pd.DataFrame(rows))

    assert report["overall"]["n_pairs"] == 6
    assert report["groups"]


def test_paired_summary_requires_seed_column():
    frame = pd.DataFrame(
        [
            {"dataset": "cicids2017", "method": "Graph-CoLD", "macro_f1": 0.9},
            {"dataset": "cicids2017", "method": "CoLD", "macro_f1": 0.8},
        ]
    )

    with pytest.raises(ValueError, match="seed column"):
        paired_summary(frame)
