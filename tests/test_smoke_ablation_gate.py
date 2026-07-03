from pathlib import Path

import pandas as pd

from src.experiments import smoke_ablation


def _frame():
    return pd.DataFrame(
        [
            {"method": "CoLD", "macro_f1": 0.91, "err_final": 0.62, "retained_fraction_clean_informative": 0.62, "n_eff_ratio": 0.78},
            {
                "method": "Graph-CoLD-full",
                "macro_f1": 0.90,
                "err_final": 0.86,
                "retained_fraction_clean_informative": 0.92,
                "n_eff_ratio": 0.82,
            },
            {
                "method": "Graph-CoLD-hard-ablation",
                "macro_f1": 0.905,
                "err_final": 0.70,
                "retained_fraction_clean_informative": 0.70,
                "n_eff_ratio": 0.76,
            },
            {"method": "Graph-CoLD-w=1", "macro_f1": 0.91, "err_final": 1.0, "retained_fraction_clean_informative": 1.0, "n_eff_ratio": 1.0},
            {
                "method": "Graph-CoLD-active-views-only",
                "macro_f1": 0.90,
                "err_final": 0.86,
                "retained_fraction_clean_informative": 0.92,
                "n_eff_ratio": 0.82,
            },
            {
                "method": "Graph-CoLD-weighted-loss-normalized",
                "macro_f1": 0.90,
                "err_final": 0.86,
                "retained_fraction_clean_informative": 0.92,
                "n_eff_ratio": 0.82,
            },
        ]
    )


def test_smoke_ablation_summary_enforces_fix_gate():
    report = smoke_ablation._summary(_frame(), "hash")

    assert report["passed"] is True
    assert report["key_metrics"]["graphcold_macro_f1"] >= report["key_metrics"]["cold_macro_f1"] - 0.03
    assert report["key_metrics"]["err_graphcold"] > report["key_metrics"]["err_hard_ablation"]


def test_smoke_ablation_writer_does_not_create_formal_d5_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    report = smoke_ablation._summary(_frame(), "hash")
    smoke_ablation._write_reports(report, _frame(), "reports")

    assert Path("reports/smoke_diagnosis_ablation.json").exists()
    assert Path("reports/smoke_diagnosis_ablation.md").exists()
    assert not Path("results/table_main.csv").exists()
    assert not Path("tables").exists()
    assert not Path("figures").exists()
    assert not Path("paper").exists()
