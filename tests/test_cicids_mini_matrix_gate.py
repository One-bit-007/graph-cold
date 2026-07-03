from pathlib import Path

import pandas as pd

from src.experiments import cicids_mini_matrix


FIELDS = cicids_mini_matrix.FIELDNAMES


def _rows():
    rows = []
    for seed in (0, 1, 2):
        for spec in [
            ("clean", 0.0, None, 0.992, 0.991),
            ("symmetric", 0.1, None, 0.975, 0.990),
            ("symmetric", 0.2, None, 0.965, 0.988),
            ("symmetric", 0.4, None, 0.940, 0.970),
            ("graph_consistency", 0.1, 0.0, 0.975, 0.990),
            ("graph_consistency", 0.2, 0.0, 0.965, 0.988),
            ("graph_consistency", 0.4, 0.0, 0.940, 0.970),
            ("graph_consistency", 0.2, 0.6, 0.955, 0.982),
        ]:
            noise_type, rate, beta, cold, graph = spec
            for method, f1, err in [
                ("CoLD", cold, 0.80),
                ("ablation_hard", cold, 0.80),
                ("Graph-CoLD", graph, 0.95),
            ]:
                rows.append(
                    {
                        "dataset": "cicids2017",
                        "dataset_hash": "hash",
                        "class_policy": "postfilter11",
                        "num_classes": 11,
                        "noise_type": noise_type,
                        "noise_rate": rate,
                        "graph_beta": beta,
                        "seed": seed,
                        "method": method,
                        "macro_f1": f1,
                        "fpr": 0.01,
                        "fnr": 0.01,
                        "err": err,
                        "err_tail": err,
                        "err_final": err,
                        "compression_ratio": 0.9,
                        "mean_weight": 0.8,
                        "retained_fraction": 0.8,
                        "retained_fraction_clean_informative": 0.9,
                        "n_eff_ratio": 0.8,
                        "active_views": "host|ip|temporal",
                        "runtime_sec": 0.1,
                        "split_id": f"split-{seed}",
                    }
                )
    return pd.DataFrame(rows, columns=FIELDS)


def test_mini_matrix_gate_passes_expected_conditions():
    gate = cicids_mini_matrix.evaluate_gate(_rows(), {"scenario": {"split_id": "split-0"}})

    assert gate["passed"] is True
    assert gate["checks"]["ablation_hard_close_to_cold"]["passed"] is True
    assert gate["checks"]["err_direction"]["passed"] is True
    assert gate["checks"]["graph_beta0_equiv_symmetric"]["passed"] is True
    assert gate["d5_allowed"] is False


def test_mini_matrix_gate_fails_graphcold_collapse():
    frame = _rows()
    mask = (
        (frame["noise_type"] == "symmetric")
        & (frame["noise_rate"] == 0.4)
        & (frame["method"] == "Graph-CoLD")
    )
    frame.loc[mask, "macro_f1"] = 0.70

    gate = cicids_mini_matrix.evaluate_gate(frame, {"scenario": {"split_id": "split-0"}})

    assert gate["passed"] is False
    assert gate["checks"]["graphcold_no_collapse"]["passed"] is False


def test_mini_matrix_reports_do_not_create_formal_d5_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = _rows()
    gate = cicids_mini_matrix.evaluate_gate(frame, {"scenario": {"split_id": "split-0"}})
    gate["results_csv"] = "results/cicids_mini_matrix.csv"

    cicids_mini_matrix.write_mini_reports(frame, gate, "reports")

    assert Path("reports/cicids_mini_matrix_report.json").exists()
    assert Path("reports/cicids_mini_matrix_gate.json").exists()
    assert not Path("results/table_main.csv").exists()
    assert not Path("results/table_ablation.csv").exists()
    assert not Path("results/table_optc.csv").exists()
    assert not Path("tables").exists()
    assert not Path("figures").exists()
