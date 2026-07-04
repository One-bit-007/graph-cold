from pathlib import Path

import pandas as pd

from src.experiments import cesnet_mini_matrix
from tests.test_cicids_mini_matrix_gate import _rows


def test_cesnet_mini_matrix_gate_uses_shared_checks():
    frame = _rows().copy()
    frame["dataset"] = "cesnet_tls_year22"
    frame["class_policy"] = "postfilter"
    frame["active_views"] = "ip|temporal"

    gate = cesnet_mini_matrix.evaluate_gate(frame, {"scenario": {"split_id": "split-0"}})

    assert gate["passed"] is True
    assert gate["class_policy"] == "postfilter"
    assert gate["d5_allowed"] is True


def test_cesnet_mini_matrix_blocked_until_smoke_passes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path("configs").mkdir()
    (Path("configs") / "datasets.yaml").write_text("cesnet_tls_year22:\n  path: missing\n  label_col: service\n", encoding="utf-8")

    gate = cesnet_mini_matrix.run_mini_matrix("cesnet_tls_year22", configs="configs", out="results", reports="reports")

    assert gate["status"] == "blocked"
    assert gate["passed"] is False
    assert not Path("results/cesnet_mini_matrix.csv").exists()
    assert Path("reports/cesnet_mini_matrix_gate.json").exists()


def test_cesnet_mini_matrix_writer_does_not_create_formal_d5_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame()
    gate = {"passed": False, "blocking_reasons": ["blocked"], "results_csv": None}

    cesnet_mini_matrix.write_mini_reports(frame, gate, "reports")

    assert Path("reports/cesnet_mini_matrix_report.json").exists()
    assert not Path("results/table_main.csv").exists()
    assert not Path("results/table_ablation.csv").exists()
    assert not Path("results/table_optc.csv").exists()
    assert not Path("tables").exists()
    assert not Path("figures").exists()
