import json
from pathlib import Path

import pandas as pd

from src.paper.p2_status import generate_p2_status


def _ensure_p2_outputs() -> dict:
    if not Path("reports/p2_status.json").exists():
        generate_p2_status()
    return json.loads(Path("reports/p2_status.json").read_text(encoding="utf-8"))


def test_p2_status_reports_all_p1_gates_passed_and_all_goals_closed():
    report = _ensure_p2_outputs()

    assert report["stage"] == "P2"
    assert all(info["passed"] for key, info in report["p1_gate"].items() if key.startswith("A"))
    assert report["p1_gate"]["grep"]["passed"] is True
    assert all(goal["status"] == "completed" for goal in report["p2"].values())


def test_p2_unsw_is_present_in_formal_outputs_when_ingested():
    report = _ensure_p2_outputs()
    assert report["p2"]["G3"]["ready_for_d5_component"] is True
    assert report["p2"]["G3"]["layout"] == "partition"
    assert report["p2"]["G3"]["active_views"] == ["temporal", "process"]

    main = pd.read_csv("results/table_main_expanded.csv")
    assert "UNSW-NB15" in set(main["reported_as"])
    assert set(main[main["reported_as"] == "UNSW-NB15"]["active_views"]) == {"temporal|process"}
    for table_path in [
        "tables/table_1_dataset_protocol.csv",
        "tables/table_2_main_performance.csv",
        "tables/table_3_high_noise_summary.csv",
        "tables/table_4_ablation_evidence.csv",
    ]:
        assert "UNSW-NB15" in Path(table_path).read_text(encoding="utf-8")
    table5 = pd.read_csv("tables/table_5_statistical_tests.csv")
    assert (table5["n"] == 51).all()
    assert table5["Test type"].str.contains("scenario-level").all()


def test_p2_prioritization_claim_is_either_measured_or_honestly_rescoped():
    report = _ensure_p2_outputs()
    claim = report["p2"]["G4"]["claim"]
    assert claim in {
        "measured_advantage_vs_cold_or_hard",
        "rescoped_to_evidence_retention_not_raw_topk_priority",
    }
    assert Path(report["p2"]["G4"]["table"]).exists()
    if claim == "rescoped_to_evidence_retention_not_raw_topk_priority":
        residual = " ".join(report["post_p2_risk"]["remaining_weaknesses"])
        assert "Top-K" in residual


def test_p2_contribution_decomposition_supports_evidence_retention_narrative():
    report = _ensure_p2_outputs()
    table_path = Path(report["p2"]["G5"]["table"])
    figure_path = Path(report["p2"]["G5"]["figure"])
    table = pd.read_csv(table_path)

    assert figure_path.exists()
    assert figure_path.stat().st_size > 1000
    assert {"macro_f1", "err_final", "tail_recall", "fnr"} == set(table["metric"])
    assert report["p2"]["G5"]["macro_f1_evidence_gain"] > 0
    assert report["p2"]["G5"]["err_evidence_gain"] > report["p2"]["G5"]["macro_f1_evidence_gain"]
