import json
from pathlib import Path

from src.experiments import cicids_mini_matrix


def test_second_dataset_decision_blocks_maltls_and_recommends_cesnet(tmp_path: Path):
    decision = cicids_mini_matrix.write_second_dataset_decision(tmp_path)

    assert decision["maltls22"]["source_verified"] is False
    assert decision["maltls22"]["allowed_for_d5"] is False
    assert decision["default_recommendation"] == "CESNET-TLS-Year22"
    assert "CESNET-TLS22" in decision["recommended_replacements"]
    assert (tmp_path / "second_dataset_decision.json").exists()
    assert (tmp_path / "second_dataset_decision.md").exists()


def test_readiness_keeps_d5_blocked_after_cicids_component_ready(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "realdata_readiness_report.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "cicids2017": {},
                    "maltls22": {"source_verified": False},
                    "optc": {"available": False},
                }
            }
        ),
        encoding="utf-8",
    )
    gate = {"passed": True}

    cicids_mini_matrix.update_readiness_after_mini(gate, reports)
    readiness = json.loads((reports / "realdata_readiness_report.json").read_text(encoding="utf-8"))

    assert readiness["datasets"]["cicids2017"]["mini_matrix_passed"] is True
    assert readiness["datasets"]["cicids2017"]["ready_for_d5_component"] is True
    assert readiness["datasets"]["maltls22"]["source_verified"] is False
    assert readiness["datasets"]["optc"]["available"] is False
    assert readiness["d5_allowed"] is False
