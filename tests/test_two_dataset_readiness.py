import json
from pathlib import Path

from src.experiments import cesnet_mini_matrix


def test_two_dataset_readiness_requires_cicids_and_cesnet(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "realdata_readiness_report.json").write_text(
        json.dumps({"datasets": {"cicids2017": {"ready_for_d5_component": True, "class_policy": "postfilter11", "mini_matrix_passed": True}}}),
        encoding="utf-8",
    )

    blocked = cesnet_mini_matrix.update_two_dataset_readiness({"passed": False}, reports)
    ready = cesnet_mini_matrix.update_two_dataset_readiness({"passed": True}, reports)

    assert blocked["d5_allowed"] is False
    assert ready["d5_allowed"] is True
    assert ready["d5_scope"] == ["CICIDS-2017", "CESNET-TLS-Year22"]
    assert ready["maltls22"]["source_verified"] is False
    assert ready["maltls22"]["evaluated"] is False
    assert ready["optc"]["formal_experiment"] is False


def test_d5_scope_decision_excludes_maltls_and_old_results(tmp_path: Path):
    decision = cesnet_mini_matrix.write_d5_scope_decision({"passed": True}, tmp_path)

    assert decision["maltls22_evaluated"] is False
    assert decision["optc_formal_experiment"] is False
    assert "MALTLS-22" in decision["future_d5_must_not_include"]
    assert decision["old_d5_d6_d7_results_invalid"] is True
    assert (tmp_path / "d5_scope_decision.json").exists()
