from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.data.audit import audit_dataset, write_audit_reports
from src.data.contracts import CICIDS2017_CONTRACT, DatasetContract


def test_cicids_temporary_fixture_passes_minimal_audit(tmp_path: Path):
    data_dir = tmp_path / "cicids2017"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            " Source IP ": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            " Destination IP ": ["10.1.0.1", "10.1.0.2", "10.1.0.3", "10.1.0.4"],
            " Source Port ": [1000, 1001, 1002, 1003],
            " Destination Port ": [80, 443, 22, 53],
            " Protocol ": [6, 6, 6, 17],
            " Timestamp ": pd.date_range("2026-01-01", periods=4, freq="min").astype(str),
            " Flow Duration ": [1.0, 2.0, 3.0, 4.0],
            " Label ": ["BENIGN", "BENIGN", "DoS", "DoS"],
        }
    )
    frame.to_csv(data_dir / "sample.csv", index=False)
    contract = replace(
        CICIDS2017_CONTRACT,
        root=str(data_dir),
        expected_files=["sample.csv"],
        min_samples=4,
        min_classes=2,
    )

    result = audit_dataset(contract)

    assert result.ready_for_smoke is True
    assert result.ready_for_d5 is True
    assert result.label_column_present is True
    assert result.class_count == 2
    assert result.num_rows == 4
    assert result.dataset_hash
    assert result.required_any_columns_status["timestamp"] is True
    assert result.actual_view_support["temporal"] == "available"


def test_schema_missing_label_column_is_blocked(tmp_path: Path):
    data_dir = tmp_path / "bad"
    data_dir.mkdir()
    pd.DataFrame({"Source IP": ["10.0.0.1"], "feature": [1.0]}).to_csv(data_dir / "bad.csv", index=False)
    contract = DatasetContract(
        name="bad",
        root=str(data_dir),
        expected_files=["bad.csv"],
        label_column="Label",
        min_samples=1,
        min_classes=1,
        source_verified=True,
    )

    result = audit_dataset(contract)

    assert result.ready_for_d5 is False
    assert result.label_column_present is False
    assert any("label column missing" in reason for reason in result.blocking_reasons)


def test_audit_reports_generate_json_and_markdown(tmp_path: Path):
    data_dir = tmp_path / "cicids2017"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "Source IP": ["10.0.0.1", "10.0.0.2"],
            "Destination IP": ["10.1.0.1", "10.1.0.2"],
            "Protocol": [6, 6],
            "Label": ["BENIGN", "DoS"],
            "f": [1.0, 2.0],
        }
    ).to_csv(data_dir / "sample.csv", index=False)
    contract = replace(
        CICIDS2017_CONTRACT,
        root=str(data_dir),
        expected_files=["sample.csv"],
        min_samples=2,
        min_classes=2,
    )
    result = audit_dataset(contract)
    paths = write_audit_reports({"cicids2017": result}, out_dir=tmp_path / "reports")

    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert "cicids2017" in Path(paths["markdown"]).read_text(encoding="utf-8")

