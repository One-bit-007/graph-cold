from pathlib import Path
import zipfile

import pandas as pd
import pytest

from scripts import check_data_ready, download_cicids2017, download_optc, download_tls_alternative, prepare_datasets
from src.data.audit import DatasetAuditResult
from src.data.contracts import CICIDS2017_CONTRACT


@pytest.fixture(autouse=True)
def _avoid_real_dataset_global_audits(monkeypatch):
    def fake_result(name: str) -> DatasetAuditResult:
        return DatasetAuditResult(
            name=name,
            root=f"data/{name}",
            exists=False,
            expected_files_present=False,
            missing_files=[],
            files_used=[],
            file_hashes={},
            dataset_hash=None,
            num_rows=0,
            num_columns=0,
            label_column=None,
            label_column_present=False,
            class_count=0,
            label_distribution={},
            missing_values=0,
            infinite_values=0,
            duplicate_rows=0,
            numeric_feature_count=0,
            required_columns_ok=False,
            required_any_columns_status={},
            expected_view_support={},
            actual_view_support={},
            ready_for_smoke=False,
            ready_for_d5=False,
            blocking_reasons=["unit-test global audit skipped"],
        )

    fake_audits = {name: fake_result(name) for name in ("cicids2017", "maltls22", "optc")}
    monkeypatch.setattr(download_cicids2017, "audit_all_datasets", lambda: fake_audits)
    monkeypatch.setattr(download_optc, "audit_all_datasets", lambda: fake_audits)
    monkeypatch.setattr(prepare_datasets, "audit_all_datasets", lambda: fake_audits)
    monkeypatch.setattr(download_tls_alternative, "audit_all_datasets", lambda: fake_audits)
    monkeypatch.setattr(download_tls_alternative, "write_audit_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "write_dataset_specific_audit_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "write_readiness_reports", lambda *args, **kwargs: {})


def _write_cicids_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in CICIDS2017_CONTRACT.expected_files:
            csv = "Source IP,Destination IP,Source Port,Destination Port,Protocol,Timestamp,Flow Duration,Label\n"
            csv += "10.0.0.1,10.0.0.2,123,80,6,2026-01-01T00:00:00,1.0,BENIGN\n"
            csv += "10.0.0.3,10.0.0.4,124,443,6,2026-01-01T00:01:00,2.0,DoS\n"
            archive.writestr(f"MachineLearningCSV/{name}", csv)


def test_cicids_local_zip_extracts_fixture_zip(tmp_path: Path):
    archive = tmp_path / "MachineLearningCSV.zip"
    out = tmp_path / "cicids2017"
    _write_cicids_zip(archive)

    report = download_cicids2017.run("local-zip", out, archive)

    assert report["download_success"] is True
    assert len(report["files_present"]) == 8
    assert not report["missing_files"]
    for name in CICIDS2017_CONTRACT.expected_files:
        assert (out / name).exists()


def test_cicids_missing_zip_returns_blocked_report_without_fake_data(tmp_path: Path):
    out = tmp_path / "cicids2017"

    report = download_cicids2017.run("local-zip", out, tmp_path / "missing.zip")

    assert report["download_success"] is False
    assert report["manual_action_required"] is True
    assert not out.exists()
    assert len(report["files_present"]) == 0


def test_cicids_auto_failure_writes_instructions_and_uses_no_third_party(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(download_cicids2017, "discover_download_url", lambda: None)

    report = download_cicids2017.run("auto", tmp_path / "cicids2017")

    assert report["download_attempted"] is True
    assert report["download_success"] is False
    assert report["third_party_mirror_used"] is False
    assert report["manual_action_required"] is True
    assert report["manual_instructions"]


def test_tls_alternative_instructions_do_not_download_large_files_or_write_maltls(tmp_path: Path):
    out = tmp_path / "tls_alternative" / "cesnet_tls_year22"

    report = download_tls_alternative.run("cesnet_tls_year22", "instructions", out)

    assert report["download_attempted"] is False
    assert report["large_download_confirmed"] is False
    assert report["not_maltls22"] is True
    assert not (tmp_path / "maltls22").exists()


def test_tls_alternative_auto_requires_large_download_confirmation(tmp_path: Path):
    report = download_tls_alternative.run("cesnet_tls_year22", "auto", tmp_path / "tls")

    assert report["download_attempted"] is False
    assert report["download_success"] is False
    assert "requires --confirm-large-download" in report["error"]


def test_tls_alternative_auto_records_insufficient_disk_without_downloading(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        download_tls_alternative,
        "_zenodo_manifest",
        lambda url: [{"key": "CESNET-TLS-Year22.zip", "size": 1000, "url": "https://example.invalid/file"}],
    )
    monkeypatch.setattr(download_tls_alternative.shutil, "disk_usage", lambda path: type("DU", (), {"free": 1})())

    report = download_tls_alternative.run("cesnet_tls_year22", "auto", tmp_path / "tls", confirm_large_download=True)

    assert report["download_attempted"] is True
    assert report["download_success"] is False
    assert "Insufficient free disk space" in report["error"]
    assert report["zenodo_files"][0]["key"] == "CESNET-TLS-Year22.zip"
    assert not (tmp_path / "tls" / "CESNET-TLS-Year22.zip").exists()


def test_optc_instructions_do_not_start_full_download(tmp_path: Path):
    report = download_optc.run("instructions", out=tmp_path / "optc")

    assert report["full_download_attempted"] is False
    assert report["events_csv_present"] is False
    assert report["audit_passed"] is False


def test_optc_local_events_missing_fields_fails_audit(tmp_path: Path):
    events = tmp_path / "events.csv"
    pd.DataFrame({"host_id": ["h1"], "label": [0]}).to_csv(events, index=False)

    report = download_optc.run("local-events", out=tmp_path / "optc", events=events)

    assert report["events_csv_present"] is True
    assert report["audit_passed"] is False
    assert any("missing required columns" in reason for reason in report["blocking_reasons"])


def test_prepare_datasets_entrypoint_can_run_instructions(tmp_path: Path):
    args = type(
        "Args",
        (),
        {
            "dataset": "cicids2017",
            "mode": "instructions",
            "out": str(tmp_path / "cicids2017"),
            "zip_path": None,
            "candidate": "cesnet_tls_year22",
            "events": None,
            "confirm_large_download": False,
        },
    )()

    report = prepare_datasets.run(args)

    assert report["dataset"] == "cicids2017"
    assert report["readiness_refreshed"] is True
    assert report["d5_full_matrix_invoked"] is False


def test_check_data_ready_script_runs():
    assert check_data_ready.main() == 0


def test_gitignore_covers_raw_data_extensions():
    text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

    for item in ["data/", "data_raw/", "*.pcap", "*.pcapng", "*.zip", "*.7z", "*.gz", "*.parquet", "*.csv"]:
        assert item in text
    assert "!tests/fixtures/**/*.csv" in text
