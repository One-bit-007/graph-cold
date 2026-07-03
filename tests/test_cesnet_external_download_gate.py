from pathlib import Path
import zipfile

from scripts import download_tls_alternative
from src.data.audit import DatasetAuditResult


def _fake_audit(name: str) -> DatasetAuditResult:
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


def test_cesnet_auto_download_uses_external_cache_and_out(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "graphcold-data"
    cache = data_root / "_downloads"
    out = data_root / "tls_alternative" / "cesnet_tls_year22"
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("export/service.csv", "timestamp,service,flow_duration,bytes\n2022-01-01,a,1,10\n")

    monkeypatch.setattr(
        download_tls_alternative,
        "_zenodo_manifest",
        lambda url: [{"key": "CESNET-TLS-Year22.zip", "size": source_zip.stat().st_size, "url": "https://example.invalid/file"}],
    )
    monkeypatch.setattr(download_tls_alternative.shutil, "disk_usage", lambda path: type("DU", (), {"free": 100 * 1024**3})())
    monkeypatch.setattr(download_tls_alternative, "_download_file", lambda url, target, expected_size=0: target.write_bytes(source_zip.read_bytes()))
    monkeypatch.setattr(download_tls_alternative, "audit_all_datasets", lambda: {name: _fake_audit(name) for name in ("cicids2017", "maltls22", "optc")})
    monkeypatch.setattr(download_tls_alternative, "write_audit_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "write_readiness_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "write_dataset_specific_audit_report", lambda *args, **kwargs: {})

    report = download_tls_alternative.run(
        "cesnet_tls_year22",
        "auto",
        out,
        confirm_large_download=True,
        data_root=data_root,
        download_cache=cache,
        min_free_gb=0,
    )

    assert report["download_success"] is True
    assert (cache / "cesnet_tls_year22" / "CESNET-TLS-Year22.zip").exists()
    assert (out / "export" / "service.csv").exists()
    assert "data/maltls22" not in "\n".join(report.get("files_present", []))
