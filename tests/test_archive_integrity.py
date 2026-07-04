from pathlib import Path
import zipfile

from scripts import download_tls_alternative


def _skip_global_audit(monkeypatch) -> None:
    monkeypatch.setattr(download_tls_alternative, "audit_all_datasets", lambda: {})
    monkeypatch.setattr(download_tls_alternative, "write_audit_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "write_readiness_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "_write_tls_reports", lambda *args, **kwargs: None)


def test_verify_archive_accepts_complete_zip(tmp_path: Path, monkeypatch):
    _skip_global_audit(monkeypatch)
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data/file.csv", "timestamp,service,bytes\n2022-01-01,a,10\n")

    report = download_tls_alternative.run("cesnet_tls_year22", "verify-archive", tmp_path / "out", archive=archive)

    assert report["archive_verified"] is True
    assert report["download_success"] is True
    assert report["archive_sha256"]
    assert report["archive_summary"]["file_count"] == 1


def test_verify_archive_rejects_aria2_sidecar(tmp_path: Path, monkeypatch):
    _skip_global_audit(monkeypatch)
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data/file.csv", "timestamp,service,bytes\n2022-01-01,a,10\n")
    Path(str(archive) + ".aria2").write_text("incomplete", encoding="utf-8")

    report = download_tls_alternative.run("cesnet_tls_year22", "verify-archive", tmp_path / "out", archive=archive)

    assert report["archive_verified"] is False
    assert report["aria2_control_file_exists"] is True
    assert "Incomplete download sidecar" in report["error"]


def test_local_archive_blocks_when_verify_fails(tmp_path: Path, monkeypatch):
    _skip_global_audit(monkeypatch)
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data/file.csv", "timestamp,service,bytes\n2022-01-01,a,10\n")
    archive.with_suffix(archive.suffix + ".part").write_text("incomplete", encoding="utf-8")

    report = download_tls_alternative.run("cesnet_tls_year22", "local-archive", tmp_path / "out", archive=archive)

    assert report["download_success"] is False
    assert not (tmp_path / "out" / "data" / "file.csv").exists()
