from pathlib import Path
import zipfile

from scripts import download_tls_alternative


def test_cesnet_verify_archive_blocks_resume_sidecars(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(download_tls_alternative, "audit_all_datasets", lambda: {})
    monkeypatch.setattr(download_tls_alternative, "write_audit_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "write_readiness_reports", lambda *args, **kwargs: {})
    monkeypatch.setattr(download_tls_alternative, "_write_tls_reports", lambda *args, **kwargs: None)
    archive = tmp_path / "CESNET-TLS-Year22.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("CESNET-TLS-Year22/sample.csv", "TIME_FIRST,APP,BYTES\n2022-01-01,a,1\n")
    Path(str(archive) + ".aria2").write_text("resume-control", encoding="utf-8")

    report = download_tls_alternative.run("cesnet_tls_year22", "verify-archive", tmp_path / "out", archive=archive)

    assert report["archive_verified"] is False
    assert report["download_complete"] is False
    assert report["aria2_control_file_exists"] is True
