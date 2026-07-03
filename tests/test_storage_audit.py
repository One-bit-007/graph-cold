from pathlib import Path

from scripts import check_storage


def test_storage_audit_allows_large_free_volume(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(check_storage.shutil, "disk_usage", lambda path: type("DU", (), {"free": 100 * 1024**3})())

    report = check_storage.run(tmp_path / "graphcold-data", required_gb=80, reports=tmp_path / "reports")

    assert report["download_allowed"] is True
    assert report["blocking_reason"] is None
    assert (tmp_path / "reports" / "storage_audit_report.json").exists()


def test_storage_audit_blocks_small_volume(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(check_storage.shutil, "disk_usage", lambda path: type("DU", (), {"free": 10 * 1024**3})())

    report = check_storage.run(tmp_path / "graphcold-data", required_gb=80, reports=tmp_path / "reports")

    assert report["download_allowed"] is False
    assert "below required" in report["blocking_reason"]
