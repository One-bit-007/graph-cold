from pathlib import Path
from types import SimpleNamespace

from src.experiments import smoke_realdata


def test_cesnet_smoke_missing_data_blocks_and_does_not_create_results(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    (Path("configs") / "datasets.yaml").write_text(
        """
cesnet_tls_year22:
  path: data/tls_alternative/cesnet_tls_year22
  label_col: service
""",
        encoding="utf-8",
    )
    audit = SimpleNamespace(ready_for_smoke=False, dataset_hash=None, blocking_reasons=["dataset root does not exist"])
    monkeypatch.setattr(smoke_realdata, "audit_dataset", lambda contract: audit)

    report = smoke_realdata.run_smoke_realdata("cesnet_tls_year22", configs="configs", out="reports")

    assert report["status"] == "blocked"
    assert report["passed"] is False
    assert Path("reports/cesnet_smoke_report.json").exists()
    assert Path("reports/cesnet_smoke_report.md").exists()
    assert not Path("results/cesnet_smoke_realdata.csv").exists()
