from pathlib import Path

from src.experiments.smoke_realdata import run_smoke_realdata


def test_missing_real_data_smoke_is_blocked_and_does_not_create_results_csv(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    (Path("configs") / "datasets.yaml").write_text(
        """
cicids2017:
  path: data/cicids2017
  label_col: Label
  drop_cols: []
  min_class_count: 1000
  train_test_split: 0.8
""",
        encoding="utf-8",
    )

    report = run_smoke_realdata("cicids2017", configs="configs", out="reports")

    assert report["status"] == "blocked"
    assert report["passed"] is False
    assert not Path("results/smoke_realdata.csv").exists()
    assert Path("reports/smoke_realdata_report.json").exists()
    assert Path("reports/smoke_realdata_report.md").exists()


def test_smoke_gate_checks_audit_before_loading_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    (Path("configs") / "datasets.yaml").write_text("{}", encoding="utf-8")

    report = run_smoke_realdata("cicids2017", configs="configs", out="reports")

    assert report["status"] == "blocked"
    assert any("root does not exist" in reason for reason in report["blocking_reasons"])


def test_data_directory_is_ignored_by_git():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")

    assert "/data/" in text
    assert "/datasets/" in text


def test_experiment_gate_does_not_reintroduce_generated_result_paths():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "src" / "experiments" / "d5.py", root / "src" / "experiments" / "smoke_realdata.py"]
    blocked_terms = ["syn" + "thetic", "fall" + "back", "emul" + "ation", "draft " + "placeholder"]

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for term in blocked_terms:
            assert term not in text
