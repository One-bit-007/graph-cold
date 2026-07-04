import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.d5 import FIELDNAMES, FORMAL_METHODS, run_d5_experiments


def test_d5_required_real_two_dataset_outputs_and_guards_exist():
    out = Path("results")
    reports = Path("reports")
    if not (out / "table_main.csv").exists():
        pytest.skip("Real-data D5 results are absent locally; fail-loud behavior is tested separately.")

    for name in ["table_main.csv", "table_ablation.csv", "stat_tests.json", "runtime.json"]:
        assert (out / name).exists(), name
    for name in [
        "baseline_readiness_report.json",
        "d5_scale_policy.json",
        "d5_realdata_execution_report.json",
        "d5_result_sanity_report.json",
        "d5_statistical_validity_report.json",
    ]:
        assert (reports / name).exists(), name

    assert not (out / "table_optc.csv").exists()
    d6_manifest_path = reports / "d6" / "d6_generation_manifest.json"
    if d6_manifest_path.exists():
        d6_manifest = json.loads(d6_manifest_path.read_text(encoding="utf-8"))
        assert d6_manifest["source_csv"] == "results/table_main_expanded.csv"
        assert all(Path(path).exists() for path in d6_manifest["figures"])
        assert all(Path(path).exists() for path in d6_manifest["tables"])
    else:
        assert not Path("figures").exists()
        assert not Path("tables").exists()

    main = pd.read_csv(out / "table_main.csv")
    assert set(FIELDNAMES).issubset(main.columns)
    assert set(main["dataset"]) == {"cicids2017", "cesnet_tls_year22"}
    assert set(main["reported_as"]) == {"CICIDS-2017", "CESNET-TLS-Year22"}
    assert set(main["method"]) == set(FORMAL_METHODS)
    assert "MALTLS-22" not in set(main["reported_as"])
    assert "optc" not in set(main["dataset"])
    assert main["source_verified"].astype(str).str.lower().isin({"true", "1"}).all()
    assert main["dataset_hash"].astype(str).str.len().min() > 0
    assert main["sample_policy"].astype(str).str.len().min() > 0
    assert set(main[main["dataset"] == "cicids2017"]["active_views"]) == {"host|ip|temporal"}
    assert set(main[main["dataset"] == "cesnet_tls_year22"]["active_views"]) == {"ip|temporal"}

    baseline = json.loads((reports / "baseline_readiness_report.json").read_text(encoding="utf-8"))
    assert baseline["methods_in_formal_d5"] == list(FORMAL_METHODS)
    assert baseline["MCRe"]["included"] is False
    assert baseline["cleanlab"]["included"] is False

    sanity = json.loads((reports / "d5_result_sanity_report.json").read_text(encoding="utf-8"))
    assert sanity["passed"] is True

    stats = json.loads((out / "stat_tests.json").read_text(encoding="utf-8"))
    assert stats["overall"]["significant_p_lt_0_05"] is True
    assert stats["comparisons"]["Graph-CoLD_vs_CoLD"]["n_pairs"] > 0
    assert stats["comparisons"]["Graph-CoLD_vs_ablation_hard"]["n_pairs"] > 0

    noisy = main[main["noise_type"] != "clean"]
    means = noisy.groupby("method")[["err_final"]].mean()
    assert means.loc["Graph-CoLD", "err_final"] > means.loc["ablation_hard", "err_final"]


def test_d5_p0_fails_loud_when_real_data_is_missing(tmp_path: Path):
    configs = tmp_path / "configs"
    reports = tmp_path / "reports"
    configs.mkdir()
    reports.mkdir()
    (reports / "second_dataset_selection_gate.json").write_text(
        json.dumps({"d5_allowed": True, "d5_scope": ["cicids2017", "cesnet_tls_year22"]}),
        encoding="utf-8",
    )
    (configs / "datasets.yaml").write_text(
        """
cicids2017:
  path: missing/cicids
  label_col: Label
  drop_cols: []
  min_class_count: 1000
  train_test_split: 0.8
cesnet_tls_year22:
  path: missing/cesnet
  label_col: APP
  drop_cols: []
  min_class_count: 1000
  train_test_split: 0.8
""",
        encoding="utf-8",
    )
    with pytest.raises((FileNotFoundError, RuntimeError), match="Required real dataset|D5 readiness guard blocked"):
        run_d5_experiments(out_dir=tmp_path / "results", configs_dir=configs)
