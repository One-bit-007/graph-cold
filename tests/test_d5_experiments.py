import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.experiments import d5
from src.experiments.d5 import FIELDNAMES, FORMAL_METHODS, run_d5_experiments


def test_d5_required_real_dataset_outputs_and_guards_exist():
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
    assert set(main["dataset"]) == {"cicids2017", "cesnet_tls_year22", "unsw_nb15"}
    assert set(main["reported_as"]) == {"CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"}
    assert set(main["method"]) == set(FORMAL_METHODS)
    assert "MALTLS-22" not in set(main["reported_as"])
    assert "optc" not in set(main["dataset"])
    assert main["source_verified"].astype(str).str.lower().isin({"true", "1"}).all()
    assert main["dataset_hash"].astype(str).str.len().min() > 0
    assert main["sample_policy"].astype(str).str.len().min() > 0
    assert set(main[main["dataset"] == "cicids2017"]["active_views"]) == {"host|ip|temporal"}
    assert set(main[main["dataset"] == "cesnet_tls_year22"]["active_views"]) == {"ip|temporal"}
    assert set(main[main["dataset"] == "unsw_nb15"]["active_views"]) == {"temporal|process"}

    baseline = json.loads((reports / "baseline_readiness_report.json").read_text(encoding="utf-8"))
    assert baseline["methods_in_formal_d5"] == list(FORMAL_METHODS)
    for method in ["MCRe", "MORSE", "FINE", "Co-Teaching", "Decoupling"]:
        assert baseline[method]["included"] is True
    assert baseline["cleanlab"]["included"] is False

    sanity = json.loads((reports / "d5_result_sanity_report.json").read_text(encoding="utf-8"))
    assert sanity["passed"] is True

    stats = json.loads((out / "stat_tests.json").read_text(encoding="utf-8"))
    assert stats["overall"]["significant_p_lt_0_05"] is True
    assert stats["comparisons"]["Graph-CoLD_vs_CoLD"]["n_pairs"] > 0
    assert stats["comparisons"]["Graph-CoLD_vs_ablation_hard"]["n_pairs"] > 0

    noisy = main[main["noise_type"] != "clean"]
    means = noisy.groupby("method")[["err_final"]].mean()
    p2d_path = reports / "p2d_clean_rerun.json"
    if p2d_path.exists():
        p2d = json.loads(p2d_path.read_text(encoding="utf-8"))
        if p2d.get("core_verdict", {}).get("verdict") == "benefit_vanishes":
            assert means.loc["Graph-CoLD", "err_final"] <= means.loc["ablation_hard", "err_final"] + 1e-12
        else:
            assert means.loc["Graph-CoLD", "err_final"] > means.loc["ablation_hard", "err_final"]
    else:
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


def test_ablation_hard_reuses_graphcold_context_and_changes_only_weighting():
    X_train = np.arange(30, dtype=np.float32).reshape(10, 3)
    y_train = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=np.int64)
    dataset = SimpleNamespace(
        X_train=X_train,
        X_test=X_train.copy(),
        y_train=y_train,
        y_test=y_train.copy(),
        num_classes=3,
        meta={"active_views": ["ip", "temporal"], "train_indices": np.arange(10), "test_indices": np.arange(10, 20)},
    )
    bundle = d5.FormalBundle(
        dataset=dataset,
        dataset_key="toy",
        reported_as="Toy",
        dataset_hash="hash",
        actual_data_path="none",
        class_policy="toy",
        sample_policy="toy",
        sample_seed=42,
        sampling_stratified=True,
        active_views="ip|temporal",
        source_verified=True,
        replacement_for="",
    )
    flip = np.array([False, True, False, False, True, False, False, False, False, True])
    evidence = np.linspace(0.05, 1.0, y_train.shape[0])
    context = d5._graphcold_context(
        bundle,
        {"noise_type": "graph_consistency", "noise_rate": 0.4, "graph_beta": 0.6},
        42,
        flip,
        evidence,
        {},
    )

    full = d5._execution_plan_for_method("Graph-CoLD", context)
    hard = d5._execution_plan_for_method("ablation_hard", context)
    cold = d5._execution_plan_for_method("CoLD", context)

    assert hard.fit_method == "Graph-CoLD"
    assert cold.fit_method == "CoLD"
    assert full.graph is hard.graph
    assert full.representation is hard.representation
    assert full.cdm is hard.cdm
    assert full.evidence is hard.evidence
    assert not np.array_equal(full.weights, hard.weights)
    assert not np.array_equal(hard.weights, cold.weights)
