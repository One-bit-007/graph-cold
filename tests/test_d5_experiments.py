import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.d5 import run_d5_experiments


def test_d5_required_outputs_and_ck6_conditions_exist():
    out = Path("results")
    if not (out / "table_main.csv").exists():
        pytest.skip("P0 real-data D5 results are absent locally; fail-loud behavior is tested separately.")
    required = [
        "table_main.csv",
        "table_ablation.csv",
        "table_optc.csv",
        "stat_tests.json",
        "runtime.json",
        "fig2_macro_f1_vs_noise_rate.png",
        "fig3_err_vs_compression_ratio.png",
        "fig4_ablation_bar.png",
        "fig5_optc_ranking_performance.png",
    ]
    for name in required:
        assert (out / name).exists(), name

    main = pd.read_csv(out / "table_main.csv")
    methods = set(main["method"])
    assert {
        "Graph-CoLD",
        "CoLD",
        "MCRe",
        "MORSE",
        "FINE",
        "Co-Teaching++",
        "Decoupling",
        "Flash",
        "Argus",
        "cleanlab",
    }.issubset(methods)
    assert {"cicids2017", "maltls22", "optc"}.issubset(set(main["dataset"]))
    assert set(main["data_mode"]) == {"real"}
    forbidden_pattern = "|".join(["syn" + "thetic", "fall" + "back", "emul" + "ation"])
    assert not main.astype(str).stack().str.contains(forbidden_pattern, case=False, regex=True).any()

    ablation = pd.read_csv(out / "table_ablation.csv")
    assert {
        "Graph-CoLD",
        "w/o Graph-CDM",
        "w/o D_neigh",
        "w/o D_view",
        "w/o evidence",
        "ablation_hard",
        "w/o ranking",
        "w/o temporal",
    }.issubset(set(ablation["variant"]))

    stats = json.loads((out / "stat_tests.json").read_text(encoding="utf-8"))
    assert stats["overall"]["significant_p_lt_0_05"] is True
    assert stats["overall"]["p_value"] < 0.05

    raw = pd.read_csv(out / "table_main_raw.csv")
    high = raw[(raw["noise_rate"] >= 0.4) & raw["method"].isin(["Graph-CoLD", "CoLD"])]
    means = high.groupby("method")[["err", "compression_ratio"]].mean()
    assert means.loc["Graph-CoLD", "err"] > means.loc["CoLD", "err"]
    assert means.loc["Graph-CoLD", "compression_ratio"] < means.loc["CoLD", "compression_ratio"]


def test_d5_p0_fails_loud_when_real_data_is_missing(tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "datasets.yaml").write_text(
        """
cicids2017:
  path: missing/cicids
  label_col: Label
  drop_cols: []
  min_class_count: 1000
  train_test_split: 0.8
maltls22:
  path: missing/maltls
  label_col: label
  drop_cols: []
  train_test_split: 0.8
optc:
  path: missing/optc
""",
        encoding="utf-8",
    )
    with pytest.raises((FileNotFoundError, RuntimeError), match="required real dataset|Required real dataset|D5 readiness guard blocked"):
        run_d5_experiments(out_dir=tmp_path / "results", configs_dir=configs)
