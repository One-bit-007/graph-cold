import json
from pathlib import Path

import pandas as pd


def test_d5_required_outputs_and_ck6_conditions_exist():
    out = Path("results")
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
    assert {"cicids2017", "maltls22", "optc_synthetic"}.issubset(set(main["dataset"]))

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
