import json
from pathlib import Path

import pandas as pd


def test_d6_tables_figures_and_narrative_are_traceable():
    if not Path("results/stat_tests.json").exists():
        import pytest

        pytest.skip("P0 removed non-real D5 results; D6 paper prep resumes after real D5 rerun.")
    for path in [
        "tables/table_1_main_results.csv",
        "tables/table_2_ablation.csv",
        "tables/table_3_optc.csv",
        "figures/fig2_macro_f1_vs_noise_rate.png",
        "figures/fig3_err_vs_compression_ratio.png",
        "figures/fig4_ablation_drop_bar.png",
        "figures/fig5_optc_soc_ranking.png",
        "reports/d6_statistical_narrative.md",
        "reports/d6_paper_prep_report.json",
    ]:
        assert Path(path).exists(), path

    table1 = pd.read_csv("tables/table_1_main_results.csv")
    assert {"Method", "Macro-F1", "FPR", "FNR", "ERR", "Compression", "Runtime"}.issubset(table1.columns)
    assert "Graph-CoLD" in set(table1["Method"])

    table2 = pd.read_csv("tables/table_2_ablation.csv")
    assert {
        "Graph-CoLD",
        "w/o Graph-CDM",
        "w/o D_neigh",
        "w/o D_view",
        "w/o evidence",
        "ablation_hard",
        "w/o ranking",
        "w/o temporal",
    }.issubset(set(table2["Variant"]))

    table3 = pd.read_csv("tables/table_3_optc.csv")
    assert {"Top-K precision", "Compression", "ERR"}.issubset(table3.columns)

    narrative = Path("reports/d6_statistical_narrative.md").read_text(encoding="utf-8")
    stats = json.loads(Path("results/stat_tests.json").read_text(encoding="utf-8"))
    assert f"p={stats['overall']['p_value']:.2e}" in narrative
    assert "Conclusion-ready insight block" in narrative
