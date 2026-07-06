import hashlib
import json
from pathlib import Path

import pandas as pd

from src.paper.d6_prep import run_d6_realdata_prep


def _ensure_d6_outputs():
    if not Path("reports/d6/d6_generation_manifest.json").exists():
        run_d6_realdata_prep()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_table_2_is_grouped_from_expanded_results():
    _ensure_d6_outputs()
    main = pd.read_csv("results/table_main_expanded.csv")
    table2 = pd.read_csv("tables/table_2_main_performance.csv")

    expected_groups = main.groupby(
        ["reported_as", "noise_type", "noise_rate", "graph_beta", "method"],
        dropna=False,
    ).ngroups

    assert len(table2) == expected_groups
    assert set(table2["Dataset"]) == {"CICIDS-2017", "CESNET-TLS-Year22"}
    assert {
        "Graph-CoLD",
        "CoLD",
        "ablation_hard",
        "Noisy-Supervised",
        "Confident-Learning",
        "Co-Teaching",
        "Decoupling",
        "FINE",
        "MCRe",
        "MORSE",
    }.issubset(set(table2["Method"]))
    assert "Macro-F1 mean +/- std" in table2.columns


def test_dataset_protocol_table_reports_real_scope_and_sample_policy():
    _ensure_d6_outputs()
    table1 = pd.read_csv("tables/table_1_dataset_protocol.csv")

    assert list(table1["Dataset"]) == ["CICIDS-2017", "CESNET-TLS-Year22"]
    assert set(table1["Reported name"]) == {"CICIDS-2017", "CESNET-TLS-Year22"}
    assert table1["Sample policy"].astype(str).str.len().min() > 0
    assert table1["Source verified"].astype(str).str.lower().eq("true").all()
    assert "MALTLS-22" not in set(table1["Dataset"])
    assert "OpTC" not in set(table1["Dataset"])


def test_ablation_and_statistical_tables_have_formal_rows_only():
    _ensure_d6_outputs()
    table4 = pd.read_csv("tables/table_4_ablation_evidence.csv")
    table5 = pd.read_csv("tables/table_5_statistical_tests.csv")

    expected_variants = {
        "Graph-CoLD-full",
        "ablation_hard",
        "Graph-CoLD-no-D_neigh",
        "Graph-CoLD-no-D_view",
        "Graph-CoLD-no-evidence",
    }
    assert expected_variants == set(table4["Variant"])
    assert "Graph-CoLD-w=1" not in set(table4["Variant"])
    assert set(table5["Comparison"]) == {
        "Graph-CoLD vs CoLD",
        "Graph-CoLD vs ablation_hard",
        "Graph-CoLD vs Noisy-Supervised",
        "Graph-CoLD vs Confident-Learning",
        "Graph-CoLD vs Co-Teaching",
        "Graph-CoLD vs Decoupling",
        "Graph-CoLD vs FINE",
        "Graph-CoLD vs MCRe",
        "Graph-CoLD vs MORSE",
    }
    assert table5["Test type"].str.contains("paired grouped").all()
    assert not table5["Test type"].str.contains("pooled", case=False).any()


def test_d6_manifest_traces_to_table_main_expanded():
    _ensure_d6_outputs()
    manifest = json.loads(Path("reports/d6/d6_generation_manifest.json").read_text(encoding="utf-8"))
    narrative = json.loads(Path("reports/d6/d6_statistical_narrative.json").read_text(encoding="utf-8"))
    source = Path("results/table_main_expanded.csv")

    assert manifest["source_csv"] == "results/table_main_expanded.csv"
    assert narrative["source_csv"] == "results/table_main_expanded.csv"
    assert manifest["source_sha256"] == _sha256(source)
    assert narrative["source_sha256"] == _sha256(source)
