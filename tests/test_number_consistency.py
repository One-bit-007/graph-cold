import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.protocol import PROTOCOL_ID, headline_table, source_hash
from src.paper.p2_status import generate_p2_status


def _ensure_p2_outputs() -> None:
    if not Path("reports/p2_status.json").exists():
        generate_p2_status()


def test_p2_protocol_hash_and_headlines_match_source_results():
    _ensure_p2_outputs()
    source = Path("results/table_main_expanded.csv")
    report = json.loads(Path("reports/p2_number_consistency.json").read_text(encoding="utf-8"))
    headline = pd.read_csv("tables/table_p2_canonical_headline.csv")
    expected = headline_table(pd.read_csv(source))

    assert report["protocol_id"] == PROTOCOL_ID
    assert report["source_csv"] == "results/table_main_expanded.csv"
    assert report["source_sha256"] == source_hash(source)
    assert set(headline["method"]) == set(expected["method"])
    for method in expected["method"]:
        got = headline.loc[headline["method"] == method, "macro_f1_mean"].iloc[0]
        want = expected.loc[expected["method"] == method, "macro_f1_mean"].iloc[0]
        assert np.isclose(got, want)


def test_paper_tables_use_same_canonical_headline_number_per_method():
    _ensure_p2_outputs()
    headline = pd.read_csv("tables/table_p2_canonical_headline.csv")
    canonical = dict(zip(headline["method"], headline["macro_f1_mean"]))

    for table_path in ["tables/table_2_main_performance.csv", "tables/table_3_high_noise_summary.csv"]:
        table = pd.read_csv(table_path)
        assert "Canonical Macro-F1 headline" in table.columns
        assert "Canonical protocol" in table.columns
        assert set(table["Canonical protocol"]) == {PROTOCOL_ID}
        for method, part in table.groupby("Method", dropna=False):
            values = pd.to_numeric(part["Canonical Macro-F1 headline"], errors="raise").to_numpy()
            assert np.allclose(values, canonical[method], atol=1e-6)


def test_p2_number_consistency_audit_passed():
    _ensure_p2_outputs()
    audit = json.loads(Path("reports/p2_number_consistency_audit.json").read_text(encoding="utf-8"))

    assert audit["protocol_id"] == PROTOCOL_ID
    assert audit["passed"] is True
    assert audit["source_sha256"] == source_hash("results/table_main_expanded.csv")
    assert all(item["passed"] for item in audit["checks"])
