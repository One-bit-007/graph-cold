from pathlib import Path
import json

import pandas as pd
import pytest

from src.analysis.result_sanity import check_results


def test_reinforced_results_if_present_have_scope_and_honest_names():
    path = Path("results/table_main_reinforced.csv")
    if not path.exists():
        pytest.skip("D9.5 reinforced results have not been generated in this checkout.")
    frame = pd.read_csv(path)

    assert {"cicids2017", "cesnet_tls_year22"} == set(frame["dataset"].unique())
    assert "MALTLS-22" not in set(frame.get("reported_as", pd.Series(dtype=str)).astype(str))
    assert "OpTC" not in set(frame.get("reported_as", pd.Series(dtype=str)).astype(str))
    methods = set(frame["method"])
    assert "FINE" not in methods
    assert "Decoupling" in methods
    audit_path = Path("reports/d9_5/d9_5_final_audit.json")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("fine_style_smoke_passed"):
            assert "FINE-style" in methods
        else:
            assert "FINE-style" not in methods
    assert check_results(frame)["passed"] is True
