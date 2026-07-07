import json
from pathlib import Path

import pandas as pd

from src.paper.p2_status import STRONG_BASELINES, generate_p2_status


def _ensure_p2_outputs() -> None:
    if not Path("reports/p2_baseline_sanity_report.json").exists():
        generate_p2_status()


def test_p2_strong_baselines_clear_clean_label_floor():
    _ensure_p2_outputs()
    table = pd.read_csv("tables/table_p2_baseline_sanity.csv")
    report = json.loads(Path("reports/p2_baseline_sanity_report.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert set(table["method"]) == set(STRONG_BASELINES)
    assert table["passes_clean_floor"].astype(str).str.lower().eq("true").all()
    assert (table["clean_macro_f1"] >= table["clean_floor"]).all()


def test_p2_baseline_margins_are_recomputed_after_calibration():
    _ensure_p2_outputs()
    report = json.loads(Path("reports/p2_baseline_sanity_report.json").read_text(encoding="utf-8"))
    margins = pd.read_csv("tables/table_p2_baseline_margin_after.csv")
    before = {row["method"]: row for row in report["before_snapshot"]["rows"]}

    assert set(margins["baseline"]) == set(STRONG_BASELINES)
    assert (margins["paired_rows"] > 0).all()
    for _, row in margins.iterrows():
        method = row["baseline"]
        assert row["macro_f1_margin_graphcold_minus_baseline"] < before[method]["macro_f1_margin_before"]


def test_p2_strong_baseline_refresh_report_documents_improvement():
    _ensure_p2_outputs()
    refresh = json.loads(Path("reports/p2_strong_baseline_refresh.json").read_text(encoding="utf-8"))

    assert set(refresh["methods_refreshed"]) == set(STRONG_BASELINES)
    assert refresh["rows_refreshed"] > 0
    for method, after in refresh["after"].items():
        before = refresh["before"][method]
        assert after["clean_macro_f1"] >= before["clean_macro_f1"]
