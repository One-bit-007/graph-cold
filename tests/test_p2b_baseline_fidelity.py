import json
from pathlib import Path

import pandas as pd

from src.analysis.protocol import PROTOCOL_ID, source_hash
from src.paper.p2b_baseline_fidelity import generate_p2b_baseline_fidelity


def _ensure_p2b_outputs() -> dict:
    if not Path("reports/p2b_baseline_fidelity.json").exists():
        generate_p2b_baseline_fidelity()
    return json.loads(Path("reports/p2b_baseline_fidelity.json").read_text(encoding="utf-8"))


def test_p2b_per_rate_table_covers_real_scope_and_focus_methods():
    report = _ensure_p2b_outputs()
    table = pd.read_csv("tables/table_p2b_baseline_noise_robustness.csv")

    assert report["protocol_id"] == PROTOCOL_ID
    assert report["source_sha256"] == source_hash("results/table_main_expanded.csv")
    assert report["robustness_table"] == "tables/table_p2b_baseline_noise_robustness.csv"
    assert set(table["reported_as"]) == {"CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"}
    assert set(table["method"]) == {"MCRe", "MORSE", "CoLD", "Co-Teaching", "Graph-CoLD"}
    assert {0.0, 0.1, 0.2, 0.4, 0.6}.issubset(set(table["noise_rate"].round(1)))
    assert len(table) == 3 * 17 * 5
    assert table["seeds"].min() == 3


def test_p2b_reports_protocol_explained_outcome_without_changing_numbers():
    report = _ensure_p2b_outputs()
    diagnosis = report["diagnosis"]

    assert report["outcome"] == "B_protocol_explained"
    assert report["result_numbers_changed"] is False
    assert report["canonical_tables_updated"] is False
    assert diagnosis["adapter_bug_found"] is False
    assert diagnosis["morse_split_ratio_tracks_actual_noise_rate"] is True
    assert diagnosis["mcre_retain_fraction_tracks_actual_noise_rate"] is True
    assert diagnosis["mcre_tail_or_clean_informative_collapse_observed"] is True
    assert diagnosis["morse_retains_evidence_but_classifier_degrades"] is True
    assert "not claimed to reproduce the original papers" in report["manuscript_caveat"]


def test_p2b_key_numbers_explain_cicids_vs_cesnet_behavior():
    _ensure_p2b_outputs()
    table = pd.read_csv("tables/table_p2b_baseline_noise_robustness.csv")
    sym60 = table[
        (table["noise_type"] == "symmetric")
        & (table["graph_beta"].astype(str) == "none")
        & (table["noise_rate"].round(1) == 0.6)
    ]

    cicids_mcre = sym60[(sym60["reported_as"] == "CICIDS-2017") & (sym60["method"] == "MCRe")].iloc[0]
    cicids_morse = sym60[(sym60["reported_as"] == "CICIDS-2017") & (sym60["method"] == "MORSE")].iloc[0]
    cesnet_mcre = sym60[(sym60["reported_as"] == "CESNET-TLS-Year22") & (sym60["method"] == "MCRe")].iloc[0]
    cesnet_morse = sym60[(sym60["reported_as"] == "CESNET-TLS-Year22") & (sym60["method"] == "MORSE")].iloc[0]

    assert cicids_mcre["macro_f1_mean"] < 0.4
    assert cicids_mcre["retained_clean_informative_mean"] < 0.5
    assert cicids_morse["macro_f1_mean"] < 0.4
    assert cicids_morse["retained_clean_informative_mean"] > 0.7
    assert cesnet_mcre["macro_f1_mean"] > 0.8
    assert cesnet_morse["macro_f1_mean"] > 0.8


def test_p2b_markdown_report_contains_required_sections_and_commands():
    _ensure_p2b_outputs()
    text = Path("reports/p2b_baseline_fidelity.md").read_text(encoding="utf-8")

    for heading in [
        "## Per-rate Table",
        "## Diagnosis",
        "## Manuscript Caveat",
        "## Updated Canonical Margins",
        "## Reproduction Commands",
    ]:
        assert heading in text
    assert "python -m src.paper.p2b_baseline_fidelity" in text
