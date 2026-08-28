"""P7/P1-1 guards: keep CoLD source identity and numeric provenance honest."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REPORT = Path("reports/p7_cold_fidelity.md")
META = Path("reports/p7_cold_fidelity.json")


def test_0534_and_0688_are_different_internal_methods():
    frame = pd.read_csv("results/p2f_tail_powered.csv")
    cicids = frame[frame["reported_as"] == "CICIDS-2017"]
    means = cicids.groupby("method")["macro_f1"].mean()
    assert means["CoLD"] == pytest.approx(0.5344923896467946)
    assert means["ablation_hard"] == pytest.approx(0.6884972364833267)
    assert means["ablation_hard"] - means["CoLD"] == pytest.approx(0.1540048468365321)


def test_frozen_cold_row_uses_proxy_path_not_legacy_model_class():
    d5 = Path("src/experiments/d5.py").read_text(encoding="utf-8")
    p2e = Path("src/paper/p2e_salvage.py").read_text(encoding="utf-8")
    p2f = Path("src/paper/p2f_tighten.py").read_text(encoding="utf-8")
    assert "def _weights_for_cold" in d5
    assert "keep = cdm <= 0.5" in d5
    assert "np.quantile(retained_evidence, 0.99)" in d5
    assert "return d5._weights_for_cold(cdm, evidence)" in p2e
    assert "from src.models.cold_baseline import CoLD" not in p2e + p2f


def test_report_does_not_claim_official_reproduction():
    meta = json.loads(META.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")
    assert meta["resolution_path"] == "b_protocol_difference_attribution"
    assert meta["official_reproduction_completed"] is False
    assert "CoLD-aligned proxy" in report
    assert "未完成、也未声称完成官方 Table IV 复现" in report
    assert meta["sources"]["official_code_commit"] == "f0b940affe521662667562888d29468279d420a8"


def test_legacy_model_docstring_no_longer_claims_no_official_code():
    source = Path("src/models/cold_baseline.py").read_text(encoding="utf-8")
    assert "no official code exists" not in source
    assert "not an official-code reproduction" in source
