from pathlib import Path

import numpy as np

from src.enterprise.optc_case import EnterpriseBaselineAdapter, run_case


def test_enterprise_baseline_adapter_fallback_predicts_probabilities():
    X = np.array([[0.0, 0.1], [1.0, 0.9], [0.2, 0.2], [0.8, 1.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])

    adapter = EnterpriseBaselineAdapter(baseline="flash", backend="sklearn", seed=42).fit(X, y)
    proba = adapter.predict_proba(X)

    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))
    assert set(adapter.predict(X)).issubset({0, 1})


def test_optc_synthetic_case_runs_five_views_d_chain_and_reports(tmp_path: Path):
    result = run_case({"backend": "sklearn", "top_k": 4, "lambda_chain": 0.1}, out_dir=tmp_path)
    report = result.report

    assert report["mode"] == "synthetic"
    assert report["num_events"] > 0
    assert report["num_hosts"] > 0
    assert report["num_processes"] > 0
    assert report["num_ips"] > 0
    assert report["five_views_non_empty"] is True
    assert report["d_chain_enabled"] is True
    assert report["lambda4"] == 0.1
    assert report["graph_cdm_computed"] is True
    assert report["ranking_topk_generated"] is True
    assert report["enterprise_baseline_adapter_ready"] is True
    assert result.ranking.shape == (4,)
    assert np.all(np.isfinite(result.graph_cdm))
    assert (tmp_path / "d4_optc_case_report.json").exists()
    assert (tmp_path / "d4_optc_case_report.md").exists()
