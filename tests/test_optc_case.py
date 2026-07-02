from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.enterprise.optc_case import EnterpriseBaselineAdapter, run_case


def test_enterprise_baseline_adapter_predicts_probabilities():
    X = np.array([[0.0, 0.1], [1.0, 0.9], [0.2, 0.2], [0.8, 1.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])

    adapter = EnterpriseBaselineAdapter(baseline="flash", backend="sklearn", seed=42).fit(X, y)
    proba = adapter.predict_proba(X)

    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))
    assert set(adapter.predict(X)).issubset({0, 1})


def test_optc_missing_real_events_fails_loud(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Real OpTC provenance events are required"):
        run_case({"path": tmp_path / "missing", "backend": "sklearn", "top_k": 4, "lambda_chain": 0.1}, out_dir=tmp_path)


def test_optc_real_case_runs_five_views_d_chain_and_reports(tmp_path: Path):
    data_dir = tmp_path / "optc"
    data_dir.mkdir()
    rows = []
    base = pd.Timestamp("2026-01-01T00:00:00")
    for idx in range(24):
        malicious = 8 <= idx < 14
        rows.append(
            {
                "host_id": f"host-{idx % 3}",
                "process_id": f"proc-{idx % 5}",
                "parent_process_id": f"proc-{(idx - 1) % 5}",
                "src_ip": f"10.0.0.{idx % 7}",
                "dst_ip": "203.0.113.10" if malicious else f"10.1.0.{idx % 9}",
                "timestamp": (base + pd.Timedelta(minutes=idx * 3)).isoformat(),
                "event_type": ["process_start", "dns_query", "net_conn"][idx % 3],
                "alert_type": "credential_access" if malicious else "benign_activity",
                "label": int(malicious),
                "risk_score": 0.8 if malicious else 0.2,
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "events.csv", index=False)

    result = run_case({"path": data_dir, "backend": "sklearn", "top_k": 4, "lambda_chain": 0.1}, out_dir=tmp_path)
    report = result.report

    assert report["mode"] == "real"
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
