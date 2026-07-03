from types import SimpleNamespace

from src.data import cesnet_policy


def test_cesnet_postfilter_policy_removes_rare_classes_and_downsamples():
    kept, removed, rule = cesnet_policy.postfilter_counts({"svc_a": 10, "svc_b": 4, "rare": 1}, min_count=2)

    assert kept == {"svc_a": 4, "svc_b": 4}
    assert removed == {"rare": 1}
    assert "Downsample dominant class" in rule


def test_cesnet_policy_report_blocks_when_data_missing(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "datasets.yaml").write_text(
        """
cesnet_tls_year22:
  path: missing
  label_col: service
  min_class_count: 1000
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cesnet_policy,
        "audit_dataset",
        lambda contract: SimpleNamespace(dataset_hash=None, ready_for_smoke=False, blocking_reasons=["dataset root does not exist"]),
    )

    report = cesnet_policy.audit_policies(tmp_path / "configs", tmp_path / "reports")

    assert report["selected_policy"] == "postfilter"
    assert report["reported_as"] == "CESNET-TLS-Year22"
    assert report["ready_for_smoke"] is False
    assert (tmp_path / "reports" / "cesnet_class_policy_report.json").exists()
    assert (tmp_path / "reports" / "cesnet_view_policy_report.json").exists()
