from types import SimpleNamespace

import pytest

from src.data import cicids_policy


def test_postfilter11_policy_drops_lt_1000_and_downsamples_dominant():
    raw = {
        "BENIGN": 5000,
        "DDoS": 2000,
        "PortScan": 1500,
        "Heartbleed": 11,
    }

    kept, removed, rule = cicids_policy.postfilter11_counts(raw, min_count=1000)

    assert kept == {"BENIGN": 2000, "DDoS": 2000, "PortScan": 1500}
    assert removed == {"Heartbleed": 11}
    assert "Downsample dominant class" in rule


def test_refined9_without_explicit_mapping_is_not_default_enabled():
    report = cicids_policy.refined9_audit({})

    assert report["is_default_enabled"] is False
    assert report["deterministic_from_cicids_labels"] is False
    assert report["class_names"] == []


def test_cicids_policy_report_selects_postfilter11(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "datasets.yaml").write_text(
        """
cicids2017:
  path: data/cicids2017
  label_col: Label
  min_class_count: 1000
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cicids_policy,
        "_read_label_counts",
        lambda root, label_col: {
            "BENIGN": 5000,
            "DDoS": 2000,
            "PortScan": 1500,
            "Heartbleed": 11,
        },
    )
    monkeypatch.setattr(cicids_policy, "audit_dataset", lambda contract: SimpleNamespace(dataset_hash="unit-hash"))

    report = cicids_policy.audit_policies(tmp_path / "configs", tmp_path / "reports")

    assert report["selected_policy"] == "postfilter11"
    assert report["policies"]["raw15"]["has_lt_1000_classes"] is True
    assert report["policies"]["postfilter11"]["consistent_with_current_smoke"] is False
    assert report["policies"]["refined9"]["is_default_enabled"] is False
    assert (tmp_path / "reports" / "cicids_class_policy_report.json").exists()
    assert (tmp_path / "reports" / "cicids_class_policy_report.md").exists()


def test_refined9_mapping_guard_rejects_default_selection(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    mapping = "\n".join([f"    raw{i}: refined{i}" for i in range(9)])
    (tmp_path / "configs" / "datasets.yaml").write_text(
        f"""
cicids2017:
  path: data/cicids2017
  label_col: Label
  min_class_count: 1000
  refined9_mapping:
{mapping}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(cicids_policy, "_read_label_counts", lambda root, label_col: {"BENIGN": 5000, "DDoS": 2000})
    monkeypatch.setattr(cicids_policy, "audit_dataset", lambda contract: SimpleNamespace(dataset_hash="unit-hash"))

    with pytest.raises(ValueError, match="refined9 cannot be selected"):
        cicids_policy.audit_policies(tmp_path / "configs")
