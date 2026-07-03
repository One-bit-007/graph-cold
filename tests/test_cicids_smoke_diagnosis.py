from types import SimpleNamespace

import numpy as np

from src.diagnostics import cicids_smoke_diagnosis


def _bundle():
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=0.0, scale=0.2, size=(18, 6))
    X1 = rng.normal(loc=3.0, scale=0.2, size=(18, 6))
    X2 = rng.normal(loc=6.0, scale=0.2, size=(18, 6))
    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.repeat([0, 1, 2], 18)
    return SimpleNamespace(
        X_train=X,
        y_train=y,
        X_test=X.copy(),
        y_test=y.copy(),
        num_classes=3,
        meta={
            "feature_names": ["src_port", "flow_bytes", "duration", "iat", "pkt_len", "rate"],
            "class_counts": {0: 18, 1: 18, 2: 18},
            "label_mapping": {"BENIGN": 0, "DoS": 1, "PortScan": 2},
            "benign_class": 0,
            "expected_view_support": {
                "host": True,
                "ip": True,
                "temporal": True,
                "process": False,
                "threat_intel": False,
            },
        },
    )


def test_diagnosis_report_records_shared_noise_and_active_views(tmp_path, monkeypatch):
    audit = SimpleNamespace(dataset_hash="unit-hash", class_count=3)
    monkeypatch.setattr(cicids_smoke_diagnosis, "audit_dataset", lambda contract: audit)
    monkeypatch.setattr(cicids_smoke_diagnosis, "load_dataset", lambda dataset, cfg: _bundle())

    report = cicids_smoke_diagnosis.run_diagnosis(configs="configs", out=tmp_path)

    assert report["dataset_hash"] == "unit-hash"
    assert report["noise"]["seed_42_reproducible"] is True
    assert report["fairness_protocol"]["same_noisy_y_train"] is True
    assert report["fairness_protocol"]["same_flip_mask"] is True
    assert report["active_view_audit"]["active_views"] == ["host", "ip", "temporal"]
    assert report["active_view_audit"]["inactive_views"] == ["process", "threat_intel"]
    assert report["active_view_audit"]["process_misused"] is False
    assert (tmp_path / "cicids_smoke_failure_diagnosis.json").exists()
    assert (tmp_path / "cicids_smoke_failure_diagnosis.md").exists()
