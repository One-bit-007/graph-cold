from pathlib import Path

import pandas as pd

from src.data.audit import audit_dataset
from src.data.contracts import DatasetContract
from src.data.loaders import load_dataset


def _fixture_csv(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for label, count in [("svc_a", 6), ("svc_b", 5), ("rare", 1)]:
        for idx in range(count):
            rows.append(
                {
                    "timestamp": f"2022-01-01T00:{idx:02d}:00",
                    "service": label,
                    "flow_duration": float(idx + 1),
                    "bytes_in": float(100 + idx),
                    "packet_count": float(5 + idx),
                    "sni_name": f"{label}.example",
                }
            )
    pd.DataFrame(rows).to_csv(root / "cesnet_fixture.csv", index=False)


def test_cesnet_fixture_audit_can_pass(tmp_path: Path):
    root = tmp_path / "cesnet"
    _fixture_csv(root)
    contract = DatasetContract(
        name="cesnet_tls_year22",
        root=str(root),
        label_column=None,
        required_any_columns={
            "label": ["service"],
            "tls_or_flow_features": ["flow", "bytes", "packet"],
            "timestamp": ["timestamp"],
        },
        min_samples=10,
        min_classes=2,
        expected_view_support={"host": False, "ip": True, "temporal": True, "process": False, "threat_intel": False},
        source_verified=True,
        replacement_for="maltls22",
        replacement_name_must_be_reported=True,
    )

    result = audit_dataset(contract)

    assert result.ready_for_smoke is True
    assert result.class_count == 3
    assert result.actual_view_support["process"] == "not_expected"


def test_load_cesnet_tls_year22_postfilter_metadata(tmp_path: Path):
    root = tmp_path / "cesnet"
    _fixture_csv(root)
    cfg = {
        "cesnet_tls_year22": {
            "path": str(root),
            "label_col": "service",
            "class_policy": "postfilter",
            "min_class_count": 2,
            "train_test_split": 0.75,
            "reported_as": "CESNET-TLS-Year22",
            "replacement_for": "maltls22",
            "source_verified": True,
            "drop_cols": [],
        },
        "seed": 7,
    }

    dataset = load_dataset("cesnet_tls_year22", cfg)

    assert dataset.num_classes == 2
    assert dataset.meta["dataset"] == "cesnet_tls_year22"
    assert dataset.meta["reported_as"] == "CESNET-TLS-Year22"
    assert dataset.meta["replacement_for"] == "maltls22"
    assert dataset.meta["class_policy"] == "postfilter"
    assert dataset.meta["active_views"] == ["ip", "temporal"]
    assert dataset.meta["expected_view_support"]["process"] is False
