from pathlib import Path

import pandas as pd

from src.data.loaders import load_dataset, load_unsw_nb15


def _write_unsw_fixture(path: Path) -> None:
    rows = [
        {"id": 1, "srcip": "10.0.0.1", "dstip": "10.0.0.2", "sport": 1, "dsport": 80, "proto": "tcp", "service": "http", "state": "FIN", "dur": 0.1, "sbytes": 10, "attack_cat": "Normal"},
        {"id": 2, "srcip": "10.0.0.1", "dstip": "10.0.0.3", "sport": 2, "dsport": 80, "proto": "tcp", "service": "http", "state": "FIN", "dur": 0.2, "sbytes": 20, "attack_cat": "Normal"},
        {"id": 3, "srcip": "10.0.0.4", "dstip": "10.0.0.5", "sport": 3, "dsport": 443, "proto": "tcp", "service": "ssl", "state": "CON", "dur": 0.3, "sbytes": 30, "attack_cat": "Normal"},
        {"id": 4, "srcip": "10.0.0.6", "dstip": "10.0.0.7", "sport": 4, "dsport": 53, "proto": "udp", "service": "dns", "state": "CON", "dur": 0.4, "sbytes": 40, "attack_cat": "Normal"},
        {"id": 5, "srcip": "10.0.0.8", "dstip": "10.0.0.9", "sport": 5, "dsport": 21, "proto": "tcp", "service": "ftp", "state": "INT", "dur": 0.5, "sbytes": 50, "attack_cat": "Exploits"},
        {"id": 6, "srcip": "10.0.0.8", "dstip": "10.0.0.10", "sport": 6, "dsport": 22, "proto": "tcp", "service": "ssh", "state": "INT", "dur": 0.6, "sbytes": 60, "attack_cat": "Exploits"},
        {"id": 7, "srcip": "10.0.0.11", "dstip": "10.0.0.12", "sport": 7, "dsport": 25, "proto": "tcp", "service": "smtp", "state": "FIN", "dur": 0.7, "sbytes": 70, "attack_cat": ""},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_unsw_loader_multiclass_attack_cat_filters_empty_labels(tmp_path: Path):
    csv_path = tmp_path / "UNSW-NB15_1.csv"
    _write_unsw_fixture(csv_path)
    cfg = {
        "unsw_nb15": {
            "path": str(tmp_path),
            "label_col": "attack_cat",
            "class_policy": "postfilter",
            "min_class_count": 2,
            "train_test_split": 0.5,
            "drop_cols": ["id"],
            "seed": 7,
        }
    }

    dataset = load_dataset("unsw-nb15", cfg)

    assert dataset.num_classes == 2
    assert set(dataset.meta["class_names"]) == {"Normal", "Exploits"}
    assert dataset.meta["reported_as"] == "unsw_nb15"
    assert dataset.meta["active_views"] == ["host", "ip", "temporal"]
    assert dataset.X_train.shape[1] == dataset.X_test.shape[1]
    assert len(dataset.y_train) + len(dataset.y_test) == 4


def test_load_unsw_nb15_wrapper_uses_standard_loader(tmp_path: Path):
    csv_path = tmp_path / "UNSW-NB15_1.csv"
    _write_unsw_fixture(csv_path)
    cfg = {
        "path": str(tmp_path),
        "label_col": "attack_cat",
        "class_policy": "postfilter",
        "min_class_count": 2,
        "train_test_split": 0.5,
        "drop_cols": ["id"],
        "seed": 7,
    }

    dataset = load_unsw_nb15(cfg)

    assert dataset.num_classes == 2


def test_unsw_partition_layout_preserves_official_train_test_and_disables_ip_without_ip_columns(tmp_path: Path):
    train = pd.DataFrame(
        [
            {"id": 1, "dur": 0.1, "proto": "tcp", "service": "http", "state": "FIN", "sbytes": 10, "attack_cat": "Normal"},
            {"id": 2, "dur": 0.2, "proto": "udp", "service": "dns", "state": "CON", "sbytes": 20, "attack_cat": "Normal"},
            {"id": 3, "dur": 0.3, "proto": "tcp", "service": "ssh", "state": "INT", "sbytes": 30, "attack_cat": "Exploits"},
            {"id": 4, "dur": 0.4, "proto": "tcp", "service": "ftp", "state": "INT", "sbytes": 40, "attack_cat": "Exploits"},
        ]
    )
    test = pd.DataFrame(
        [
            {"id": 5, "dur": 0.5, "proto": "tcp", "service": "http", "state": "FIN", "sbytes": 50, "attack_cat": "Normal"},
            {"id": 6, "dur": 0.6, "proto": "tcp", "service": "ftp", "state": "INT", "sbytes": 60, "attack_cat": "Exploits"},
        ]
    )
    train.to_csv(tmp_path / "UNSW_NB15_training-set.csv", index=False)
    test.to_csv(tmp_path / "UNSW_NB15_testing-set.csv", index=False)

    dataset = load_dataset(
        "unsw_nb15",
        {
            "unsw_nb15": {
                "path": str(tmp_path),
                "label_col": "attack_cat",
                "class_policy": "postfilter",
                "min_class_count": 1,
                "drop_cols": ["id"],
                "seed": 11,
            }
        },
    )

    assert dataset.meta["layout"] == "partition"
    assert dataset.meta["active_views"] == ["temporal", "process"]
    assert len(dataset.y_train) == 4
    assert len(dataset.y_test) == 2
