from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_dataset


def test_cicids_loader_drops_small_classes_and_uses_clean_split(tmp_path: Path):
    labels = (
        ["BENIGN"] * 3000
        + ["DoS"] * 1500
        + ["PortScan"] * 1200
        + ["TinyAttack"] * 999
    )
    n_rows = len(labels)
    df = pd.DataFrame(
        {
            "Flow ID": [f"flow-{idx}" for idx in range(n_rows)],
            "Src IP": ["10.0.0.1"] * n_rows,
            "Dst IP": ["10.0.0.2"] * n_rows,
            "Timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="s").astype(str),
            "duration": np.linspace(0, 1, n_rows),
            "bytes": np.arange(n_rows) % 257,
            "Label": labels,
        }
    )
    data_file = tmp_path / "cicids.csv"
    df.to_csv(data_file, index=False)

    dataset = load_dataset(
        "cicids2017",
        {
            "cicids2017": {
                "path": str(data_file),
                "label_col": "Label",
                "drop_cols": ["Flow ID", "Src IP", "Dst IP", "Timestamp"],
                "min_class_count": 1000,
                "train_test_split": 0.8,
                "seed": 0,
            }
        },
    )

    assert "TinyAttack" not in dataset.meta["class_names"]
    assert min(dataset.meta["class_counts"].values()) >= 1000
    total = dataset.X_train.shape[0] + dataset.X_test.shape[0]
    assert dataset.X_train.shape[0] / total == 0.8
    assert "Timestamp" not in dataset.meta["feature_names"]
    assert dataset.meta["timestamps"]["train"] is not None


def test_maltls_loader_keeps_imbalanced_distribution(tmp_path: Path):
    labels = ["benign"] * 50 + ["attack_a"] * 10 + ["attack_b"] * 5
    df = pd.DataFrame(
        {
            "f1": np.arange(len(labels), dtype=float),
            "f2": np.arange(len(labels), dtype=float) % 3,
            "label": labels,
        }
    )
    data_file = tmp_path / "maltls.csv"
    df.to_csv(data_file, index=False)

    dataset = load_dataset(
        "maltls22",
        {
            "path": str(data_file),
            "label_col": "label",
            "drop_cols": [],
            "train_test_split": 0.8,
            "seed": 2,
        },
    )

    assert dataset.meta["class_counts"] == {"benign": 50, "attack_a": 10, "attack_b": 5}
    total = dataset.X_train.shape[0] + dataset.X_test.shape[0]
    assert dataset.X_train.shape[0] == int(0.8 * total)
