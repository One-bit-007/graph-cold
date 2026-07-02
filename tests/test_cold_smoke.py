from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_dataset
from src.models.cold_baseline import CoLD, feature_reordering


def test_feature_reordering_returns_a_permutation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))

    order = feature_reordering(X)

    assert sorted(order.tolist()) == list(range(6))


def test_cold_smoke_runs_end_to_end_on_small_maltls_sample(tmp_path: Path):
    rng = np.random.default_rng(5)
    n_per_class = 45
    frames = []
    for label in range(3):
        center = np.full(6, label * 4.0)
        X = rng.normal(loc=center, scale=0.4, size=(n_per_class, 6))
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        frame["label"] = f"class_{label}"
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    data_file = tmp_path / "maltls.csv"
    df.to_csv(data_file, index=False)

    dataset = load_dataset(
        "maltls22",
        {
            "path": str(data_file),
            "label_col": "label",
            "drop_cols": [],
            "train_test_split": 0.8,
            "seed": 0,
        },
    )
    model = CoLD(
        {
            "graph": {"max_subsets": 3},
            "train": {"seeds": [0]},
            "cold": {"classifier_estimators": 20, "gmm_max_iter": 50},
        }
    )

    model.fit_representation(dataset.X_train)
    keep_mask = model.purify(dataset.X_train, dataset.y_train)
    model.fit_classifier(dataset.X_train, dataset.y_train, keep_mask)
    preds = model.predict(dataset.X_test)

    assert preds.shape == dataset.y_test.shape
    assert set(np.unique(preds)).issubset(set(np.unique(dataset.y_train)))
