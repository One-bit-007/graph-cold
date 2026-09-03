"""Read-only evidence for the UNSW binary-label feature caught by P9.

No classifier training and no historical artifact edits. Run from repo root.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.paper import p3_proxy_audit as p3, p4_cl_graph_combo as p4
from src.paper.p9_tail_metric_unification import sha256, assert_unchanged
from src.paper.p9_fine_final_protocol import assert_non_oracle_schema


def run():
    ds = p4._load_primary_dataset("unsw_nb15", Path("configs"))
    names = list(ds.meta["feature_names"])
    column = names.index("label")
    partitions = {}
    for split, X, y in [("train", ds.X_train, ds.y_train), ("test", ds.X_test, ds.y_test)]:
        feature = X[:, column]
        partitions[split] = {
            "n": len(y), "scaled_feature_values": np.unique(feature).tolist(),
            "positive_feature_equals_non_benign": bool(np.array_equal(feature > 0, y != ds.meta["benign_class"])),
            "agreement": float(np.mean((feature > 0) == (y != ds.meta["benign_class"]))),
            "class_cross_counts": [
                {"class_id": int(c), "class_name": ds.meta["class_names"][int(c)],
                 "scaled_feature_value": float(v), "n": int(np.sum((y == c) & (feature == v)))}
                for c in np.unique(y) for v in np.unique(feature) if np.any((y == c) & (feature == v))],
        }
    caught = None
    try:
        assert_non_oracle_schema(ds)
    except AssertionError as exc:
        caught = str(exc)
    if not caught:
        raise AssertionError("Expected label-column guard did not fire")
    partial = Path("results/p9_fine_final_protocol.partial.csv")
    frame = pd.read_csv(partial)
    report = {
        "status": "BLOCKED_UNSW_BINARY_LABEL_FEATURE", "completed": False, "submission_ready": False,
        "target_column": ds.meta["label_column"], "banned_feature": "label",
        "feature_index_zero_based": column, "n_features": len(names),
        "class_names": ds.meta["class_names"], "benign_class": ds.meta["benign_class"],
        "partitions": partitions, "guard_exception": caught,
        "effective_drop_cols": p4._dataset_cfg("unsw_nb15", Path("configs"))["unsw_nb15"]["drop_cols"],
        "proxy_tiers": p3.PROXY_TIERS["unsw_nb15"],
        "completed_fine_scenarios": len(frame), "expected_fine_scenarios": 90,
        "fine_rows_by_dataset": {k: int(v) for k, v in frame.groupby("dataset").size().items()},
        "source_sha256": {p: sha256(p) for p in ["src/data/loaders.py", "configs/datasets.yaml", "src/paper/p3_proxy_audit.py", "src/paper/p4_cl_graph_combo.py", str(partial)]},
        "canonical_sensitivity_qualification": "null on the existing frozen inputs, NOT evidence of label-proxy-free UNSW evaluation",
    }
    audit = json.loads(Path("reports/p9_tail_metric_unification.json").read_text(encoding="utf-8"))
    assert_unchanged(audit["protected_sha256"])
    Path("reports/p9_unsw_label_blocker.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"completed": False, "fine_rows": len(frame), "train_agreement": partitions["train"]["agreement"], "test_agreement": partitions["test"]["agreement"], "guard": caught}), flush=True)
    return report


if __name__ == "__main__":
    run()
