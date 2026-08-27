"""Guard tests for P6/P0-6 leakage ablation (Phase 3).

Pure-function level only; no real dataset loading (covered by p3/p4
regression tests).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.paper import p3_proxy_audit as p3
from src.paper import p4_cl_graph_combo as p4
from src.paper import p6_leakage_ablation as p6


def test_condition_grid_and_support_matrix_frozen():
    assert p6.CONDITIONS == ("FINAL", "DUP", "ENT", "PRX", "ORA", "ALL")
    assert ("cicids2017", "ENT") in p6.UNSUPPORTED
    assert p6.SEEDS == tuple(range(10))
    assert p6.RATE == 0.6


def test_oracle_cdm_is_exactly_one_minus_flip():
    flip = np.array([True, False, True, False])
    np.testing.assert_array_equal(p6._oracle_cdm(flip), np.array([0.0, 1.0, 0.0, 1.0]))


def test_apply_train_dedup_removes_exact_duplicates_keeps_first():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0]])
    y = np.array([0, 0, 1, 1, 0])
    X_d, y_d = p6._apply_train_dedup(X, y, {"source": "toy"})
    assert X_d.shape[0] == 3
    np.testing.assert_array_equal(y_d, np.array([0, 1, 0]))
    np.testing.assert_array_equal(X_d[:, 0], np.array([1.0, 3.0, 5.0]))


def test_condition_cfg_masking_toggles():
    configs = Path("configs")
    final = p6._condition_cfg("cesnet_tls_year22", "FINAL", configs)
    prx = p6._condition_cfg("cesnet_tls_year22", "PRX", configs)
    tiers = p3.PROXY_TIERS["cesnet_tls_year22"]
    final_drops = set(final["cesnet_tls_year22"]["drop_cols"])
    prx_drops = set(prx["cesnet_tls_year22"]["drop_cols"])
    for field in tiers["tier1"] + tiers["tier2"]:
        assert field in final_drops, f"{field} must stay masked in FINAL"
        assert field not in prx_drops, f"{field} must be unmasked in PRX"


def test_condition_cfg_split_toggles():
    configs = Path("configs")
    assert p6._condition_cfg("cesnet_tls_year22", "FINAL", configs)["cesnet_tls_year22"]["split_strategy"] == "temporal_entity"
    assert p6._condition_cfg("cesnet_tls_year22", "ENT", configs)["cesnet_tls_year22"]["split_strategy"] == "stratified"
    # CICIDS keeps its frozen temporal split even under ALL (ENT unsupported there).
    assert p6._condition_cfg("cicids2017", "ALL", configs)["cicids2017"]["split_strategy"] == "temporal"


def test_final_cfg_matches_frozen_primary_protocol():
    configs = Path("configs")
    for dataset in p6.DATASETS:
        final_cfg = p6._condition_cfg(dataset, "FINAL", configs)[dataset]
        primary_cfg = p4._dataset_cfg(dataset, configs)[dataset]
        assert final_cfg["split_strategy"] == primary_cfg["split_strategy"]
        assert final_cfg.get("drop_cols") == primary_cfg.get("drop_cols")


def test_tail_override_locks_metric_definition():
    """Without the FINAL-locked tail, toggling a factor shifts the F-2 tail
    boundary (window-relative median rule) and cross-condition Tail-F1 deltas
    become apples-to-oranges. The override must win over the local window."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    y_train = np.concatenate([np.full(140, 0), np.full(60, 1), np.full(30, 2), np.full(10, 3)]).astype(int)
    y_test = np.concatenate([np.full(45, 0), np.full(20, 1), np.full(10, 2), np.full(5, 3)]).astype(int)
    ds = SimpleNamespace(
        X_train=rng.normal(size=(240, 5)) + y_train[:, None] * 0.4,
        y_train=y_train,
        X_test=rng.normal(size=(80, 5)) + y_test[:, None] * 0.4,
        y_test=y_test,
        num_classes=4,
        meta={"active_views": ["ip"], "benign_class": 0, "class_names": ["a", "b", "c", "d"]},
        notes={},
    )
    rows_free = p6._scenario_rows("unsw_nb15", "FINAL", ds, 0)
    locked = np.array([2, 3])
    rows_lock = p6._scenario_rows("unsw_nb15", "PRX", ds, 0, tail_override=locked)
    assert rows_lock[0]["tail_locked"] is True or rows_lock[0]["tail_locked"] == True  # numpy.bool_ safe
    assert json_loads_tail(rows_lock[0]) == [2, 3]
    assert rows_free[0]["tail_locked"] in (False,)


def json_loads_tail(row):
    import json

    return json.loads(row["tail_classes"])
