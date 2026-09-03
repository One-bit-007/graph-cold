"""Guard tests for P5/P0-3 factor ablation (WP-2).

These tests run on a small synthetic dataset namespace - they never touch the
real CICIDS/CESNET/UNSW files. They guard the frozen semantics of the ablation
grid, not dataset loading (that is covered by the p3/p4 regression tests).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.paper import p5_factor_ablation as p5


def _synthetic_ds(seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    n_train, n_test, n_features = 240, 80, 6
    # 4 classes with a clear tail: class 3 is rare.
    y_train = np.concatenate([
        np.full(140, 0), np.full(60, 1), np.full(30, 2), np.full(10, 3)
    ]).astype(int)
    y_test = np.concatenate([
        np.full(45, 0), np.full(20, 1), np.full(10, 2), np.full(5, 3)
    ]).astype(int)
    X_train = rng.normal(size=(n_train, n_features)) + y_train[:, None] * 0.5
    X_test = rng.normal(size=(n_test, n_features)) + y_test[:, None] * 0.5
    return SimpleNamespace(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        num_classes=4,
        meta={"active_views": ["ip"], "benign_class": 0, "class_names": ["a", "b", "c", "d"]},
    )


def test_family_sizes_frozen():
    assert p5.FAMILY_A_SIZE == 9
    assert p5.FAMILY_C_SIZE == 18
    assert p5.CELL_FULL == ("3_graph_cdm_hard", "4_graph_cdm_soft", "5_graph_cdm_freqfloor", "6_nograph_cbce")
    assert p5.CELL_SCREEN == ("1_nograph_ce", "2_graph_no_cdm", "7a_cdm_neigh_only", "7b_cdm_evid_only", "7c_cdm_full")


def test_prepare_scenario_bit_deterministic():
    ds = _synthetic_ds()
    a = p5._prepare_scenario(ds, "cicids2017", seed=3, rate=0.6)
    b = p5._prepare_scenario(ds, "cicids2017", seed=3, rate=0.6)
    for key in ("train_idx", "test_idx", "noisy", "flip", "evidence", "cdm"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"scenario field {key} not deterministic")
    np.testing.assert_array_equal(np.asarray(a["tail"]), np.asarray(b["tail"]))
    # Noise injection must actually have flipped some tail labels at rate 0.6.
    assert a["flip"].sum() > 0


def test_cells_1_and_2_identical_by_construction():
    """Documented pipeline degeneracy: in the frozen final protocol the graph
    enters training only via the CDM diagnostic, so #1 (no graph) and #2
    (graph without CDM) must produce bit-identical predictions."""
    ds = _synthetic_ds()
    scenario = p5._prepare_scenario(ds, "cicids2017", seed=1, rate=0.6)
    w1, l1, m1 = p5._cell_weights("1_nograph_ce", scenario)
    w2, l2, m2 = p5._cell_weights("2_graph_no_cdm", scenario)
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(l1, l2)
    assert m1 == m2 == "plain"
    p1 = p5._fit_predict_cell(scenario["X_train"], l1, scenario["X_test"], w1, m1, seed=11)
    p2 = p5._fit_predict_cell(scenario["X_train"], l2, scenario["X_test"], w2, m2, seed=11)
    np.testing.assert_array_equal(p1, p2)


def test_freq_floor_weights_properties():
    noisy = np.concatenate([np.full(900, 0), np.full(90, 1), np.full(10, 2)]).astype(int)
    w = p5._freq_floor_weights(noisy)
    assert w.shape == noisy.shape
    assert np.all(w >= p5.TAU_RET), "freq-floor weights must respect the retention floor"
    assert np.all(w > 0), "freq-floor weights must be strictly positive (no silent deletion)"
    # Majority class weight must not exceed the rarest class weight.
    assert w[noisy == 0].mean() <= w[noisy == 2].mean() + 1e-12
    # Frequent class (count above median) sits at the floor.
    assert np.isclose(w[noisy == 0].max(), p5.TAU_RET)


def test_cbce_weights_favour_rare_classes():
    y = np.concatenate([np.full(1000, 0), np.full(50, 1)]).astype(int)
    w = p5._cbce_weights(y)
    assert np.isclose(w.mean(), 1.0, atol=1e-6)
    assert w[y == 1].mean() > w[y == 0].mean()


def test_cell_row_outputs_finite_metrics():
    ds = _synthetic_ds()
    scenario = p5._prepare_scenario(ds, "cicids2017", seed=2, rate=0.6)
    for cell in ("3_graph_cdm_hard", "4_graph_cdm_soft", "5_graph_cdm_freqfloor", "6_nograph_cbce"):
        row = p5._cell_row("synthetic", scenario, cell)
        assert np.isfinite(row["macro_f1"])
        assert np.isfinite(row["tail_macro_f1"])
        assert 0.0 <= row["macro_f1"] <= 1.0
        assert row["weight_min"] >= 0.0
