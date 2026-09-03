"""P10 guards for active frozen-protocol paths; legacy metrics stay explicit."""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import loaders
from src.metrics_canonical import canonical_tail_f1_from_perclass
from src.paper import p3_proxy_audit as p3, p4_cl_graph_combo as p4, p5_factor_ablation as p5
from src.paper import p6_leakage_ablation as p6, p7_natural_noise as p7
from src.paper import p9_fine_final_protocol as p9
from src.paper.leak_scan import scan_features


@pytest.mark.parametrize("partition", [True, False])
@pytest.mark.parametrize("binary", [True, False])
def test_unsw_alternate_targets_never_enter_features(tmp_path, partition, binary):
    frame = pd.DataFrame({"dur": np.arange(20)+0.1, "attack_cat": ["Normal", "Exploits"]*10,
                          "label": [0, 1]*10, "LABEL": [0, 1]*10, "Attack Cat": ["Normal", "Exploits"]*10})
    if partition:
        frame.iloc[:12].to_csv(tmp_path/"UNSW_NB15_training-set.csv", index=False)
        frame.iloc[12:].to_csv(tmp_path/"UNSW_NB15_testing-set.csv", index=False)
    else:
        frame.to_csv(tmp_path/"UNSW-NB15_1.csv", index=False)
    cfg = {"unsw_nb15": {"path": str(tmp_path), "label_col": "label" if binary else "attack_cat",
                         "label_mode": "binary" if binary else "multiclass", "drop_cols": ["label"],
                         "class_policy": "raw", "train_test_split": 0.6}}
    ds = loaders.load_dataset("unsw_nb15", cfg)
    assert ds.meta["feature_names"] == ["dur"]
    assert ds.num_classes == 2


def test_permanent_exclusion_cannot_be_disabled_by_unmasking():
    frame = pd.DataFrame({" label ": [0, 1], "ATTACK_CAT": ["Normal", "DoS"], "dur": [1, 2]})
    result = loaders._feature_inputs(frame, "ATTACK_CAT", "unsw_nb15")
    assert list(result) == ["dur"]
    for condition in p6.CONDITIONS:
        cfg = p6._condition_cfg("unsw_nb15", condition, Path("configs"))
        assert "label" in cfg["unsw_nb15"]["drop_cols"]


def test_primary_noise_target_routes_are_shared():
    for function in [p4._sample, p5._prepare_scenario, p6._scenario_rows, p9.prepare_scenario]:
        assert "p3._noise_target(" in inspect.getsource(function)
    assert p3._noise_target("cesnet_tls_year22", np.arange(25), {})[0] == 11


def test_primary_canonical_metric_routes_are_shared():
    assert "canonical_tail_f1_from_perclass(" in inspect.getsource(p3._tail_f1)
    for function in [p4._scenario_rows, p5._cell_row, p6._scenario_rows, p7._scenario_rows]:
        source = inspect.getsource(function)
        assert "p3._tail_f1(" in source
        assert "p2e._tail_macro_f1(" not in source
    assert "canonical_tail_f1_from_perclass(" in inspect.getsource(p9.evaluate)


def test_tail_definition_and_loader_routes_are_shared():
    assert "_tail_labels(" in inspect.getsource(p3._tail_noise_labels)
    assert "p3._tail_labels(" in inspect.getsource(p5._prepare_scenario)
    assert "p3._tail_noise_labels(" in inspect.getsource(p4._sample)
    assert "p4._dataset_cfg(" in inspect.getsource(p6._condition_cfg)
    assert "p4._load_primary_dataset(" in inspect.getsource(p5.run)
    assert "p4._load_primary_dataset(" in inspect.getsource(p9.run)
    assert p5.TRAIN_WINDOW == p4.TRAIN_WINDOW == 10000
    assert p5.TEST_WINDOW == p4.TEST_WINDOW == 5000
    assert p4.FULL_SEEDS == p5.FULL_SEEDS == tuple(range(10))


def test_cl_is_not_given_injected_rate():
    assert "noise_rate=" not in inspect.getsource(p4._scenario_rows)


def test_binary_projection_positive_control():
    y = np.array([0]*700+[1]*200+[2]*100)
    # Standardized values still reveal the exact binary target projection.
    x = np.where(y == 0, -1.83, 0.545)[:, None]
    result = scan_features(x, y, ["label"], benign_class=0)
    assert result.iloc[0].binary_normalized_mi == pytest.approx(1)
    assert result.iloc[0].binary_mi_nats < 1  # raw MI is NOT normalized MI
    assert result.iloc[0].flagged


def test_continuous_unique_noise_not_mistaken_for_exact_mi_one():
    rng = np.random.default_rng(0)
    result = scan_features(rng.normal(size=(2000, 1)), rng.integers(0, 3, 2000), ["random"], benign_class=0)
    assert result.iloc[0].discretization == "quantile_binned"
    assert result.iloc[0].multiclass_normalized_mi < 0.1
    assert not result.iloc[0].flagged


def test_cesnet_binary_semantics_not_invented():
    result = scan_features(np.arange(100)[:, None], np.tile([0, 1], 50), ["x"])
    assert result.iloc[0].binary_normalized_mi is None


def test_full_test_metric_counts_head_false_positives():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0, 1, 1])
    assert p3._tail_f1(y, pred, np.array([1])) == pytest.approx(2/3)
    assert canonical_tail_f1_from_perclass({"1": 2/3}, [1]) == pytest.approx(2/3)
