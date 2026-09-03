"""P10 orchestration must not change frozen cells or other datasets."""
import numpy as np
import pandas as pd
import pytest

from src.paper import p10_unsw_rerun as p10


def test_cell_row_matches_frozen_implementation(monkeypatch):
    scenario = {"X_train": np.zeros((4, 2)), "X_test": np.zeros((3, 2)),
                "y_train": np.array([0, 0, 1, 2]), "y_test": np.array([0, 1, 2]),
                "seed": 0, "rate": .6, "tail": np.array([1, 2]), "noise_target": 0}
    monkeypatch.setattr(p10.p5, "_cell_weights", lambda *args: (np.array([.1, .2, .3, 1]), scenario['y_train'], 'plain'))
    monkeypatch.setattr(p10.p5, "_fit_predict_cell", lambda *args: np.array([1, 1, 2]))
    actual, guard = p10.cell_and_guard(scenario, '4_graph_cdm_soft')
    expected = p10.p5._cell_row('unsw_nb15', scenario, '4_graph_cdm_soft')
    assert actual == expected
    assert guard['noise_target'] == 0


def test_other_dataset_csv_bytes_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(p10, 'ROOT', tmp_path)
    old = tmp_path/'old.csv'
    target = tmp_path/'new.csv'
    old.write_bytes(b'dataset,seed,x\r\ncesnet_tls_year22,0,0.30000000000000004\r\nunsw_nb15,0,1\r\ncicids2017,0,0.0000001\r\n')
    replacement = pd.DataFrame([{'dataset': 'unsw_nb15', 'seed': 0, 'x': 2}])
    info = p10.merge_unsw(old, replacement, target)
    assert info['other_dataset_record_bytes_unchanged']
    assert b'cesnet_tls_year22,0,0.30000000000000004\r\n' in target.read_bytes()
    assert b'cicids2017,0,0.0000001\r\n' in target.read_bytes()


def test_gate_refuses_to_continue_after_decision_change(tmp_path):
    path = tmp_path/'gate.json'
    p10.dump(path, {'stop_required': True})
    with pytest.raises(RuntimeError, match='STOP'):
        p10.require_gate(path)


def test_grid_rejects_duplicates():
    row = {'dataset': 'unsw_nb15', 'seed': 0, 'tail_flip_rate': .4, 'cell': 'x'}
    with pytest.raises(AssertionError):
        p10.validate_grid(pd.DataFrame([row, row]), ['x'])
