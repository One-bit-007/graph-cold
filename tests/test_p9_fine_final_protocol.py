import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.baselines.base import array_hash
from src.baselines.fine_tabular import FINETabularBaseline, METHOD, FAITHFULNESS, squared_gram_alignment
from src.baselines.registry import get_baseline_entry
from src.paper import p3_proxy_audit as p3
from src.paper import p9_fine_final_protocol as p9


def toy():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(80, 6))
    return X, np.repeat([0, 1], 40)


def test_uncentered_gram_and_squared_scores():
    scores, v = squared_gram_alignment(np.array([[3., 0.], [3., 0.], [0., 1.]]))
    np.testing.assert_allclose(scores, [9, 9, 0])
    assert abs(v[0]) == pytest.approx(1)
    np.testing.assert_allclose(scores, (np.array([[3., 0.], [3., 0.], [0., 1.]]) @ -v)**2)


def test_gmm_selects_high_component_not_fixed_quota():
    Z = np.concatenate([np.tile([1., 0.], (30, 1)), np.tile([0., 1.], (10, 1))])
    keep, scores, info = FINETabularBaseline(seed=0).select(Z, np.zeros(40, dtype=int))
    assert keep[:30].all() and not keep[30:].any()
    assert info["0"]["fallback_reason"] is None
    assert info["0"]["gmm_converged"]
    np.testing.assert_array_equal(scores, np.r_[np.ones(30), np.zeros(10)])


@pytest.mark.parametrize("count,reason", [(3, "small_class_fewer_than_4"), (8, "degenerate_alignment_scores")])
def test_small_or_degenerate_classes_are_logged(count, reason):
    keep, _, info = FINETabularBaseline().select(np.ones((count, 2)), np.zeros(count))
    assert keep.all()
    assert info["0"]["fallback_reason"] == reason
    assert info["0"]["fallback_action"] == "retain_all_in_class"
    assert info["0"]["n_retained"] == count


def test_numeric_gmm_failure_is_logged(monkeypatch):
    import src.baselines.fine_tabular as fine
    def fail(*args, **kwargs):
        raise ValueError("synthetic numerical failure")
    monkeypatch.setattr(fine.GaussianMixture, "fit", fail)
    Z = np.arange(30).reshape(10, 3)
    keep, _, info = FINETabularBaseline().select(Z, np.zeros(10))
    assert keep.all() and info["0"]["fallback_reason"] == "gmm_numeric_failure"


@pytest.mark.parametrize("bad", ["noise_rate", "true_noise_rate", "estimated_noise_rate", "y_val", "clean_y_train", "flip_mask", "attempted"])
def test_no_oracle_input_paths(bad):
    X, y = toy()
    with pytest.raises(TypeError):
        FINETabularBaseline(**{bad: 0.6})
    with pytest.raises(TypeError):
        FINETabularBaseline().fit(X, y, **{bad: 0.6})
    with pytest.raises(TypeError):
        FINETabularBaseline().fit_predict(X, y, X[:5], 2, **{bad: 0.6})


def test_estimator_not_called_and_only_noisy_labels_used(monkeypatch):
    import src.baselines.noise_estimate as ne
    monkeypatch.setattr(ne, "estimate_noise_rate", lambda *a, **k: pytest.fail("noise estimator must not run"))
    X, y = toy()
    noisy = y.copy()
    noisy[::7] = 1-noisy[::7]
    a = FINETabularBaseline(seed=0).fit_predict(X, noisy, X[:5], 2)
    b = FINETabularBaseline(seed=0).fit_predict(X, noisy, X[:5]+1000, 2)
    assert a.details["training_label_hash"] == array_hash(noisy)
    assert a.details["noise_rate_source"] == "not_used"
    np.testing.assert_array_equal(a.retained_mask, b.retained_mask)


def test_training_api_does_not_receive_scenario_canaries(monkeypatch):
    X, y = toy()
    calls = []
    def fake(self, X_train, noisy_y_train, X_test, num_classes):
        calls.append((X_train, noisy_y_train, X_test, num_classes))
        return "ok"
    monkeypatch.setattr(FINETabularBaseline, "fit_predict", fake)
    scenario = {"X_train": X, "noisy": y, "X_test": X[:5], "num_classes": 2,
                "y_train": object(), "y_test": object(), "flip": object(), "rate": object(), "Attempted": object()}
    assert p9.fit_baseline(scenario, 0) == "ok"
    assert calls[0][1] is y and len(calls[0]) == 4


def test_train_scaler_uses_training_only():
    X, y = toy()
    model = FINETabularBaseline().fit(X, y)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean(axis=0))
    model.predict(X[:2]+1e6)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean(axis=0))


@pytest.mark.parametrize("field", ["Attempted Category", "clean_y", "flip_mask", "Label", "__row_in_file"])
def test_attempted_and_oracle_schema_rejected(field):
    with pytest.raises(AssertionError):
        p9.assert_non_oracle_schema(SimpleNamespace(meta={"feature_names": ["duration", field]}))


def test_attempted_ml_class_rejected():
    with pytest.raises(AssertionError):
        p9.assert_non_oracle_schema(SimpleNamespace(meta={"feature_names": ["duration"], "class_names": ["BENIGN", "DoS - Attempted"]}))


def test_frozen_protocol_and_noise_helper_used(monkeypatch):
    X, _ = toy()
    y = np.tile(np.arange(25), 3)
    X = X[:75]
    scenario = {"X_train": X, "X_test": X[:5], "y_train": y, "y_test": y[:5],
                "noisy": y.copy(), "train_idx": np.arange(75), "test_idx": np.arange(5),
                "noise_target": 11, "tail": p3._tail_labels(y)}
    window = {f"{k}_sha256": array_hash(scenario[k]) for k in ["X_train", "X_test", "y_train", "y_test", "train_idx", "test_idx"]}
    window.update(noise_target=11, tail_labels=scenario["tail"].tolist(),
                  noise_by_rate={"0.4": {"noisy_y_sha256": array_hash(y)}})
    calls = []
    def prepare(ds, dataset, seed, rate):
        calls.append(dataset)
        return scenario
    monkeypatch.setattr(p9.p5, "_prepare_scenario", prepare)
    ds = SimpleNamespace(meta={})
    assert p9.prepare_scenario(ds, "cesnet_tls_year22", 0, 0.4, window) is scenario
    assert calls == ["cesnet_tls_year22"]
    assert "p3._noise_target(" in inspect.getsource(p9.prepare_scenario)
    assert "p4._load_primary_dataset(" in inspect.getsource(p9.run)
    window["noise_target"] = 0
    with pytest.raises(AssertionError):
        p9.prepare_scenario(ds, "cesnet_tls_year22", 0, 0.4, window)


def test_registry_identity_and_f5_flags():
    entry = get_baseline_entry(METHOD)
    assert entry.faithfulness_level == FAITHFULNESS
    assert entry.uses_noisy_y_train and entry.uses_clean_y_test_for_eval_only
    assert entry.factory is FINETabularBaseline
    assert entry.verified and entry.include_in_formal_results


def test_p9_has_no_new_inferential_family():
    source = inspect.getsource(p9)
    for forbidden in ["_holm(", "_family_tests(", "_paired_stat(", "ttest"]:
        assert forbidden not in source


def test_canonical_evaluation_uses_full_test_set():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0, 1, 1])
    noisy = np.array([0, 0, 1])
    scenario = {"y_test": y, "noisy": noisy, "tail": np.array([1]), "seed": 0,
                "rate": 0.4, "noise_target": 0, "num_classes": 2}
    details = {"training_label_hash": array_hash(noisy), "fallback_class_count": 0, "fallback_sample_count": 0}
    result = SimpleNamespace(y_pred=pred, retained_mask=np.ones(3, dtype=bool), details=details)
    row = p9.evaluate("cicids2017", scenario, result)
    assert row["tail_macro_f1"] == pytest.approx(2/3)
    assert json.loads(row["per_class_f1_json"])["1"] == pytest.approx(2/3)
