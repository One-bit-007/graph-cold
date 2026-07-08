import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.experiments import d5
from src.models.evidence import compute as compute_evidence


def _toy_bundle(y_clean):
    X = np.arange(60, dtype=np.float32).reshape(15, 4)
    dataset = SimpleNamespace(
        X_train=X,
        X_test=X.copy(),
        y_train=np.asarray(y_clean, dtype=np.int64),
        y_test=np.asarray(y_clean, dtype=np.int64),
        num_classes=3,
        meta={"active_views": ["host", "ip", "temporal"]},
    )
    return d5.FormalBundle(
        dataset=dataset,
        dataset_key="toy",
        reported_as="Toy",
        dataset_hash="hash",
        actual_data_path="none",
        class_policy="toy",
        sample_policy="toy",
        sample_seed=42,
        sampling_stratified=True,
        active_views="host|ip|temporal",
        source_verified=True,
        replacement_for="",
    )


def test_graph_evidence_and_cdm_do_not_change_when_clean_labels_are_corrupted():
    y_clean = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_corrupt_clean = (y_clean + 1) % 3
    y_observed_noisy = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2], dtype=np.int64)
    evidence_cfg = {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}

    bundle_a = _toy_bundle(y_clean)
    bundle_b = _toy_bundle(y_corrupt_clean)
    anomaly_a = d5._unsupervised_feature_anomaly(bundle_a.dataset.X_train)
    anomaly_b = d5._unsupervised_feature_anomaly(bundle_b.dataset.X_train)
    graph_a = d5._lightweight_graph(bundle_a.dataset)
    graph_b = d5._lightweight_graph(bundle_b.dataset)
    evidence_a = compute_evidence(y_observed_noisy, evidence_cfg, anomaly=anomaly_a)
    evidence_b = compute_evidence(y_observed_noisy, evidence_cfg, anomaly=anomaly_b)
    cdm_a = d5._cdm_from_observed_labels(y_observed_noisy, evidence_a, graph_a, 3)
    cdm_b = d5._cdm_from_observed_labels(y_observed_noisy, evidence_b, graph_b, 3)

    assert np.allclose(anomaly_a, anomaly_b)
    assert np.array_equal(next(iter(graph_a.views.values())).edge_index, next(iter(graph_b.views.values())).edge_index)
    assert np.allclose(evidence_a, evidence_b)
    assert np.allclose(cdm_a, cdm_b)


def test_graphcold_context_ignores_flip_mask_canary():
    y_clean = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_observed_noisy = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2], dtype=np.int64)
    bundle = _toy_bundle(y_clean)
    evidence = compute_evidence(
        y_observed_noisy,
        {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
        anomaly=d5._unsupervised_feature_anomaly(bundle.dataset.X_train),
    )
    spec = {"noise_type": "symmetric", "noise_rate": 0.4, "graph_beta": "none"}
    ctx_a = d5._graphcold_context(bundle, spec, 42, np.zeros(y_clean.shape[0], dtype=bool), evidence, {}, noisy=y_observed_noisy)
    ctx_b = d5._graphcold_context(bundle, spec, 42, np.ones(y_clean.shape[0], dtype=bool), evidence, {}, noisy=y_observed_noisy)

    assert np.allclose(ctx_a.cdm, ctx_b.cdm)
    assert np.array_equal(next(iter(ctx_a.graph.views.values())).edge_index, next(iter(ctx_b.graph.views.values())).edge_index)


def test_live_pipeline_source_has_no_legacy_oracle_hooks():
    assert "flip.astype" not in inspect.getsource(d5._cdm_from_scenario)
    assert "smoke_realdata._smoke_cdm" not in inspect.getsource(d5._graphcold_context)
    assert "y_train" not in inspect.getsource(d5._lightweight_graph)
    assert "train_exact_dedup" in Path("reports/p2d_clean_rerun.md").read_text(encoding="utf-8") or Path(
        "reports/p2d_clean_rerun.json"
    ).exists()


def test_p2d_outputs_supersede_oracle_era_results():
    report_path = Path("reports/p2d_clean_rerun.json")
    assert report_path.exists()
    main = pd.read_csv("results/table_main_expanded.csv")
    report_text = report_path.read_text(encoding="utf-8")

    assert set(main["reported_as"]) == {"CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"}
    assert main["sample_policy"].astype(str).str.contains("p2d_real_audit_window").all()
    assert "benefit_vanishes" in report_text
    assert pd.to_numeric(main["cicids_train_dedup_removed"], errors="coerce").max() > 0
