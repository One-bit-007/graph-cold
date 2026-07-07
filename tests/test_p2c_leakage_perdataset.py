import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.experiments import d5


REPORT = Path("reports/p2c_leakage_and_perdataset.json")
LEAKAGE = Path("reports/p2c_leakage_audit.json")
PER_DATASET = Path("tables/table_p2c_per_dataset_vs_cold.csv")
DELEAKED = Path("tables/table_p2c_cicids_deleaked_per_rate.csv")
CLAIMS = Path("tables/table_p2c_corrected_claims_input.csv")
INFORMATIVENESS = Path("tables/table_p2c_graph_informativeness.csv")


def _require_p2c_artifacts() -> None:
    missing = [path for path in [REPORT, LEAKAGE, PER_DATASET, DELEAKED, CLAIMS, INFORMATIVENESS] if not path.exists()]
    if missing:
        pytest.skip(f"P2c artifacts are absent locally: {missing}")


def test_p2c_report_records_p2b_gate_and_leakage_verdict():
    _require_p2c_artifacts()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    leakage = json.loads(LEAKAGE.read_text(encoding="utf-8"))

    assert report["p2b_gate"]["regenerated"] is True
    assert report["p2b_gate"]["outcome"] == "B_protocol_explained"
    assert report["p2b_gate"]["result_numbers_changed"] is False
    assert report["p2b_gate"]["number_consistency_green"] is True
    assert report["p2b_gate"]["frozen_hash_intact"] is True

    verdict = leakage["verdict"]
    assert verdict["leakage_found_in_frozen_cicids_results"] is True
    assert verdict["current_d5_runner_fixed"] is True
    assert verdict["split_crossing_edges_zero"] is True
    assert verdict["test_labels_seen_by_graph_cdm"] is False
    assert leakage["split_boundary"]["total_crossing_edges"] == 0
    assert report["canonical_numbers"]["old_cicids_rows_valid_for_claims"] is False
    assert report["canonical_numbers"]["corrected_formal_matrix_required"] is True


def test_p2c_tables_are_per_dataset_and_corrected_claims_are_not_pooled_only():
    _require_p2c_artifacts()
    per_dataset = pd.read_csv(PER_DATASET)
    claims = pd.read_csv(CLAIMS)
    info = pd.read_csv(INFORMATIVENESS)

    assert {"CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"}.issubset(set(per_dataset["dataset"]))
    assert {"CICIDS-2017", "CESNET-TLS-Year22", "UNSW-NB15"} == set(claims["dataset"])
    assert "p_macro_f1_graphcold_vs_cold" in per_dataset.columns
    assert per_dataset.groupby("dataset").size().min() > 1

    old_cicids_delta = per_dataset.loc[per_dataset["dataset"] == "CICIDS-2017", "delta_macro_f1_vs_cold"].mean()
    corrected_cicids_delta = claims.loc[claims["dataset"] == "CICIDS-2017", "graphcold_minus_cold_macro_f1"].iloc[0]
    unsw_delta = claims.loc[claims["dataset"] == "UNSW-NB15", "graphcold_minus_cold_macro_f1"].iloc[0]
    cesnet_delta = claims.loc[claims["dataset"] == "CESNET-TLS-Year22", "graphcold_minus_cold_macro_f1"].iloc[0]

    assert corrected_cicids_delta < old_cicids_delta
    assert corrected_cicids_delta < 0.15
    assert abs(cesnet_delta) < 0.01
    assert unsw_delta < 0.0
    assert "Do not use old +28 pp CICIDS headline" in claims.loc[
        claims["dataset"] == "CICIDS-2017", "claim_framing"
    ].iloc[0]
    assert info.loc[info["dataset"] == "UNSW-NB15", "interpretation"].iloc[0] == "boundary_case_low_view_support_no_positive_margin"


def test_p2c_deoracle_cicids_curve_removes_duplicate_edges_and_collapses_flat_curve():
    _require_p2c_artifacts()
    leakage = json.loads(LEAKAGE.read_text(encoding="utf-8"))
    deleaked = pd.read_csv(DELEAKED)

    assert set(deleaked["method"]) == {"Graph-CoLD", "CoLD", "ablation_hard"}
    assert deleaked["leakage_removed"].astype(str).str.lower().isin({"true", "1"}).all()
    assert pd.to_numeric(deleaked["exact_duplicate_train_rows_removed"], errors="raise").mean() > 0
    assert pd.to_numeric(deleaked["near_duplicate_graph_edges_removed"], errors="raise").mean() > 0
    assert leakage["deleaked_audit"]["flat_099_survives"] is False

    high = deleaked[(deleaked["noise_type"] != "clean") & (pd.to_numeric(deleaked["noise_rate"]) >= 0.4)]
    graphcold_high = high.loc[high["method"] == "Graph-CoLD", "macro_f1"].mean()
    assert graphcold_high < 0.95


def test_current_d5_context_is_independent_of_flip_mask():
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=np.int64)
    dataset = SimpleNamespace(
        X_train=X,
        X_test=X.copy(),
        y_train=y,
        y_test=y.copy(),
        num_classes=3,
        meta={"active_views": ["ip", "temporal"]},
    )
    bundle = d5.FormalBundle(
        dataset=dataset,
        dataset_key="toy",
        reported_as="Toy",
        dataset_hash="hash",
        actual_data_path="none",
        class_policy="toy",
        sample_policy="toy",
        sample_seed=42,
        sampling_stratified=True,
        active_views="ip|temporal",
        source_verified=True,
        replacement_for="",
    )
    noisy = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 2], dtype=np.int64)
    evidence = np.linspace(0.05, 1.0, y.shape[0])
    spec = {"noise_type": "symmetric", "noise_rate": 0.4, "graph_beta": "none"}

    context_a = d5._graphcold_context(bundle, spec, 42, np.zeros(y.shape[0], dtype=bool), evidence, {}, noisy=noisy)
    context_b = d5._graphcold_context(bundle, spec, 42, np.ones(y.shape[0], dtype=bool), evidence, {}, noisy=noisy)

    assert np.allclose(context_a.cdm, context_b.cdm)
    assert "flip.astype" not in inspect.getsource(d5._cdm_from_scenario)
    assert "y_train" not in inspect.getsource(d5._lightweight_graph)
