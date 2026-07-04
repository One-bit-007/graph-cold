from pathlib import Path

import pandas as pd

from src.analysis.result_sanity import check_results
from src.experiments.d5_baseline_expansion import (
    EXPANDED_FIELDNAMES,
    _annotate_original_rows,
    _file_hash,
)


def _original_rows():
    rows = []
    for method, macro, err in [
        ("Graph-CoLD", 0.91, 0.82),
        ("CoLD", 0.86, 0.70),
        ("ablation_hard", 0.85, 0.69),
    ]:
        rows.append(
            {
                "dataset": "cicids2017",
                "reported_as": "CICIDS-2017",
                "dataset_hash": "hash",
                "actual_data_path": "data/cicids2017",
                "class_policy": "postfilter11",
                "num_classes": 11,
                "sample_policy": "full",
                "sample_size": 100,
                "sample_seed": 42,
                "sampling_stratified": True,
                "noise_type": "symmetric",
                "noise_rate": 0.2,
                "graph_beta": "none",
                "seed": 0,
                "split_id": "split",
                "noise_seed": 0,
                "model_seed": 0,
                "method": method,
                "macro_f1": macro,
                "fpr": 0.05,
                "fnr": 0.06,
                "err": err,
                "err_tail": err,
                "err_final": err,
                "compression_ratio": 0.5,
                "mean_weight": 0.8,
                "retained_fraction": 0.8,
                "retained_fraction_clean_informative": 0.8,
                "n_eff_ratio": 0.8,
                "runtime_sec": 0.1,
                "memory_mb": 1.0,
                "active_views": "host|ip|temporal",
                "source_verified": True,
                "replacement_for": "",
            }
        )
    return pd.DataFrame(rows)


def test_original_rows_are_annotated_without_changing_required_schema():
    annotated = _annotate_original_rows(_original_rows())

    assert list(annotated.columns) == list(EXPANDED_FIELDNAMES)
    assert set(annotated["implementation_status"]) == {"reused_verified_d5"}
    assert set(annotated["method_family"]) == {"graph_cold", "cold", "hard_ablation"}


def test_result_sanity_allows_only_smoke_passed_expanded_baselines():
    frame = _annotate_original_rows(_original_rows())
    extra = frame.iloc[[0]].copy()
    extra["method"] = "Noisy-Supervised"
    extra["method_family"] = "noisy_supervised"
    extra["implementation_status"] = "implemented_smoke_passed"
    extra["macro_f1"] = 0.84
    extra["err_final"] = 1.0
    extra["err"] = 1.0
    extra["err_tail"] = 1.0
    expanded = pd.concat([frame, extra], ignore_index=True)

    report = check_results(expanded)

    assert report["checks"]["no_fake_baseline_rows"] is True
    assert report["checks"]["implementation_status_valid"] is True


def test_result_sanity_blocks_fake_or_placeholder_methods():
    frame = _annotate_original_rows(_original_rows())
    fake = frame.iloc[[0]].copy()
    fake["method"] = "Random-Dummy"
    fake["method_family"] = "dummy"
    fake["implementation_status"] = "implemented_smoke_passed"
    expanded = pd.concat([frame, fake], ignore_index=True)

    report = check_results(expanded)

    assert report["passed"] is False
    assert "no_fake_baseline_rows" in report["blocking_reasons"]
    assert "no_forbidden_result_strings" in report["blocking_reasons"]


def test_file_hash_detects_original_table_changes(tmp_path: Path):
    path = tmp_path / "table_main.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = _file_hash(path)
    path.write_text("a,b\n1,3\n", encoding="utf-8")

    assert _file_hash(path) != first
