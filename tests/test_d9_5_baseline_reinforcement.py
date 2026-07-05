from pathlib import Path

import pandas as pd

from src.experiments.d9_5_baseline_common import REINFORCED_FIELDNAMES, assert_original_rows_unchanged, original_expanded_with_extra
from src.experiments.d9_5_baseline_reinforcement import _reinforcement_complete


def _row(method="Graph-CoLD"):
    return {
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
        "method_family": "graph_cold",
        "implementation_status": "reused_verified_d5",
        "macro_f1": 0.9,
        "fpr": 0.1,
        "fnr": 0.1,
        "err": 0.8,
        "err_tail": 0.8,
        "err_final": 0.8,
        "compression_ratio": 0.7,
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


def test_original_expanded_rows_gain_extra_columns_without_changing_original(tmp_path: Path):
    path = tmp_path / "table_main_expanded.csv"
    pd.DataFrame([_row("Graph-CoLD"), _row("CoLD")]).to_csv(path, index=False)

    out = original_expanded_with_extra(path)

    assert list(out.columns) == list(REINFORCED_FIELDNAMES)
    assert assert_original_rows_unchanged(path, out) is True


def test_reinforcement_complete_requires_expected_schema_and_smoke():
    rows = []
    for dataset in ("cicids2017", "cesnet_tls_year22"):
        for seed in (0, 1, 2):
            for idx in range(17):
                row = _row("Decoupling")
                row.update({"dataset": dataset, "reported_as": "CICIDS-2017" if dataset == "cicids2017" else "CESNET-TLS-Year22", "seed": seed, "noise_rate": idx / 10, "faithfulness_level": "standard", "baseline_source": "source", "smoke_passed": True, "implementation_notes": "notes"})
                rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=REINFORCED_FIELDNAMES)

    assert _reinforcement_complete(frame, ["Decoupling"]) is True
    frame.loc[0, "smoke_passed"] = False
    assert _reinforcement_complete(frame, ["Decoupling"]) is False
