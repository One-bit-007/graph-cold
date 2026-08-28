"""P0-2 regression tests: label-construction proxy fields must never enter
features or graph construction, and group-disjoint splits must hold.

These tests extend the project's existing 'no clean-label path' regression
culture to the proxy-field channel identified in the Phase-1 audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import loaders


# ---------------------------------------------------------------------------
# _make_split guarantees
# ---------------------------------------------------------------------------

def _split(strategy, y, **kwargs):
    indices = np.arange(len(y))
    return loaders._make_split(indices, np.asarray(y), strategy, test_size=0.2, seed=0, **kwargs)


def test_entity_split_is_disjoint():
    rng = np.random.default_rng(0)
    entities = np.array([f"10.0.0.{i % 50}" for i in range(2000)])
    y = rng.integers(0, 3, size=2000)
    train_idx, test_idx, notes = loaders._make_split(
        np.arange(2000), y, "entity", timestamps=None, entities=entities, test_size=0.2, seed=0
    )
    assert notes["applied_strategy"] == "entity"
    assert set(entities[train_idx]).isdisjoint(set(entities[test_idx]))
    assert notes["entity_overlap"] == 0


def test_temporal_split_orders_train_before_test():
    rng = np.random.default_rng(1)
    base = pd.Timestamp("2022-01-01", tz="UTC")
    timestamps = np.array([(base + pd.Timedelta(seconds=int(s))).isoformat() for s in rng.integers(0, 100000, size=500)])
    y = rng.integers(0, 2, size=500)
    train_idx, test_idx, notes = loaders._make_split(
        np.arange(500), y, "temporal", timestamps=timestamps, entities=None, test_size=0.2, seed=0
    )
    assert notes["applied_strategy"] == "temporal"
    seconds = loaders._timestamps_to_seconds(timestamps)
    assert seconds[train_idx].max() <= seconds[test_idx].min()


def test_block_holdout_reports_unseen_classes():
    days = np.repeat(np.arange(5), 200)
    timestamps = np.array([pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=int(d)) for d in days])
    y = days.copy()  # one class per day -> held-out day classes unseen
    train_idx, test_idx, notes = loaders._make_split(
        np.arange(1000), y, "block_holdout", timestamps=timestamps, entities=None, test_size=0.2, seed=0
    )
    assert notes["applied_strategy"] == "block_holdout"
    assert notes["held_out_blocks"] == [4]
    assert set(np.unique(y[test_idx])).isdisjoint(set(np.unique(y[train_idx])))


def test_source_file_weekday_fallback():
    names = np.array(["Monday-WorkingHours.pcap_ISCX.csv"] * 400 + ["Friday-WorkingHours.pcap_ISCX.csv"] * 100)
    y = np.array([0] * 400 + [1] * 100)
    train_idx, test_idx, notes = loaders._make_split(
        np.arange(500), y, "temporal", timestamps=None, entities=None, source_files=names, test_size=0.2, seed=0
    )
    assert notes["applied_strategy"] == "temporal_by_source_file"
    assert set(names[test_idx]) == {"Friday-WorkingHours.pcap_ISCX.csv"}
    assert set(names[train_idx]) == {"Monday-WorkingHours.pcap_ISCX.csv"}


def test_coalesced_timestamps_parse_first_part():
    coalesced = np.array(["2022-01-01T05:00:00|2022-01-01T05:00:00.167979"])
    seconds = loaders._timestamps_to_seconds(coalesced)
    assert np.isfinite(seconds[0])
    assert seconds[0] == pytest.approx(pd.Timestamp("2022-01-01T05:00:00", tz="UTC").timestamp(), rel=1e-6)


# ---------------------------------------------------------------------------
# Proxy fields never reach the feature matrix (masked policy)
# ---------------------------------------------------------------------------

def _write_tiny_cicids_like(dir_path, filename="Monday-tiny.pcap_ISCX.csv"):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "Destination Port": rng.integers(1, 65535, size=n),
            "Flow Duration": rng.normal(1000, 100, size=n),
            "Total Fwd Packets": rng.integers(1, 100, size=n),
            "Label": ["BENIGN"] * 150 + ["DDoS"] * 50,
        }
    )
    frame.to_csv(dir_path / filename, index=False)


def test_masked_proxy_columns_never_enter_features(tmp_path):
    data_dir = tmp_path / "cicids"
    _write_tiny_cicids_like(data_dir)
    cfg = {
        "cicids2017": {
            "path": str(data_dir),
            "class_policy": "raw",
            "label_col": "Label",
            "train_test_split": 0.8,
            "min_class_count": 1,
            "drop_cols": ["Destination Port"],  # masked policy
            "split_strategy": "stratified",
        }
    }
    ds = loaders.load_dataset("cicids2017", cfg)
    feature_names = [str(c) for c in ds.meta["feature_names"]]
    assert not any("Destination Port" in name for name in feature_names)
    assert not any("__source_file" in name for name in feature_names)
    assert not any("Label" == name for name in feature_names)


def test_source_file_column_never_enters_features(tmp_path):
    data_dir = tmp_path / "cicids"
    _write_tiny_cicids_like(data_dir)
    cfg = {
        "cicids2017": {
            "path": str(data_dir),
            "class_policy": "raw",
            "label_col": "Label",
            "train_test_split": 0.8,
            "min_class_count": 1,
            "drop_cols": [],
            "split_strategy": "stratified",
        }
    }
    ds = loaders.load_dataset("cicids2017", cfg)
    feature_names = [str(c) for c in ds.meta["feature_names"]]
    assert not any("__source_file" in name for name in feature_names)


def test_cesnet_category_and_dst_port_blocked_under_masked_policy():
    """Regression for the Phase-1 finding: CATEGORY (label parent) and
    DST_PORT must not appear in CESNET features when the masked drop list is
    applied."""
    from src.paper import p3_proxy_audit as p3

    proxies = p3.PROXY_TIERS["cesnet_tls_year22"]["tier1"] + p3.PROXY_TIERS["cesnet_tls_year22"]["tier2"]
    assert "CATEGORY" in proxies and "DST_PORT" in proxies
    # Config-level guarantee: the masked condition extends drop_cols with proxies.
    base = {"cesnet_tls_year22": {"path": "x", "label_col": "APP", "drop_cols": ["ID"]}}
    cfg = p3._condition_cfg("cesnet_tls_year22", "C1_S0_masked_t1", base)
    drop = cfg["cesnet_tls_year22"]["drop_cols"]
    assert "CATEGORY" in drop and "DST_PORT" in drop


def test_unsw_partition_unsupported_conditions_declared():
    from src.paper import p3_proxy_audit as p3

    assert ("unsw_nb15", "C2_S1_temporal") in p3.UNSUPPORTED
    assert ("unsw_nb15", "C3_S2_entity") in p3.UNSUPPORTED
    assert ("cicids2017", "C3_S2_entity") in p3.UNSUPPORTED
    assert ("unsw_nb15", "C6_S1S2_temporal_entity") in p3.UNSUPPORTED


def test_cesnet_noise_target_is_frozen_service_not_class_zero():
    from src.paper import p3_proxy_audit as p3

    y = np.repeat(np.arange(12), 2)
    target, name = p3._noise_target(
        "cesnet_tls_year22", y, {"class_names": [f"service-{i}" for i in range(11)] + ["microsoft-diagnostic"]}
    )
    assert target == 11
    assert name == "microsoft-diagnostic"


def test_ids_tail_noise_excludes_benign_target():
    from src.paper import p3_proxy_audit as p3

    y = np.asarray([0, 0, 1, 1, 2, 2, 2, 2])
    tail = p3._tail_noise_labels("cicids2017", y, target=0)
    assert 0 not in tail
    assert 1 in tail
