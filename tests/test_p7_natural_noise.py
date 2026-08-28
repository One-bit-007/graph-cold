"""P1-4 guard tests: Engelen corrected-label derivation + real-noise experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.paper import p7_engelen_labels as eng
from src.paper import p7_natural_noise as p7

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Attempted discipline (mandated: Attempted -> BENIGN, never an ML class)
# ---------------------------------------------------------------------------

def test_attempted_flows_become_benign():
    corrected = pd.Series(["DDoS", "FTP-Patator - Attempted", "BENIGN", "DoS Hulk - Attempted"])
    attempted = pd.Series([-1, 0, -1, 3])
    out = eng.apply_attempted_discipline(corrected, attempted)
    assert out.tolist() == ["DDoS", "BENIGN", "BENIGN", "BENIGN"]


def test_attempted_discipline_asserts_on_violation():
    # If the mapping logic were broken (attempted not mapped), the internal
    # assertions must fire - simulate by feeding a non-BENIGN attempted row
    # through a corrupted code path is impossible from outside; instead verify
    # the invariants on a mixed frame including every attempted category 0-6.
    corrected = pd.Series([f"X - Attempted"] * 7 + ["DDoS"])
    attempted = pd.Series(list(range(7)) + [-1])
    out = eng.apply_attempted_discipline(corrected, attempted)
    assert (out[:7] == "BENIGN").all()
    assert not out.str.contains("Attempted").any()


# ---------------------------------------------------------------------------
# Label namespace canonicalisation
# ---------------------------------------------------------------------------

def test_canonicalize_original_namespace():
    assert eng.canonicalize_original("PortScan") == "Portscan"
    assert eng.canonicalize_original("Bot") == "Botnet"
    assert eng.canonicalize_original("DoS slowloris") == "DoS Slowloris"
    assert eng.canonicalize_original("Web Attack – Brute Force") == "Web Attack - Brute Force"
    assert eng.canonicalize_original("Web Attack \x96 Brute Force") == "Web Attack - Brute Force"
    assert eng.canonicalize_original("BENIGN") == "BENIGN"


# ---------------------------------------------------------------------------
# label_flows semantics (tiny synthetic frame)
# ---------------------------------------------------------------------------

def _tiny_flow_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": ["a", "b", "c"],
            "Timestamp": pd.to_datetime(["2017-07-04 10:00", "2017-07-04 12:00", "2017-07-04 12:00"]),
            "Source IP": ["172.16.0.1", "9.9.9.9", "192.168.10.50"],
            "Destination IP": ["192.168.10.50", "8.8.8.8", "172.16.0.1"],
            "Source Port": [5000, 5001, 21],
            "Destination Port": [21, 80, 5000],
            "Total Length of Fwd Packets": [10, 0, 5],
            "Total Length of Bwd Packets": [3, 0, 0],
            "Total Fwd Packets": [2, 1, 1],
            "Total Backward Packets": [2, 1, 1],
            "Flow Duration": [100, 100, 100],
            "Fwd Packet Length Max": [5, 0, 5],
            "Label": ["NeedManualLabel"] * 3,
            "Attempted Category": [-1] * 3,
        }
    )


def test_label_flows_window_and_flip():
    df = _tiny_flow_df()
    # window covers 09:59-10:01 only; flip should catch row 'c' (reversed direction)
    eng.label_flows(df, "FTP-Patator", pd.Timestamp("2017-07-04 09:59").value, pd.Timestamp("2017-07-04 10:01").value,
                    ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[21], also_flip_flow_direction=True)
    assert df["Label"].tolist() == ["FTP-Patator", "NeedManualLabel", "NeedManualLabel"]
    # widen the window: now the flipped row c (src 192.168.10.50 -> dst 172.16.0.1, src port 21) matches
    eng.label_flows(df, "FTP-Patator", pd.Timestamp("2017-07-04 09:59").value, pd.Timestamp("2017-07-04 13:00").value,
                    ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[21], also_flip_flow_direction=True)
    assert df["Label"].tolist() == ["FTP-Patator", "NeedManualLabel", "FTP-Patator"]


def test_timestamp_reconstruction_twelve_hour_clock():
    df = pd.DataFrame(
        {
            "Flow ID": ["a", "b"],
            "Timestamp": ["6/7/2017 1:00", "6/7/2017 9:00"],
            "Source IP": ["x", "x"],
            "Destination IP": ["y", "y"],
            "Label": ["BENIGN", "BENIGN"],
        }
    )
    out = eng.format_csv_for_labelling(df)
    # 1:00 (hour<7) -> +12h -> 13:00 local -> +3h -> 16:00 UTC
    assert out["Timestamp"].iloc[0] == pd.Timestamp("2017-07-06 16:00")
    # 9:00 stays morning -> +3h -> 12:00 UTC
    assert out["Timestamp"].iloc[1] == pd.Timestamp("2017-07-06 12:00")


# ---------------------------------------------------------------------------
# Derivation reconciliation against the raw copy totals
# ---------------------------------------------------------------------------

def test_diff_summary_reconciles_with_raw_copy():
    summary_path = REPO_ROOT / "reports" / "p7_engelen_diff_summary.json"
    if not summary_path.exists():
        pytest.skip("derivation summary not generated on this machine")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_rows_total"] == 3_119_345  # raw copy row count (incl. artefacts)
    assert summary["n_rows_valid_label"] == 2_830_743
    assert summary["n_attempted_to_benign"] == 34_415  # attempted flows forced to BENIGN
    # the two dominant published findings must be present
    pairs = summary["by_orig_corrected_pair"]
    assert pairs.get("BENIGN -> DDoS", 0) > 10_000  # original release missed DDoS flows
    assert pairs.get("DoS Hulk -> BENIGN", 0) > 1_000  # attempted flows mislabeled as attacks


# ---------------------------------------------------------------------------
# No leakage: rule/entity columns never enter the experiment feature space
# ---------------------------------------------------------------------------

def test_p7_config_drops_rule_columns_and_tier1_port():
    cfg = p7._final_parity_cfg(REPO_ROOT / "configs")
    drop = set(cfg["cicids2017_raw"]["drop_cols"])
    for col in ("Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp", "Protocol"):
        assert col in drop
    assert "Destination Port" in drop  # FINAL tier1 mask parity
    assert cfg["cicids2017_raw"]["temporal_order"] == "source_file"  # FINAL split parity


def test_row_in_file_never_in_features(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    n = 60
    pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(n)],
            "Source IP": ["10.0.0.1"] * n,
            "Destination IP": ["10.0.0.2"] * n,
            "Timestamp": ["6/7/2017 9:00"] * n,
            "Flow Duration": rng.normal(1000, 10, size=n),
            "Label": ["BENIGN"] * 40 + ["DDoS"] * 20,
        }
    ).to_csv(data_dir / "Monday-tiny.pcap_ISCX.csv", index=False)
    from src.data import loaders

    ds = loaders.load_dataset("cicids2017_raw", {
        "cicids2017_raw": {
            "path": str(data_dir), "class_policy": "raw", "label_col": "Label",
            "train_test_split": 0.8, "min_class_count": 1,
            "drop_cols": ["Flow ID", "Source IP", "Destination IP", "Timestamp"],
            "split_strategy": "stratified",
        }
    })
    assert not any("__row_in_file" in str(c) for c in ds.meta["feature_names"])
    rif = ds.meta["row_in_file"]
    assert rif["train"] is not None and rif["test"] is not None
    assert len(rif["train"]) == len(ds.y_train) and len(rif["test"]) == len(ds.y_test)


def test_experiment_grid_complete():
    if not p7.OUT_CSV.exists():
        pytest.skip("p7_natural_noise.csv not generated on this machine")
    for path in (p7.OUT_CSV, Path("results/p7_natural_noise_stratified.csv")):
        frame = pd.read_csv(path)
        assert set(frame["method"]) == set(p7.METHODS)
        assert frame.groupby("method")["seed"].nunique().eq(10).all()
        assert len(frame) == 40
