"""WP-3 guard tests: CICIDS-2017 raw entity isolation (union-find grouping).

Covers:
- union-find connected components are correct on synthetic graphs;
- entity columns (Flow ID / IPs / ports / Timestamp) never enter features;
- temporal_entity split achieves zero cross-split IP overlap on synthetic data;
- a giant-component (degenerate) topology falls back honestly with a note;
- raw feature set matches the ML copy feature set (header-level, skipped when
  the raw data directory is unavailable on the machine running the tests).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import loaders

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# union-find correctness (synthetic graphs, hand-computed expectations)
# ---------------------------------------------------------------------------

def test_union_find_components_hand_computed():
    src = np.array(["a", "b", "d", "c", "e"])
    dst = np.array(["b", "c", "e", "a", "f"])
    # edges: a-b, b-c, d-e, c-a, e-f  ->  comps {a,b,c} and {d,e,f}
    row_entities, component_ips = loaders._union_find_entity_groups(src, dst)
    comp_abc = row_entities[0]
    comp_def = row_entities[2]
    assert row_entities[1] == comp_abc  # b-c
    assert row_entities[3] == comp_abc  # c-a closes the triangle
    assert row_entities[4] == comp_def  # e-f extends second component
    assert comp_abc != comp_def
    assert sorted(component_ips[comp_abc]) == ["a", "b", "c"]
    assert sorted(component_ips[comp_def]) == ["d", "e", "f"]


def test_union_find_src_dst_share_component_and_missing_handled():
    src = np.array(["10.0.0.1", None, "10.0.0.1"], dtype=object)
    dst = np.array(["10.0.0.2", "10.0.0.3", "10.0.0.2"], dtype=object)
    row_entities, component_ips = loaders._union_find_entity_groups(src, dst)
    assert row_entities[0] == row_entities[2]  # same edge endpoints
    assert row_entities[1] != row_entities[0]  # <missing>-10.0.0.3 is separate
    assert any("<missing>" in ips for ips in component_ips.values())


# ---------------------------------------------------------------------------
# entity columns never enter features (tiny raw-like CSV)
# ---------------------------------------------------------------------------

def _write_tiny_raw_cicids(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_a, n_b = 120, 80
    frame = pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(n_a + n_b)],
            "Source IP": ["10.0.0.1", "10.0.0.2"] * (n_a // 2) + ["192.168.0.1", "192.168.0.2"] * (n_b // 2),
            "Source Port": rng.integers(1024, 65535, size=n_a + n_b),
            "Destination IP": ["10.0.0.2", "10.0.0.1"] * (n_a // 2) + ["192.168.0.2", "192.168.0.1"] * (n_b // 2),
            "Destination Port": rng.integers(1, 1024, size=n_a + n_b),
            "Protocol": [6] * (n_a + n_b),
            "Timestamp": ["6/7/2017 1:00"] * (n_a // 2) + ["6/7/2017 2:00"] * (n_a // 2)
            + ["7/7/2017 1:00"] * (n_b // 2) + ["7/7/2017 2:00"] * (n_b // 2),
            "Flow Duration": rng.normal(1000, 100, size=n_a + n_b),
            "Total Fwd Packets": rng.integers(1, 100, size=n_a + n_b),
            "Label": (["BENIGN"] * (n_a // 2) + ["DDoS"] * (n_a // 2)) + (["BENIGN"] * (n_b // 2) + ["PortScan"] * (n_b // 2)),
        }
    )
    frame.to_csv(dir_path / "tiny-WorkingHours.pcap_ISCX.csv", index=False)


def _tiny_raw_cfg(data_dir: Path, strategy: str) -> dict:
    return {
        "cicids2017_raw": {
            "path": str(data_dir),
            "class_policy": "raw",
            "label_col": "Label",
            "train_test_split": 0.8,
            "min_class_count": 1,
            "drop_cols": ["Flow ID", "Source IP", "Destination IP", "Source Port", "Protocol", "Timestamp"],
            "entity_cols": ["Source IP", "Destination IP"],
            "entity_union_find": True,
            "split_strategy": strategy,
        }
    }


def test_entity_columns_never_enter_features(tmp_path):
    data_dir = tmp_path / "cicids_raw"
    _write_tiny_raw_cicids(data_dir)
    ds = loaders.load_dataset("cicids2017_raw", _tiny_raw_cfg(data_dir, "stratified"))
    feature_names = [str(c) for c in ds.meta["feature_names"]]
    for forbidden in ("Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp", "Protocol"):
        assert not any(forbidden in name for name in feature_names), forbidden
    # Destination Port stays a feature (aligned with the ML copy).
    assert any("Destination Port" in name for name in feature_names)
    assert ds.meta["entities"]["column"] == "unionfind(src_ip,dst_ip)"


def test_temporal_entity_zero_cross_split_ip_overlap(tmp_path):
    data_dir = tmp_path / "cicids_raw"
    _write_tiny_raw_cicids(data_dir)
    ds = loaders.load_dataset("cicids2017_raw", _tiny_raw_cfg(data_dir, "temporal_entity"))
    notes = ds.meta["split_notes"]
    assert notes["applied_strategy"] == "temporal_entity"
    ent_train = set(int(e) for e in ds.meta["entities"]["train"])
    ent_test = set(int(e) for e in ds.meta["entities"]["test"])
    assert ent_train.isdisjoint(ent_test)
    comp_ips = ds.meta["entity_component_ips"]
    ips_train = set().union(*(comp_ips[c] for c in ent_train)) if ent_train else set()
    ips_test = set().union(*(comp_ips[c] for c in ent_test)) if ent_test else set()
    assert ips_train.isdisjoint(ips_test)
    assert notes["entity_overlap"] == 0


def test_giant_component_temporal_entity_falls_back_honestly(tmp_path):
    """When one component spans the whole timeline (the real CICIDS topology),
    temporal_entity cannot keep a viable training partition; the loader must
    record the degenerate fallback instead of pretending success."""
    data_dir = tmp_path / "cicids_raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(n)],
            "Source IP": ["10.0.0.1"] * n,  # single hub -> one giant component
            "Source Port": rng.integers(1024, 65535, size=n),
            "Destination IP": [f"10.0.1.{i % 250}" for i in range(n)],
            "Destination Port": rng.integers(1, 1024, size=n),
            "Protocol": [6] * n,
            "Timestamp": [f"6/7/2017 {1 + (i % 8)}:00" for i in range(n)],
            "Flow Duration": rng.normal(1000, 100, size=n),
            "Label": ["BENIGN" if i % 2 == 0 else "DDoS" for i in range(n)],
        }
    )
    frame.to_csv(data_dir / "hub-WorkingHours.pcap_ISCX.csv", index=False)
    ds = loaders.load_dataset("cicids2017_raw", _tiny_raw_cfg(data_dir, "temporal_entity"))
    notes = ds.meta["split_notes"]
    assert notes["applied_strategy"] != "temporal_entity"
    assert "fallback" in notes  # honest degradation, not a fake pass


# ---------------------------------------------------------------------------
# raw vs ML feature-set alignment (header-level; needs the real data dirs)
# ---------------------------------------------------------------------------

def _header_columns(csv_path: Path) -> set[str]:
    return {str(c).strip() for c in pd.read_csv(csv_path, nrows=0, low_memory=False).columns}


def test_raw_feature_set_matches_ml_copy():
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "datasets.yaml", encoding="utf-8"))
    raw_cfg, ml_cfg = cfg["cicids2017_raw"], cfg["cicids2017"]
    raw_path, ml_path = Path(raw_cfg["path"]), Path(ml_cfg["path"])
    if not raw_path.exists() or not ml_path.exists():
        pytest.skip("raw/ML CICIDS data directories not available on this machine")

    def feature_set(ds_cfg: dict, path: Path) -> set[str]:
        first = sorted(path.glob("*.csv"))[0] if path.is_dir() else path
        cols = _header_columns(first)
        return cols - set(ds_cfg.get("drop_cols", [])) - {ds_cfg.get("label_col", "Label")}

    assert feature_set(raw_cfg, raw_path) == feature_set(ml_cfg, ml_path)
