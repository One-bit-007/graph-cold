"""Dataset loading.

Codex: implement loaders that return a standardized bundle so that downstream
graph construction and CoLD baseline share the same interface.

Contract
--------
load_dataset(name, cfg) -> Dataset
    Dataset fields:
        X_train: np.ndarray [N, d]   float features (scaled)
        y_train: np.ndarray [N]      int labels in [0, K)
        X_test:  np.ndarray [M, d]
        y_test:  np.ndarray [M]
        meta:    dict  (feature_names, class_names, timestamps if available)

Notes
-----
* Refined CICIDS-2017: downsample dominant classes, drop classes < 1000.
* MALTLS-22: keep original imbalanced distribution.
* Timestamps (if present) are retained in meta for the temporal view; they are
  NOT used as classification features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.paths import apply_data_root_to_config


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    meta: dict[str, Any] = field(default_factory=dict)


def load_dataset(name: str, cfg: dict) -> Dataset:
    """Load one of {cicids2017, maltls22, cesnet_tls_year22, unsw_nb15, optc}."""
    if isinstance(cfg, dict) and cfg.get("data_root"):
        cfg = apply_data_root_to_config(cfg, _normalize_dataset_key(name), cfg.get("data_root"))
    key, ds_cfg = _resolve_dataset_cfg(name, cfg)
    if key == "optc":
        raise NotImplementedError("OpTC is reserved for the D4 enterprise case study.")

    seed = int(ds_cfg.get("seed", cfg.get("seed", 0) if isinstance(cfg, dict) else 0))
    rng = np.random.default_rng(seed)
    sample_rows = ds_cfg.get("sample_rows")
    unsw_layout: dict[str, Any] | None = None
    if key == "unsw_nb15":
        unsw_layout = _detect_unsw_layout(Path(ds_cfg["path"]))
        if unsw_layout["layout"] == "partition":
            return _load_unsw_partition_dataset(ds_cfg, seed, rng, unsw_layout)
        df = _read_unsw_frame(ds_cfg["path"], sample_rows=int(sample_rows) if sample_rows is not None else None)
    else:
        df = _read_dataset_frame(ds_cfg["path"], sample_rows=int(sample_rows) if sample_rows is not None else None)
    df.columns = [str(col).strip() for col in df.columns]
    source_columns = list(df.columns)
    if sample_rows is not None:
        sample_rows = int(sample_rows)
        if sample_rows > 0 and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)

    label_col = _resolve_loader_label_column(key, df, ds_cfg)
    if not label_col or label_col not in df.columns:
        raise ValueError(f"Dataset '{name}' requires label_col '{label_col}' in the input table.")

    drop_cols = [col for col in ds_cfg.get("drop_cols", []) if col in df.columns]
    timestamp_cols = _timestamp_columns(df.columns)
    timestamps = _coalesce_timestamps(df, timestamp_cols)
    entity_col = next((col for col in ds_cfg.get("entity_cols", []) if col in df.columns), None)
    entities = df[entity_col].astype("string").to_numpy() if entity_col is not None else None
    entity_component_ips: dict[int, list[str]] | None = None
    if bool(ds_cfg.get("entity_union_find")):
        # F-4 faithful grouping: connected components of the (src, dst) flow
        # graph, so that NO IP (source or destination) can cross the split.
        src_col = next((c for c in ("Source IP", "Src IP", "SRC_IP") if c in df.columns), None)
        dst_col = next((c for c in ("Destination IP", "Dst IP", "DST_IP") if c in df.columns), None)
        if src_col is None or dst_col is None:
            raise ValueError("entity_union_find requires Source IP and Destination IP columns")
        entities, entity_component_ips = _union_find_entity_groups(
            df[src_col].astype("string").to_numpy(), df[dst_col].astype("string").to_numpy()
        )
        entity_col = "unionfind(src_ip,dst_ip)"
    source_files = df["__source_file"].astype("string").to_numpy() if "__source_file" in df.columns else None

    valid_label_mask = df[label_col].notna().to_numpy()
    if key == "unsw_nb15" and label_col.lower() == "attack_cat":
        stripped = df[label_col].astype("string").str.strip()
        valid_label_mask = (stripped.notna() & (stripped != "") & (stripped.str.lower() != "nan")).to_numpy()
    if timestamps is not None:
        timestamps = timestamps[valid_label_mask]
    if entities is not None:
        entities = entities[valid_label_mask]
    if source_files is not None:
        source_files = source_files[valid_label_mask]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    df.drop(columns=[col for col in timestamp_cols if col in df.columns], errors="ignore", inplace=True)
    df.drop(columns=["__source_file"], errors="ignore", inplace=True)
    work = df.loc[valid_label_mask] if not np.all(valid_label_mask) else df
    del df

    if key in {"cicids2017", "cicids2017_raw"}:
        min_count = int(ds_cfg.get("min_class_count", 1000))
        work, timestamps, entities, source_files = _filter_and_downsample_cicids(
            work, timestamps, label_col, min_count, rng, entities, source_files
        )
    elif key in {"cesnet_tls_year22", "unsw_nb15"}:
        min_count = int(ds_cfg.get("min_class_count", 1000))
        if str(ds_cfg.get("class_policy", "postfilter")) == "postfilter":
            work, timestamps, entities, source_files = _filter_and_downsample_postfilter(
                work, timestamps, label_col, min_count, rng, entities, source_files
            )

    row_in_file = None
    if "__row_in_file" in work.columns:
        # Row-level provenance key (source_file comes from meta["source_files"]);
        # survives filtering as a plain column, removed before feature building.
        row_in_file = work["__row_in_file"].to_numpy(dtype=np.int64)
        work = work.drop(columns=["__row_in_file"])

    labels_raw = work[label_col].astype(str).to_numpy()
    class_names = _ordered_class_names(labels_raw)
    label_mapping = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_mapping[label] for label in labels_raw], dtype=np.int64)

    features = _build_feature_frame(work.drop(columns=[label_col]))
    feature_names = list(features.columns)
    X = features.to_numpy(dtype=np.float32)

    train_ratio = float(ds_cfg.get("train_test_split", 0.8))
    test_size = 1.0 - train_ratio
    indices = np.arange(len(y))
    split_strategy = str(ds_cfg.get("split_strategy", "stratified"))
    # FINAL parity option: order the temporal split by source-file weekday even
    # when a real timestamp column exists (CICIDS ML copy semantics). Timestamps
    # stay in meta; only the split path changes.
    timestamps_for_split = timestamps
    if str(ds_cfg.get("temporal_order", "")) == "source_file":
        timestamps_for_split = None
    train_idx, test_idx, split_notes = _make_split(
        indices,
        y,
        split_strategy,
        timestamps=timestamps_for_split,
        entities=entities,
        source_files=source_files,
        test_size=test_size,
        seed=seed,
        holdout_blocks=int(ds_cfg.get("holdout_blocks", 1)),
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    y_train = y[train_idx]
    y_test = y[test_idx]

    if timestamps is not None:
        timestamp_train = timestamps[train_idx]
        timestamp_test = timestamps[test_idx]
    else:
        timestamp_train = None
        timestamp_test = None

    if entities is not None:
        entity_train = entities[train_idx]
        entity_test = entities[test_idx]
    else:
        entity_train = None
        entity_test = None

    if source_files is not None:
        source_file_train = source_files[train_idx]
        source_file_test = source_files[test_idx]
    else:
        source_file_train = None
        source_file_test = None

    if row_in_file is not None:
        row_in_file_train = row_in_file[train_idx]
        row_in_file_test = row_in_file[test_idx]
    else:
        row_in_file_train = None
        row_in_file_test = None

    class_counts = {class_names[idx]: int(count) for idx, count in enumerate(np.bincount(y))}
    view_support = _expected_view_support(key, source_columns, unsw_layout)
    meta = {
        "dataset": key,
        "dataset_name": key,
        "data_source": str(Path(ds_cfg["path"]).resolve()),
        "data_version": str(ds_cfg.get("version", "real-local")),
        "dataset_hash": ds_cfg.get("dataset_hash"),
        "label_column": label_col,
        "num_classes": len(class_names),
        "class_policy": str(ds_cfg.get("class_policy", "postfilter11" if key == "cicids2017" else "raw")),
        "active_views": [view for view, enabled in view_support.items() if enabled],
        "source_verified": bool(ds_cfg.get("source_verified", key != "maltls22")),
        "replacement_for": ds_cfg.get("replacement_for"),
        "reported_as": ds_cfg.get("reported_as", key),
        "feature_names": feature_names,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "class_counts": class_counts,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "timestamps": {"train": timestamp_train, "test": timestamp_test},
        "entities": {"train": entity_train, "test": entity_test, "column": entity_col},
        "entity_component_ips": entity_component_ips,
        "source_files": {"train": source_file_train, "test": source_file_test},
        "row_in_file": {"train": row_in_file_train, "test": row_in_file_test},
        "split_strategy": split_strategy,
        "split_notes": split_notes,
        "scaler": scaler,
        "benign_class": _benign_class_index(class_names),
        "expected_view_support": view_support,
        "layout": unsw_layout["layout"] if unsw_layout else None,
    }

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=len(class_names),
        meta=meta,
    )


def _resolve_dataset_cfg(name: str, cfg: dict) -> tuple[str, dict]:
    key = _normalize_dataset_key(name)
    if key in cfg:
        ds_cfg = dict(cfg[key])
        if "seed" in cfg and "seed" not in ds_cfg:
            ds_cfg["seed"] = cfg["seed"]
        return key, ds_cfg

    ds_cfg = dict(cfg)
    return key, ds_cfg


def _normalize_dataset_key(name: str) -> str:
    aliases = {
        "cicids": "cicids2017",
        "cicids-2017": "cicids2017",
        "cicids2017": "cicids2017",
        "cicids2017_raw": "cicids2017_raw",
        "cicids2017-raw": "cicids2017_raw",
        "maltls": "maltls22",
        "maltls-22": "maltls22",
        "maltls22": "maltls22",
        "cesnet": "cesnet_tls_year22",
        "cesnet-tls-year22": "cesnet_tls_year22",
        "cesnet_tls_year22": "cesnet_tls_year22",
        "cesnettlsyear22": "cesnet_tls_year22",
        "unsw": "unsw_nb15",
        "unsw-nb15": "unsw_nb15",
        "unsw_nb15": "unsw_nb15",
        "unswnb15": "unsw_nb15",
        "ustc": "ustc_tfc2016",
        "ustc-tfc2016": "ustc_tfc2016",
        "ustc_tfc2016": "ustc_tfc2016",
        "optc": "optc",
    }
    key = aliases.get(name.lower())
    if key is None:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {sorted(set(aliases.values()))}.")
    return key


def load_unsw_nb15(cfg: dict) -> Dataset:
    """Load real UNSW-NB15 files using the standard Dataset contract."""
    return load_dataset("unsw_nb15", cfg)


def _load_unsw_partition_dataset(
    ds_cfg: dict,
    seed: int,
    rng: np.random.Generator,
    layout: dict[str, Any],
) -> Dataset:
    del seed
    train_path = Path(layout["train_file"])
    test_path = Path(layout["test_file"])
    train_df = _read_table(train_path)
    test_df = _read_table(test_path)
    train_df.columns = [str(col).strip() for col in train_df.columns]
    test_df.columns = [str(col).strip() for col in test_df.columns]
    source_columns = sorted({*train_df.columns, *test_df.columns})
    train_df["__graphcold_split"] = "train"
    test_df["__graphcold_split"] = "test"
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)
    label_col = _resolve_loader_label_column("unsw_nb15", df, ds_cfg)
    if not label_col or label_col not in df.columns:
        raise ValueError(f"UNSW-NB15 partition layout requires label_col '{label_col}'.")

    timestamp_cols = _timestamp_columns(df.columns)
    timestamps = _coalesce_timestamps(df, timestamp_cols)
    stripped = df[label_col].astype("string").str.strip()
    valid = (stripped.notna() & (stripped != "") & (stripped.str.lower() != "nan")).to_numpy()
    split = df["__graphcold_split"].to_numpy()[valid]
    if timestamps is not None:
        timestamps = timestamps[valid]

    drop_cols = [col for col in ds_cfg.get("drop_cols", []) if col in df.columns]
    df.drop(columns=[*drop_cols, "__graphcold_split"], errors="ignore", inplace=True)
    df.drop(columns=[col for col in timestamp_cols if col in df.columns], errors="ignore", inplace=True)
    work = df.loc[valid].reset_index(drop=True)
    del df

    if str(ds_cfg.get("class_policy", "postfilter")) == "postfilter":
        min_count = int(ds_cfg.get("min_class_count", 1000))
        counts = work[label_col].value_counts()
        keep_labels = counts[counts >= min_count].index
        kept_counts = counts.loc[keep_labels]
        if kept_counts.empty:
            raise ValueError(f"No UNSW-NB15 classes remain after min_class_count={min_count}.")
        if len(kept_counts) <= 1:
            selected = np.flatnonzero(work[label_col].isin(keep_labels).to_numpy())
        else:
            dominant_label = kept_counts.idxmax()
            cap = int(kept_counts.drop(index=dominant_label).max())
            labels = work[label_col].to_numpy()
            selected_parts: list[np.ndarray] = []
            for label in keep_labels:
                positions = np.flatnonzero(labels == label)
                if label == dominant_label and len(positions) > cap:
                    positions = np.sort(rng.choice(positions, size=cap, replace=False))
                selected_parts.append(positions)
            selected = np.sort(np.concatenate(selected_parts))
        work = work.iloc[selected].reset_index(drop=True)
        split = split[selected]
        if timestamps is not None:
            timestamps = timestamps[selected]

    labels_raw = work[label_col].astype(str).to_numpy()
    class_names = _ordered_class_names(labels_raw)
    label_mapping = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_mapping[label] for label in labels_raw], dtype=np.int64)
    features = _build_feature_frame(work.drop(columns=[label_col]))
    feature_names = list(features.columns)
    X = features.to_numpy(dtype=np.float32)
    train_idx = np.flatnonzero(split == "train")
    test_idx = np.flatnonzero(split == "test")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("UNSW-NB15 partition layout must contain non-empty train and test splits.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    view_support = _expected_view_support("unsw_nb15", source_columns, layout)
    class_counts = {class_names[idx]: int(count) for idx, count in enumerate(np.bincount(y, minlength=len(class_names)))}
    meta = {
        "dataset": "unsw_nb15",
        "dataset_name": "unsw_nb15",
        "data_source": str(Path(ds_cfg["path"]).resolve()),
        "data_version": str(ds_cfg.get("version", "real-local")),
        "dataset_hash": ds_cfg.get("dataset_hash"),
        "label_column": label_col,
        "num_classes": len(class_names),
        "class_policy": str(ds_cfg.get("class_policy", "postfilter")),
        "active_views": [view for view, enabled in view_support.items() if enabled],
        "source_verified": bool(ds_cfg.get("source_verified", True)),
        "replacement_for": ds_cfg.get("replacement_for"),
        "reported_as": ds_cfg.get("reported_as", "UNSW-NB15"),
        "feature_names": feature_names,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "class_counts": class_counts,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "timestamps": {
            "train": timestamps[train_idx] if timestamps is not None else None,
            "test": timestamps[test_idx] if timestamps is not None else None,
        },
        "scaler": scaler,
        "benign_class": _benign_class_index(class_names),
        "expected_view_support": view_support,
        "layout": layout["layout"],
        "layout_files": layout["files"],
    }
    return Dataset(
        X_train=X_train,
        y_train=y[train_idx],
        X_test=X_test,
        y_test=y[test_idx],
        num_classes=len(class_names),
        meta=meta,
    )


def _resolve_loader_label_column(key: str, df: pd.DataFrame, ds_cfg: dict) -> str | None:
    requested = ds_cfg.get("label_col")
    if requested and requested in df.columns:
        return requested
    if key == "unsw_nb15":
        label_mode = str(ds_cfg.get("label_mode", "multiclass")).lower()
        candidates = ["attack_cat", "Attack_cat", "Attack Cat"] if label_mode != "binary" else ["label", "Label"]
        candidates.extend(["label", "Label", "attack_cat", "Attack_cat"])
        for candidate in candidates:
            for col in df.columns:
                if str(col).strip().lower() == candidate.lower():
                    return str(col).strip()
    return requested


def _detect_unsw_layout(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"layout": "absent", "files": [], "columns": []}
    files = [path] if path.is_file() else sorted(path.rglob("*.csv"))
    by_name = {file.name.lower(): file for file in files}
    train = by_name.get("unsw_nb15_training-set.csv")
    test = by_name.get("unsw_nb15_testing-set.csv")
    if train is not None and test is not None:
        columns = sorted({*_safe_columns(train), *_safe_columns(test)})
        return {
            "layout": "partition",
            "train_file": str(train),
            "test_file": str(test),
            "files": [str(train), str(test)],
            "columns": columns,
        }
    full = [
        file
        for file in files
        if file.name.lower().startswith("unsw-nb15_")
        and file.name.lower().endswith(".csv")
        and "features" not in file.name.lower()
    ]
    if full:
        columns = _safe_columns(full[0])
        features = [str(file) for file in files if "features" in file.name.lower()]
        return {
            "layout": "full_csv",
            "files": [str(file) for file in sorted(full)],
            "features_file": features[0] if features else None,
            "columns": columns,
        }
    return {"layout": "unknown", "files": [str(file) for file in files], "columns": _safe_columns(files[0]) if files else []}


def _read_unsw_frame(path_value: str | Path, sample_rows: int | None = None) -> pd.DataFrame:
    path = Path(path_value)
    layout = _detect_unsw_layout(path)
    if layout["layout"] == "full_csv":
        frames = []
        remaining = sample_rows if sample_rows and sample_rows > 0 else None
        for file_value in layout["files"]:
            file = Path(file_value)
            frame = _read_table(file, nrows=remaining if remaining is not None else None)
            frames.append(frame)
            if remaining is not None:
                remaining -= len(frame)
                if remaining <= 0:
                    break
        return pd.concat(frames, axis=0, ignore_index=True)
    return _read_dataset_frame(path, sample_rows=sample_rows)


def _safe_columns(path: Path) -> list[str]:
    try:
        return [str(col).strip() for col in pd.read_csv(path, nrows=0, low_memory=False).columns]
    except Exception:
        return []


def _read_dataset_frame(path_value: str | Path, sample_rows: int | None = None) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"Required real dataset path does not exist: {path}. "
            "Download/place the dataset under data/ as configured in configs/datasets.yaml; "
            "P0 submission mode forbids generated stand-ins."
        )

    if path.is_file():
        return _read_table(path, nrows=sample_rows if sample_rows and sample_rows > 0 else None)

    files: list[Path] = []
    for pattern in ("*.csv", "*.csv.gz", "*.csv.xz", "*.parquet"):
        files.extend(sorted(path.rglob(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No CSV/Parquet files found under required real dataset path: {path}. "
            "Expected CICIDS-2017/MALTLS-22 tables in data/ according to configs/datasets.yaml."
        )

    frames = []
    remaining = sample_rows if sample_rows and sample_rows > 0 else None
    for file in files:
        nrows = remaining if remaining is not None else None
        frame = _read_table(file, nrows=nrows)
        frame["__source_file"] = file.name
        frame["__row_in_file"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break
    return pd.concat(frames, axis=0, ignore_index=True)


def _read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz") or suffixes.endswith(".csv.xz"):
        try:
            return pd.read_csv(path, low_memory=False, nrows=nrows)
        except UnicodeDecodeError:
            # Raw CICIDS-2017 GeneratedLabelledFlows contain a few cp1252 bytes
            # (e.g. 0x96 en-dash); cp1252 maps 0x96 to U+2013 so labels match
            # the cleaned ML copy ("Web Attack – Brute Force").
            warnings.warn(f"UTF-8 decode failed for {path.name}; fell back to cp1252", stacklevel=2)
            return pd.read_csv(path, low_memory=False, nrows=nrows, encoding="cp1252")
    raise ValueError(f"Unsupported dataset file type: {path}")


def _timestamp_columns(columns) -> list[str]:
    timestamp_names = []
    for col in columns:
        normalized = str(col).strip().lower()
        if normalized in {"timestamp", "time", "ts"} or "timestamp" in normalized or normalized.startswith("time_"):
            timestamp_names.append(col)
    return timestamp_names


def _coalesce_timestamps(df: pd.DataFrame, timestamp_cols: list[str]) -> np.ndarray | None:
    if not timestamp_cols:
        return None
    if len(timestamp_cols) == 1:
        return df[timestamp_cols[0]].to_numpy()
    return df[timestamp_cols].astype(str).agg("|".join, axis=1).to_numpy()


def _filter_and_downsample_cicids(
    df: pd.DataFrame,
    timestamps: np.ndarray | None,
    label_col: str,
    min_count: int,
    rng: np.random.Generator,
    entities: np.ndarray | None = None,
    source_files: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= min_count].index
    kept_counts = counts.loc[keep_labels]
    if kept_counts.empty:
        raise ValueError(f"No CICIDS classes remain after min_class_count={min_count}.")
    counts = kept_counts
    if len(counts) <= 1:
        keep_mask = df[label_col].isin(keep_labels).to_numpy()
        out = df.loc[keep_mask].reset_index(drop=True)
        kept_timestamps = _reset_timestamps(timestamps[keep_mask] if timestamps is not None else None)
        kept_entities = entities[keep_mask] if entities is not None else None
        kept_sources = source_files[keep_mask] if source_files is not None else None
        return out, kept_timestamps, kept_entities, kept_sources

    dominant_label = counts.idxmax()
    cap = int(counts.drop(index=dominant_label).max())
    selected_positions: list[np.ndarray] = []
    labels = df[label_col].to_numpy()
    for label in keep_labels:
        positions = np.flatnonzero(labels == label)
        if label == dominant_label and len(positions) > cap:
            positions = np.sort(rng.choice(positions, size=cap, replace=False))
        selected_positions.append(positions)

    selected = np.sort(np.concatenate(selected_positions))
    downsampled = df.iloc[selected].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[selected]
    if entities is not None:
        entities = entities[selected]
    if source_files is not None:
        source_files = source_files[selected]
    return downsampled, _reset_timestamps(timestamps), entities, source_files


def _filter_and_downsample_postfilter(
    df: pd.DataFrame,
    timestamps: np.ndarray | None,
    label_col: str,
    min_count: int,
    rng: np.random.Generator,
    entities: np.ndarray | None = None,
    source_files: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= min_count].index
    kept_counts = counts.loc[keep_labels]
    if kept_counts.empty:
        raise ValueError(f"No classes remain after postfilter min_class_count={min_count}.")

    counts = kept_counts
    if len(counts) <= 1:
        keep_mask = df[label_col].isin(keep_labels).to_numpy()
        kept_timestamps = _reset_timestamps(timestamps[keep_mask] if timestamps is not None else None)
        kept_entities = entities[keep_mask] if entities is not None else None
        kept_sources = source_files[keep_mask] if source_files is not None else None
        return df.loc[keep_mask].reset_index(drop=True), kept_timestamps, kept_entities, kept_sources
    dominant_label = counts.idxmax()
    cap = int(counts.drop(index=dominant_label).max())
    selected_positions: list[np.ndarray] = []
    labels = df[label_col].to_numpy()
    for label in keep_labels:
        positions = np.flatnonzero(labels == label)
        if label == dominant_label and len(positions) > cap:
            positions = np.sort(rng.choice(positions, size=cap, replace=False))
        selected_positions.append(positions)
    selected = np.sort(np.concatenate(selected_positions))
    out = df.iloc[selected].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[selected]
    if entities is not None:
        entities = entities[selected]
    if source_files is not None:
        source_files = source_files[selected]
    return out, _reset_timestamps(timestamps), entities, source_files


def _reset_timestamps(timestamps: np.ndarray | None) -> np.ndarray | None:
    if timestamps is None:
        return None
    return np.asarray(timestamps)


def _union_find_entity_groups(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict[int, list[str]]]:
    """Connected components of the (Source IP, Destination IP) flow graph.

    Returns per-row component ids plus the component -> member-IP mapping for
    audit. Built with scipy sparse connected_components (undirected)."""
    from scipy import sparse
    from scipy.sparse import csgraph

    src_s = pd.Series(src).fillna("<missing>").astype(str).to_numpy()
    dst_s = pd.Series(dst).fillna("<missing>").astype(str).to_numpy()
    ips, inv = np.unique(np.concatenate([src_s, dst_s]), return_inverse=True)
    n = src_s.shape[0]
    src_inv, dst_inv = inv[:n], inv[n:]
    graph = sparse.coo_matrix(
        (np.ones(n, dtype=np.int8), (src_inv, dst_inv)), shape=(ips.shape[0], ips.shape[0])
    )
    n_comp, labels = csgraph.connected_components(graph, directed=False)
    component_ips = {
        int(comp): sorted(ips[labels == comp].tolist()) for comp in range(n_comp)
    }
    row_entities = labels[src_inv]  # per-row component id (src and dst share it)
    return row_entities.astype(np.int64), component_ips


def _make_split(
    indices: np.ndarray,
    y: np.ndarray,
    strategy: str,
    *,
    timestamps: np.ndarray | None,
    entities: np.ndarray | None,
    source_files: np.ndarray | None = None,
    test_size: float,
    seed: int,
    holdout_blocks: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """S0-S3 split dispatcher for the P0-2 leakage audit.

    Strategies:
        stratified     : S0 baseline (random stratified split)
        temporal       : S1 time-ordered split, earliest rows train / latest test
                         (falls back to weekday-ordered source files when the
                         local copy carries no timestamp column, e.g. CICIDS)
        entity         : S2 group-disjoint split by entity (e.g. source IP)
        temporal_entity: S1 and S2 jointly; latest rows are test and any test
                         entity is removed from the earlier training partition
        block_holdout  : S3 hold out the last ``holdout_blocks`` day-blocks or
                         source-file day groups only where scenario is not
                         identical to the class label (CICIDS in P3)

    Falls back to stratified with a note when required metadata is missing.
    """
    notes: dict[str, Any] = {"requested_strategy": strategy}

    if strategy == "stratified":
        stratify = y if _can_stratify(y) else None
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)
        notes["applied_strategy"] = "stratified"
        return train_idx, test_idx, notes

    if strategy == "temporal" and timestamps is not None:
        seconds = _timestamps_to_seconds(timestamps)
        order = np.argsort(seconds, kind="stable")
        cut = int(np.floor((1.0 - test_size) * len(order)))
        boundary = float(seconds[order[cut]])
        train_mask = seconds < boundary
        test_mask = seconds >= boundary
        if train_mask.any() and test_mask.any():
            notes.update({"applied_strategy": "temporal", "train_time_max": float(np.nanmax(seconds[train_mask])), "test_time_min": float(np.nanmin(seconds[test_mask])), "strict_time_boundary": True})
            return indices[train_mask], indices[test_mask], notes
        notes["fallback"] = "temporal split produced degenerate strict timestamp boundary"

    if strategy == "temporal" and source_files is not None:
        day_ids = _source_file_day_ids(source_files)
        if day_ids is not None:
            order = np.argsort(day_ids, kind="stable")
            cut = int(np.floor((1.0 - test_size) * len(order)))
            notes.update({
                "applied_strategy": "temporal_by_source_file",
                "note": "no timestamp column in local copy; ordered by source-file weekday",
                "test_days": sorted(int(d) for d in np.unique(day_ids[order[cut:]])),
            })
            return np.sort(order[:cut]), np.sort(order[cut:]), notes

    if strategy == "entity" and entities is not None:
        unique_entities, inverse = np.unique(entities, return_inverse=True)
        rng = np.random.default_rng(seed)
        group_order = rng.permutation(len(unique_entities))
        target = test_size * len(indices)
        test_groups: list[int] = []
        running = 0
        for group in group_order:
            test_groups.append(group)
            running += int(np.sum(inverse == group))
            if running >= target:
                break
        test_mask = np.isin(inverse, np.asarray(test_groups))
        notes.update({
            "applied_strategy": "entity",
            "n_entities": int(len(unique_entities)),
            "test_entities": int(len(test_groups)),
            "entity_overlap": int(len(set(entities[test_mask]) & set(entities[~test_mask]))),
        })
        return indices[~test_mask], indices[test_mask], notes

    if strategy == "temporal_entity" and timestamps is not None and entities is not None:
        seconds = _timestamps_to_seconds(timestamps)
        order = np.argsort(seconds, kind="stable")
        cut = int(np.floor((1.0 - test_size) * len(order)))
        boundary = float(seconds[order[cut]])
        temporal_test = seconds >= boundary
        temporal_train = seconds < boundary
        test_entities = set(entities[temporal_test].tolist())
        train_mask = temporal_train & (~np.isin(entities, list(test_entities)))
        test_mask = temporal_test
        if train_mask.any() and test_mask.any() and np.unique(y[train_mask]).size >= 2:
            notes.update({
                "applied_strategy": "temporal_entity",
                "train_time_max": float(np.nanmax(seconds[train_mask])),
                "test_time_min": float(np.nanmin(seconds[test_mask])),
                "test_entities": int(len(test_entities)),
                "entity_overlap": int(len(set(entities[test_mask]) & set(entities[train_mask]))),
                "train_rows_removed_for_entity_overlap": int(np.sum(temporal_train & (~train_mask))),
                "strict_time_boundary": True,
            })
            return indices[train_mask], indices[test_mask], notes
        notes["fallback"] = "temporal_entity produced degenerate training partition"

    if strategy == "block_holdout" and timestamps is not None:
        seconds = _timestamps_to_seconds(timestamps)
        day_blocks = np.floor((seconds - np.nanmin(seconds)) / 86400.0).astype(np.int64)
        blocks = np.unique(day_blocks)
        holdout = set(blocks[-max(1, holdout_blocks):].tolist())
        test_mask = np.isin(day_blocks, list(holdout))
        if test_mask.all() or not test_mask.any():
            notes["fallback"] = "block_holdout produced degenerate split; used stratified"
        else:
            notes.update({
                "applied_strategy": "block_holdout",
                "n_blocks": int(len(blocks)),
                "held_out_blocks": sorted(int(b) for b in holdout),
                "test_classes": sorted(int(c) for c in np.unique(y[test_mask])),
                "unseen_test_classes": sorted(int(c) for c in set(np.unique(y[test_mask])) - set(np.unique(y[~test_mask]))),
            })
            return indices[~test_mask], indices[test_mask], notes

    if strategy == "block_holdout" and source_files is not None:
        day_ids = _source_file_day_ids(source_files)
        if day_ids is not None:
            days = np.unique(day_ids)
            holdout = set(days[-max(1, holdout_blocks):].tolist())
            test_mask = np.isin(day_ids, list(holdout))
            if not (test_mask.all() or not test_mask.any()):
                notes.update({
                    "applied_strategy": "block_holdout_by_source_file",
                    "note": "no timestamp column in local copy; held out last weekday source-file group(s)",
                    "held_out_days": sorted(int(d) for d in holdout),
                    "test_classes": sorted(int(c) for c in np.unique(y[test_mask])),
                    "unseen_test_classes": sorted(int(c) for c in set(np.unique(y[test_mask])) - set(np.unique(y[~test_mask]))),
                })
                return indices[~test_mask], indices[test_mask], notes
        notes["fallback"] = "block_holdout produced degenerate split; used stratified"

    stratify = y if _can_stratify(y) else None
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)
    notes.update({"applied_strategy": "stratified", "fallback": f"strategy '{strategy}' unavailable (missing metadata or degenerate)"})
    return train_idx, test_idx, notes


_WEEKDAY_ORDER = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _source_file_day_ids(source_files: np.ndarray) -> np.ndarray | None:
    """Map source file names to weekday ids; None if no weekday is parseable."""
    names = np.asarray(source_files).astype(str)
    day_ids = np.full(names.shape[0], -1, dtype=np.int64)
    for idx, weekday in enumerate(_WEEKDAY_ORDER):
        mask = np.char.find(np.char.lower(names), weekday) >= 0
        day_ids[mask] = idx
    if (day_ids < 0).any():
        return None
    return day_ids


def _timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    values = pd.Series(timestamps)
    if values.astype("string").str.contains("|", regex=False).any():
        # Multi-column coalesced timestamps ("first|last"); use the first part.
        values = values.astype("string").str.split("|", n=1).str[0]
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.notna().any():
        seconds = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.float64) / 1e9
        finite = np.isfinite(seconds)
        fill = np.nanmin(seconds[finite]) if finite.any() else 0.0
        seconds[~finite] = fill
        return seconds
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(numeric).all():
        return np.arange(len(timestamps), dtype=np.float64)
    return np.nan_to_num(numeric, nan=np.nanmin(numeric))


def _ordered_class_names(labels: np.ndarray) -> list[str]:
    unique = sorted({str(label) for label in labels})

    def sort_key(label: str) -> tuple[int, str]:
        lowered = label.lower()
        if "benign" in lowered or "normal" in lowered:
            return (0, lowered)
        return (1, lowered)

    return sorted(unique, key=sort_key)


def _benign_class_index(class_names: list[str]) -> int | None:
    for idx, label in enumerate(class_names):
        lowered = label.lower()
        if "benign" in lowered or "normal" in lowered:
            return idx
    return None


def _build_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    numeric_parts: list[pd.Series] = []
    categorical_cols: list[str] = []

    for col in features.columns:
        series = features[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_parts.append(pd.to_numeric(series, errors="coerce").rename(col))
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        numeric_fraction = 0.0 if non_null == 0 else converted.notna().sum() / non_null
        if numeric_fraction >= 0.95:
            numeric_parts.append(converted.rename(col))
        else:
            categorical_cols.append(col)

    parts: list[pd.DataFrame] = []
    if numeric_parts:
        numeric = pd.concat(numeric_parts, axis=1)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.dropna(axis=1, how="all")
        if not numeric.empty:
            numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
            parts.append(numeric.astype(np.float32))

    if categorical_cols:
        categorical = pd.get_dummies(
            features[categorical_cols].astype("string").fillna("<missing>"),
            dummy_na=False,
        )
        if not categorical.empty:
            parts.append(categorical.astype(np.float32))

    if not parts:
        raise ValueError("No usable feature columns remain after dropping labels/identifiers/timestamps.")
    return pd.concat(parts, axis=1)


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and bool(np.all(counts >= 2))


def _expected_view_support(
    key: str,
    columns: list[str] | None = None,
    layout: dict[str, Any] | None = None,
) -> dict[str, bool]:
    if key == "unsw_nb15":
        lowered = {str(col).strip().lower() for col in (columns or [])}
        has_ip = bool(lowered.intersection({"srcip", "dstip", "saddr", "daddr", "source ip", "destination ip"}))
        has_time = bool(lowered.intersection({"stime", "ltime", "timestamp", "time", "ts"})) or any(
            "time" in col for col in lowered
        )
        layout_name = (layout or {}).get("layout")
        if layout_name == "partition" and not has_ip:
            return {
                "host": False,
                "ip": False,
                "temporal": True,
                "process": True,
                "threat_intel": False,
            }
        return {
            "host": has_ip,
            "ip": has_ip,
            "temporal": bool(has_time or layout_name in {"partition", "full_csv"}),
            "process": not has_ip,
            "threat_intel": False,
        }
    if key == "cesnet_tls_year22":
        return {
            "host": False,
            "ip": True,
            "temporal": True,
            "process": False,
            "threat_intel": False,
        }
    if key in {"cicids2017", "maltls22"}:
        return {
            "host": True,
            "ip": True,
            "temporal": True,
            "process": False,
            "threat_intel": False,
        }
    return {
        "host": True,
        "ip": True,
        "temporal": True,
        "process": True,
        "threat_intel": True,
    }
