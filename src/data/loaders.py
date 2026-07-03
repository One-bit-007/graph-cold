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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    meta: dict[str, Any] = field(default_factory=dict)


def load_dataset(name: str, cfg: dict) -> Dataset:
    """Load one of {cicids2017, maltls22, cesnet_tls_year22, optc}."""
    key, ds_cfg = _resolve_dataset_cfg(name, cfg)
    if key == "optc":
        raise NotImplementedError("OpTC is reserved for the D4 enterprise case study.")

    seed = int(ds_cfg.get("seed", cfg.get("seed", 0) if isinstance(cfg, dict) else 0))
    rng = np.random.default_rng(seed)
    df = _read_dataset_frame(ds_cfg["path"])
    df.columns = [str(col).strip() for col in df.columns]
    sample_rows = ds_cfg.get("sample_rows")
    if sample_rows is not None:
        sample_rows = int(sample_rows)
        if sample_rows > 0 and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)

    label_col = ds_cfg.get("label_col")
    if not label_col or label_col not in df.columns:
        raise ValueError(f"Dataset '{name}' requires label_col '{label_col}' in the input table.")

    drop_cols = [col for col in ds_cfg.get("drop_cols", []) if col in df.columns]
    timestamp_cols = _timestamp_columns(df.columns)
    timestamps = _coalesce_timestamps(df, timestamp_cols)

    valid_label_mask = df[label_col].notna().to_numpy()
    if timestamps is not None:
        timestamps = timestamps[valid_label_mask]
    work = df.loc[valid_label_mask].drop(columns=drop_cols, errors="ignore").copy()
    work = work.drop(columns=[col for col in timestamp_cols if col in work.columns], errors="ignore")

    if key == "cicids2017":
        min_count = int(ds_cfg.get("min_class_count", 1000))
        work, timestamps = _filter_and_downsample_cicids(work, timestamps, label_col, min_count, rng)
    elif key == "cesnet_tls_year22":
        min_count = int(ds_cfg.get("min_class_count", 1000))
        if str(ds_cfg.get("class_policy", "postfilter")) == "postfilter":
            work, timestamps = _filter_and_downsample_postfilter(work, timestamps, label_col, min_count, rng)

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
    stratify = y if _can_stratify(y) else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
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

    class_counts = {class_names[idx]: int(count) for idx, count in enumerate(np.bincount(y))}
    meta = {
        "dataset": key,
        "dataset_name": key,
        "data_source": str(Path(ds_cfg["path"]).resolve()),
        "data_version": str(ds_cfg.get("version", "real-local")),
        "label_column": label_col,
        "num_classes": len(class_names),
        "class_policy": str(ds_cfg.get("class_policy", "postfilter11" if key == "cicids2017" else "raw")),
        "active_views": [view for view, enabled in _expected_view_support(key).items() if enabled],
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
        "scaler": scaler,
        "benign_class": _benign_class_index(class_names),
        "expected_view_support": _expected_view_support(key),
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
    aliases = {
        "cicids": "cicids2017",
        "cicids-2017": "cicids2017",
        "cicids2017": "cicids2017",
        "maltls": "maltls22",
        "maltls-22": "maltls22",
        "maltls22": "maltls22",
        "cesnet": "cesnet_tls_year22",
        "cesnet-tls-year22": "cesnet_tls_year22",
        "cesnet_tls_year22": "cesnet_tls_year22",
        "cesnettlsyear22": "cesnet_tls_year22",
        "optc": "optc",
    }
    key = aliases.get(name.lower())
    if key is None:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {sorted(set(aliases.values()))}.")

    if key in cfg:
        ds_cfg = dict(cfg[key])
        if "seed" in cfg and "seed" not in ds_cfg:
            ds_cfg["seed"] = cfg["seed"]
        return key, ds_cfg

    ds_cfg = dict(cfg)
    return key, ds_cfg


def _read_dataset_frame(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"Required real dataset path does not exist: {path}. "
            "Download/place the dataset under data/ as configured in configs/datasets.yaml; "
            "P0 submission mode forbids generated stand-ins."
        )

    if path.is_file():
        return _read_table(path)

    files: list[Path] = []
    for pattern in ("*.csv", "*.csv.gz", "*.parquet"):
        files.extend(sorted(path.rglob(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No CSV/Parquet files found under required real dataset path: {path}. "
            "Expected CICIDS-2017/MALTLS-22 tables in data/ according to configs/datasets.yaml."
        )

    frames = [_read_table(file) for file in files]
    return pd.concat(frames, axis=0, ignore_index=True)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported dataset file type: {path}")


def _timestamp_columns(columns) -> list[str]:
    timestamp_names = []
    for col in columns:
        normalized = str(col).strip().lower()
        if normalized in {"timestamp", "time", "ts"} or "timestamp" in normalized:
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
) -> tuple[pd.DataFrame, np.ndarray | None]:
    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= min_count].index
    keep_mask = df[label_col].isin(keep_labels).to_numpy()
    filtered = df.loc[keep_mask].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[keep_mask]

    counts = filtered[label_col].value_counts()
    if len(counts) <= 1:
        return filtered.reset_index(drop=True), _reset_timestamps(timestamps)

    dominant_label = counts.idxmax()
    cap = int(counts.drop(index=dominant_label).max())
    selected_positions: list[np.ndarray] = []
    for label, group in filtered.groupby(label_col, sort=False):
        positions = group.index.to_numpy()
        if label == dominant_label and len(positions) > cap:
            positions = np.sort(rng.choice(positions, size=cap, replace=False))
        selected_positions.append(positions)

    selected = np.sort(np.concatenate(selected_positions))
    downsampled = filtered.iloc[selected].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[selected]
    return downsampled, _reset_timestamps(timestamps)


def _filter_and_downsample_postfilter(
    df: pd.DataFrame,
    timestamps: np.ndarray | None,
    label_col: str,
    min_count: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= min_count].index
    keep_mask = df[label_col].isin(keep_labels).to_numpy()
    filtered = df.loc[keep_mask].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[keep_mask]
    if filtered.empty:
        raise ValueError(f"No classes remain after postfilter min_class_count={min_count}.")

    counts = filtered[label_col].value_counts()
    if len(counts) <= 1:
        return filtered, _reset_timestamps(timestamps)
    dominant_label = counts.idxmax()
    cap = int(counts.drop(index=dominant_label).max())
    selected_positions: list[np.ndarray] = []
    for label, group in filtered.groupby(label_col, sort=False):
        positions = group.index.to_numpy()
        if label == dominant_label and len(positions) > cap:
            positions = np.sort(rng.choice(positions, size=cap, replace=False))
        selected_positions.append(positions)
    selected = np.sort(np.concatenate(selected_positions))
    out = filtered.iloc[selected].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[selected]
    return out, _reset_timestamps(timestamps)


def _reset_timestamps(timestamps: np.ndarray | None) -> np.ndarray | None:
    if timestamps is None:
        return None
    return np.asarray(timestamps)


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


def _expected_view_support(key: str) -> dict[str, bool]:
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
