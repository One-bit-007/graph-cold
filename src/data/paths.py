"""External data-root resolution for real dataset gates."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DATASET_RELATIVE_PATHS = {
    "cicids2017": Path("cicids2017"),
    "maltls22": Path("maltls22"),
    "cesnet_tls_year22": Path("tls_alternative") / "cesnet_tls_year22",
    "optc": Path("optc"),
}


def get_data_root(data_root: str | Path | None = None, configs: str | Path = "configs") -> Path:
    if data_root:
        return Path(data_root)
    env_root = os.environ.get("GRAPH_COLD_DATA_ROOT")
    if env_root:
        return Path(env_root)
    cfg = _paths_config(configs)
    if cfg.get("external_data_enabled") and cfg.get("data_root"):
        return Path(str(cfg["data_root"]))
    return Path("data")


def get_download_cache(
    data_root: str | Path | None = None,
    download_cache: str | Path | None = None,
    configs: str | Path = "configs",
) -> Path:
    if download_cache:
        return Path(download_cache)
    env_cache = os.environ.get("GRAPH_COLD_DOWNLOAD_CACHE")
    if env_cache:
        return Path(env_cache)
    cfg = _paths_config(configs)
    if cfg.get("download_cache"):
        return Path(str(cfg["download_cache"]))
    return get_data_root(data_root, configs) / "_downloads"


def resolve_dataset_path(dataset_name: str, data_root: str | Path | None = None, configs: str | Path = "configs") -> Path:
    key = dataset_name.lower().replace("-", "_")
    if key not in DATASET_RELATIVE_PATHS:
        raise ValueError(f"Unknown dataset for data-root resolution: {dataset_name}")
    return get_data_root(data_root, configs) / DATASET_RELATIVE_PATHS[key]


def apply_data_root_to_config(cfg: dict[str, Any], dataset_name: str, data_root: str | Path | None) -> dict[str, Any]:
    if not data_root:
        return cfg
    out = dict(cfg)
    ds_cfg = dict(out.get(dataset_name, {}))
    ds_cfg["path"] = str(resolve_dataset_path(dataset_name, data_root))
    ds_cfg["external_data_root"] = str(Path(data_root))
    ds_cfg["actual_data_path"] = ds_cfg["path"]
    out[dataset_name] = ds_cfg
    return out


def _paths_config(configs: str | Path = "configs") -> dict[str, Any]:
    path = Path(configs) / "paths.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
