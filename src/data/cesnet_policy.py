"""CESNET-TLS-Year22 class and view policy audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.audit import audit_dataset
from src.data.contracts import CESNET_TLS_YEAR22_CONTRACT


SELECTED_POLICY = "postfilter"
REPORTED_AS = "CESNET-TLS-Year22"


def audit_policies(configs: str | Path = "configs", reports: str | Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    ds_cfg = cfg.get("cesnet_tls_year22", {})
    root = Path(ds_cfg.get("path", CESNET_TLS_YEAR22_CONTRACT.root))
    label_col = ds_cfg.get("label_col")
    min_count = int(ds_cfg.get("min_class_count", 1000))
    audit = audit_dataset(CESNET_TLS_YEAR22_CONTRACT)
    raw_counts: dict[str, int] = {}
    if root.exists() and label_col:
        try:
            raw_counts = _read_label_counts(root, str(label_col))
        except Exception:
            raw_counts = {}
    post_counts, removed, downsample_rule = postfilter_counts(raw_counts, min_count=min_count)
    report = {
        "stage": "cesnet-class-policy",
        "dataset": "cesnet_tls_year22",
        "reported_as": REPORTED_AS,
        "replacement_for": "maltls22",
        "source_verified": True,
        "dataset_hash": audit.dataset_hash,
        "selected_policy": SELECTED_POLICY,
        "min_class_count": min_count,
        "policies": {
            "raw": {
                "class_names": list(raw_counts),
                "class_counts": raw_counts,
                "num_classes": len(raw_counts),
                "macro_imbalance_ratio": _imbalance_ratio(raw_counts),
                "suitable_for_experiment": bool(raw_counts) and min(raw_counts.values()) >= min_count,
            },
            "postfilter": {
                "class_names": list(post_counts),
                "class_counts": post_counts,
                "num_classes": len(post_counts),
                "removed_classes": removed,
                "min_class_count": min_count,
                "downsample_rule": downsample_rule,
                "macro_imbalance_ratio": _imbalance_ratio(post_counts),
                "suitable_for_experiment": bool(post_counts) and len(post_counts) >= 2,
            },
        },
        "ready_for_smoke": bool(audit.ready_for_smoke),
        "blocking_reasons": audit.blocking_reasons,
    }
    if reports is not None:
        write_policy_reports(report, reports)
        write_view_policy_report(audit, ds_cfg, reports)
    return report


def postfilter_counts(raw_counts: dict[str, int], min_count: int = 1000):
    kept = {label: int(count) for label, count in raw_counts.items() if int(count) >= min_count}
    removed = {label: int(count) for label, count in raw_counts.items() if int(count) < min_count}
    downsample = "No dominant-class downsampling needed."
    if len(kept) > 1:
        dominant = max(kept, key=kept.get)
        cap = max(count for label, count in kept.items() if label != dominant)
        if kept[dominant] > cap:
            kept[dominant] = int(cap)
            downsample = f"Downsample dominant class {dominant!r} to {cap} samples."
    return _ordered(kept), _ordered(removed), downsample


def write_policy_reports(report: dict[str, Any], reports: str | Path = "reports") -> None:
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cesnet_class_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CESNET Class Policy Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Reported as: {report['reported_as']}",
        f"- Selected policy: `{report['selected_policy']}`",
        f"- Ready for smoke: {report['ready_for_smoke']}",
        "",
        "## Raw",
        f"- Classes: {report['policies']['raw']['num_classes']}",
        f"- Suitable: {report['policies']['raw']['suitable_for_experiment']}",
        "",
        "## Postfilter",
        f"- Classes: {report['policies']['postfilter']['num_classes']}",
        f"- Removed classes: {json.dumps(report['policies']['postfilter']['removed_classes'], ensure_ascii=False)}",
        f"- Min class count: {report['min_class_count']}",
        "",
        "## Blocking Reasons",
    ]
    lines.extend([f"- {reason}" for reason in report.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    (out / "cesnet_class_policy_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_view_policy_report(audit, ds_cfg: dict, reports: str | Path = "reports") -> dict[str, Any]:
    columns = _schema_columns(Path(ds_cfg.get("path", CESNET_TLS_YEAR22_CONTRACT.root)))
    support = {
        "ip": _matching_columns(columns, CESNET_TLS_YEAR22_CONTRACT.required_any_columns["tls_or_flow_features"]),
        "temporal": _matching_columns(columns, CESNET_TLS_YEAR22_CONTRACT.required_any_columns["timestamp"]),
        "host": _matching_columns(columns, ["host", "Host", "src_ip", "dst_ip", "client_ip", "server_ip"]),
        "process": [],
        "threat_intel": [],
    }
    active = ["ip", "temporal"]
    if support["host"] and bool(ds_cfg.get("enable_host_view", False)):
        active.insert(0, "host")
    report = {
        "stage": "cesnet-view-policy",
        "dataset": "cesnet_tls_year22",
        "reported_as": REPORTED_AS,
        "active_views": active,
        "unsupported_views": ["process", "threat_intel"],
        "optional_views": {"host": bool(support["host"])},
        "view_columns": support,
        "must_not_claim": ["process lineage", "threat-intel view"],
        "ready_for_smoke": bool(audit.ready_for_smoke),
    }
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cesnet_view_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CESNET View Policy Report",
        "",
        f"- Active views: {' | '.join(active)}",
        "- Unsupported views: process | threat_intel",
        f"- Host optional columns present: {bool(support['host'])}",
        "",
        "## View Columns",
        *[f"- {view}: {', '.join(cols) if cols else 'none'}" for view, cols in support.items()],
        "",
    ]
    (out / "cesnet_view_policy_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _read_label_counts(root: Path, label_col: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in _table_files(root):
        frame = _read_table(path, columns=[label_col])
        frame.columns = [str(col).strip() for col in frame.columns]
        if label_col not in frame.columns:
            continue
        part = frame[label_col].dropna().astype(str).value_counts()
        for label, count in part.items():
            counts[str(label)] = counts.get(str(label), 0) + int(count)
    return _ordered(counts)


def _schema_columns(root: Path) -> list[str]:
    for path in _table_files(root):
        try:
            if path.suffix.lower() == ".parquet":
                return [str(col).strip() for col in pd.read_parquet(path).columns]
            return [str(col).strip() for col in pd.read_csv(path, nrows=1, low_memory=False).columns]
        except Exception:
            continue
    return []


def _table_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in ("*.csv", "*.csv.gz", "*.parquet"):
        files.extend(sorted(root.rglob(pattern)))
    return files


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if columns:
        wanted = {col.strip() for col in columns}
        return pd.read_csv(path, low_memory=False, usecols=lambda col: str(col).strip() in wanted)
    return pd.read_csv(path, low_memory=False)


def _matching_columns(columns: list[str], candidates: list[str]) -> list[str]:
    out = []
    lowered = [(col, col.lower()) for col in columns]
    for candidate in candidates:
        needle = candidate.lower()
        for col, lower in lowered:
            if col == candidate or lower == needle or needle in lower:
                if col not in out:
                    out.append(col)
    return out


def _ordered(counts: dict[str, int]) -> dict[str, int]:
    return {label: int(counts[label]) for label in sorted(counts)}


def _imbalance_ratio(counts: dict[str, int]) -> float:
    values = np.asarray(list(counts.values()), dtype=float)
    if values.size == 0 or values.min() <= 0:
        return 0.0
    return float(values.max() / values.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    report = audit_policies(args.configs, args.reports)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
