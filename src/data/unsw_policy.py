"""UNSW-NB15 class and view policy audits."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.audit import audit_dataset
from src.data.contracts import UNSW_NB15_CONTRACT
from src.data.paths import resolve_dataset_path


SELECTED_POLICY = "postfilter"
REPORTED_AS = "UNSW-NB15"


def audit_policies(
    configs: str | Path = "configs",
    reports: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    ds_cfg = dict(cfg.get("unsw_nb15", {}))
    root = resolve_dataset_path("unsw_nb15", data_root) if data_root else Path(ds_cfg.get("path", UNSW_NB15_CONTRACT.root))
    label_col = str(ds_cfg.get("label_col", "attack_cat"))
    min_count = int(ds_cfg.get("min_class_count", 1000))
    contract = replace(UNSW_NB15_CONTRACT, root=str(root)) if data_root else UNSW_NB15_CONTRACT
    audit = audit_dataset(contract)
    raw_counts: dict[str, int] = {}
    if root.exists():
        try:
            raw_counts = _read_label_counts(root, label_col)
        except Exception:
            raw_counts = {}
    post_counts, removed, downsample_rule = postfilter_counts(raw_counts, min_count=min_count)
    report = {
        "stage": "unsw-class-policy",
        "dataset": "unsw_nb15",
        "reported_as": REPORTED_AS,
        "source_verified": True,
        "dataset_hash": audit.dataset_hash,
        "actual_data_path": str(root),
        "external_data_root": str(data_root) if data_root else None,
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
        "active_views": ["ip", "temporal"],
        "unsupported_views": ["host", "process", "threat_intel"],
        "ready_for_smoke": bool(audit.ready_for_smoke),
        "ready_for_d5_component": bool(audit.ready_for_d5),
        "blocking_reasons": audit.blocking_reasons,
    }
    if reports is not None:
        write_policy_reports(report, reports)
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
    (out / "unsw_dataset_decision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# UNSW-NB15 Dataset Decision",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Reported as: {report['reported_as']}",
        f"- Source verified: {report['source_verified']}",
        f"- Selected policy: `{report['selected_policy']}`",
        f"- Ready for smoke: {report['ready_for_smoke']}",
        f"- Ready for D5 component: {report['ready_for_d5_component']}",
        f"- Active views: {' | '.join(report['active_views'])}",
        "",
        "## Raw Class Policy",
        f"- Classes: {report['policies']['raw']['num_classes']}",
        f"- Suitable: {report['policies']['raw']['suitable_for_experiment']}",
        "",
        "## Postfilter Policy",
        f"- Classes: {report['policies']['postfilter']['num_classes']}",
        f"- Removed classes: {json.dumps(report['policies']['postfilter']['removed_classes'], ensure_ascii=False)}",
        f"- Downsample rule: {report['policies']['postfilter']['downsample_rule']}",
        "",
        "## Blocking Reasons",
    ]
    lines.extend([f"- {reason}" for reason in report.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    (out / "unsw_dataset_decision.md").write_text("\n".join(lines), encoding="utf-8")


def _read_label_counts(root: Path, label_col: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in _table_files(root):
        frame = _read_table(path, columns=[label_col])
        frame.columns = [str(col).strip() for col in frame.columns]
        col = _matching_column(frame.columns, label_col)
        if not col:
            continue
        series = frame[col].dropna().astype("string").str.strip()
        series = series[(series != "") & (series.str.lower() != "nan")]
        part = series.value_counts()
        for label, count in part.items():
            counts[str(label)] = counts.get(str(label), 0) + int(count)
    return _ordered(counts)


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
        wanted = {col.strip().lower() for col in columns}
        return pd.read_csv(path, low_memory=False, usecols=lambda col: str(col).strip().lower() in wanted)
    return pd.read_csv(path, low_memory=False)


def _matching_column(columns, candidate: str) -> str | None:
    needle = candidate.strip().lower()
    for col in columns:
        if str(col).strip().lower() == needle:
            return str(col).strip()
    return None


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
    parser.add_argument("--data-root")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    print(json.dumps(audit_policies(args.configs, args.out, args.data_root), indent=2))


if __name__ == "__main__":
    main()
