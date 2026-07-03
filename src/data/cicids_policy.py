"""CICIDS-2017 class-policy reconciliation.

This module audits the three policies discussed before D5:

* raw15: official labels as present in the local CICIDS CSV files.
* postfilter11: the deterministic loader policy used by current smoke tests.
* refined9: documented as a goal in early prompts, but not enabled unless a
  strict mapping is present in configs/docs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.audit import audit_dataset
from src.data.contracts import CICIDS2017_CONTRACT


SELECTED_POLICY = "postfilter11"


def audit_policies(configs: str | Path = "configs", reports: str | Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((Path(configs) / "datasets.yaml").read_text(encoding="utf-8"))
    cicids_cfg = cfg.get("cicids2017", cfg)
    root = Path(cicids_cfg.get("path", CICIDS2017_CONTRACT.root))
    label_col = str(cicids_cfg.get("label_col", CICIDS2017_CONTRACT.label_column))
    min_count = int(cicids_cfg.get("min_class_count", 1000))
    raw_counts = _read_label_counts(root, label_col)
    audit = audit_dataset(CICIDS2017_CONTRACT)
    post_counts, removed, downsample = postfilter11_counts(raw_counts, min_count=min_count)
    refined = refined9_audit(cicids_cfg)
    selected = SELECTED_POLICY if not refined["is_default_enabled"] else "refined9"
    if selected != SELECTED_POLICY:
        raise ValueError("refined9 cannot be selected without a strict deterministic mapping.")

    report = {
        "stage": "cicids-class-policy-reconciliation",
        "dataset": "cicids2017",
        "dataset_hash": audit.dataset_hash,
        "selected_policy": selected,
        "selected_num_classes": len(post_counts),
        "decision_reason": (
            "No authoritative refined9 mapping exists in docs/configs; current deterministic smoke policy "
            "is postfilter11 and every retained class has >= min_class_count samples."
        ),
        "policies": {
            "raw15": {
                "class_names": list(raw_counts),
                "class_counts": raw_counts,
                "num_classes": len(raw_counts),
                "macro_imbalance_ratio": _imbalance_ratio(raw_counts),
                "has_lt_1000_classes": any(count < 1000 for count in raw_counts.values()),
                "suitable_for_experiment": False,
                "reason": "Raw labels include classes below 1000 samples, conflicting with the current CICIDS loader contract.",
            },
            "postfilter11": {
                "class_names": list(post_counts),
                "class_counts": post_counts,
                "num_classes": len(post_counts),
                "removed_classes": removed,
                "delete_rule": f"Drop labels with count < {min_count}.",
                "downsample_rule": downsample,
                "macro_imbalance_ratio": _imbalance_ratio(post_counts),
                "reproducible": True,
                "consistent_with_current_smoke": len(post_counts) == 11,
                "suitable_for_experiment": True,
            },
            "refined9": refined,
        },
        "paper_statement": "CICIDS-2017 post-filtered 11-class setting",
        "must_not_claim": "CICIDS-2017 refined 9-class setting",
    }
    if reports is not None:
        write_policy_reports(report, reports)
    return report


def postfilter11_counts(raw_counts: dict[str, int], min_class_count: int | None = None, min_count: int | None = None):
    threshold = int(min_count if min_count is not None else min_class_count if min_class_count is not None else 1000)
    kept = {label: int(count) for label, count in raw_counts.items() if int(count) >= threshold}
    removed = {label: int(count) for label, count in raw_counts.items() if int(count) < threshold}
    if len(kept) > 1:
        dominant = max(kept, key=kept.get)
        cap = max(count for label, count in kept.items() if label != dominant)
        if kept[dominant] > cap:
            kept[dominant] = int(cap)
            downsample = f"Downsample dominant class {dominant!r} to {cap} samples."
        else:
            downsample = "No dominant-class downsampling needed."
    else:
        downsample = "No dominant-class downsampling needed."
    return _ordered_counts(kept), _ordered_counts(removed), downsample


def refined9_audit(cicids_cfg: dict) -> dict[str, Any]:
    mapping = cicids_cfg.get("refined9_mapping")
    if not isinstance(mapping, dict) or not mapping:
        return {
            "class_names": [],
            "num_classes": 0,
            "merge_or_delete_rules": {},
            "deterministic_from_cicids_labels": False,
            "would_change_smoke_results": True,
            "cold_paper_comparable": False,
            "is_default_enabled": False,
            "reason": "No strict refined9_mapping is defined in docs or configs; refined9 is therefore not used.",
        }
    names = sorted({str(target) for target in mapping.values()})
    return {
        "class_names": names,
        "num_classes": len(names),
        "merge_or_delete_rules": {str(k): str(v) for k, v in mapping.items()},
        "deterministic_from_cicids_labels": len(names) == 9,
        "would_change_smoke_results": True,
        "cold_paper_comparable": len(names) == 9,
        "is_default_enabled": len(names) == 9,
        "reason": "Explicit refined9_mapping found.",
    }


def write_policy_reports(report: dict[str, Any], reports: str | Path = "reports") -> None:
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cicids_class_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CICIDS Class Policy Report",
        "",
        f"- Selected policy: `{report['selected_policy']}`",
        f"- Selected classes: {report['selected_num_classes']}",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- Paper statement: {report['paper_statement']}",
        "",
        "## Policy A: raw15",
        f"- Classes: {report['policies']['raw15']['num_classes']}",
        f"- Has <1000 classes: {report['policies']['raw15']['has_lt_1000_classes']}",
        f"- Suitable: {report['policies']['raw15']['suitable_for_experiment']}",
        "",
        "## Policy B: postfilter11",
        f"- Classes: {report['policies']['postfilter11']['num_classes']}",
        f"- Removed classes: {json.dumps(report['policies']['postfilter11']['removed_classes'], ensure_ascii=False)}",
        f"- Consistent with smoke: {report['policies']['postfilter11']['consistent_with_current_smoke']}",
        "",
        "## Policy C: refined9",
        f"- Enabled: {report['policies']['refined9']['is_default_enabled']}",
        f"- Reason: {report['policies']['refined9']['reason']}",
        "",
        "## Decision",
        report["decision_reason"],
        "",
    ]
    (out / "cicids_class_policy_report.md").write_text("\n".join(lines), encoding="utf-8")


def _read_label_counts(root: Path, label_col: str) -> dict[str, int]:
    if not root.exists():
        raise FileNotFoundError(f"CICIDS root does not exist: {root}")
    counts: dict[str, int] = {}
    files = [root / name for name in CICIDS2017_CONTRACT.expected_files or []]
    files = [path for path in files if path.exists()]
    if not files:
        files = sorted(root.rglob("*.csv"))
    for path in files:
        series = pd.read_csv(path, low_memory=False, usecols=lambda col: str(col).strip() == label_col)
        series.columns = [str(col).strip() for col in series.columns]
        if label_col not in series.columns:
            raise ValueError(f"Label column {label_col!r} missing in {path}")
        part = series[label_col].dropna().astype(str).value_counts()
        for label, count in part.items():
            counts[str(label)] = counts.get(str(label), 0) + int(count)
    return _ordered_counts(counts)


def _ordered_counts(counts: dict[str, int]) -> dict[str, int]:
    def key(label: str) -> tuple[int, str]:
        lowered = label.lower()
        if "benign" in lowered or "normal" in lowered:
            return (0, lowered)
        return (1, lowered)

    return {label: int(counts[label]) for label in sorted(counts, key=key)}


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
