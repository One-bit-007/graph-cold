"""Real dataset audit and readiness reporting.

The audit layer verifies that local files satisfy the declared contracts before
smoke tests or D5 can run. It never repairs labels, deletes classes, trains a
model, or writes formal experiment tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import DATASET_CONTRACTS, DatasetContract


@dataclass
class DatasetAuditResult:
    name: str
    root: str
    exists: bool
    expected_files_present: bool
    missing_files: list[str]
    files_used: list[str]
    file_hashes: dict[str, str]
    dataset_hash: str | None
    num_rows: int
    num_columns: int
    label_column: str | None
    label_column_present: bool
    class_count: int
    label_distribution: dict[str, int]
    missing_values: int
    infinite_values: int
    duplicate_rows: int
    numeric_feature_count: int
    required_columns_ok: bool
    required_any_columns_status: dict[str, bool]
    expected_view_support: dict[str, bool]
    actual_view_support: dict[str, str]
    ready_for_smoke: bool
    ready_for_d5: bool
    blocking_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_dataset(contract: DatasetContract) -> DatasetAuditResult:
    root = Path(contract.root)
    blocking: list[str] = []
    exists = root.exists()
    if not exists:
        blocking.append(f"dataset root does not exist: {contract.root}")

    files, missing_files = _files_for_contract(root, contract)
    expected_files_present = not missing_files
    if missing_files:
        blocking.append(f"missing expected files: {', '.join(missing_files)}")
    if exists and not files:
        blocking.append("no readable CSV files found")

    file_hashes = {str(path): _sha256(path) for path in files if path.exists()}
    dataset_hash = _dataset_hash(file_hashes) if file_hashes else None

    frame = _read_frames(files) if files else pd.DataFrame()
    if not frame.empty:
        frame.columns = [str(col).strip() for col in frame.columns]

    label_col = _resolve_label_column(frame, contract)
    label_present = bool(label_col and label_col in frame.columns)
    if not label_present:
        blocking.append(f"label column missing: {contract.label_column or 'contract label candidate'}")

    label_distribution: dict[str, int] = {}
    class_count = 0
    if label_present:
        counts = frame[label_col].astype(str).value_counts(dropna=False)
        label_distribution = {str(label): int(count) for label, count in counts.items()}
        class_count = int(counts.shape[0])

    required_columns_ok = all(col in frame.columns for col in contract.required_columns)
    missing_required = [col for col in contract.required_columns if col not in frame.columns]
    if missing_required:
        blocking.append(f"missing required columns: {', '.join(missing_required)}")

    any_status = {key: _has_any_column(frame.columns, candidates) for key, candidates in contract.required_any_columns.items()}
    _append_required_any_blockers(contract, any_status, blocking)

    if frame.shape[0] < contract.min_samples:
        blocking.append(f"row count {frame.shape[0]} below min_samples {contract.min_samples}")
    if label_present and class_count < contract.min_classes:
        blocking.append(f"class count {class_count} below min_classes {contract.min_classes}")
    if not contract.source_verified:
        blocking.append("dataset source is not verified; do not report this dataset")

    numeric = frame.select_dtypes(include=[np.number]) if not frame.empty else pd.DataFrame()
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if not numeric.empty else 0
    actual_view = _actual_view_support(contract, any_status, set(frame.columns))

    ready = exists and expected_files_present and bool(files) and label_present and required_columns_ok
    ready = ready and frame.shape[0] >= contract.min_samples and class_count >= contract.min_classes
    ready = ready and _required_any_ready(contract, any_status) and contract.source_verified
    ready = ready and not any(reason.startswith("no readable") for reason in blocking)

    return DatasetAuditResult(
        name=contract.name,
        root=str(root),
        exists=exists,
        expected_files_present=expected_files_present,
        missing_files=missing_files,
        files_used=[str(path) for path in files],
        file_hashes=file_hashes,
        dataset_hash=dataset_hash,
        num_rows=int(frame.shape[0]),
        num_columns=int(frame.shape[1]),
        label_column=label_col,
        label_column_present=label_present,
        class_count=class_count,
        label_distribution=label_distribution,
        missing_values=int(frame.isna().sum().sum()) if not frame.empty else 0,
        infinite_values=inf_count,
        duplicate_rows=int(frame.duplicated().sum()) if not frame.empty else 0,
        numeric_feature_count=int(_numeric_feature_count(frame, label_col)),
        required_columns_ok=required_columns_ok,
        required_any_columns_status=any_status,
        expected_view_support=dict(contract.expected_view_support),
        actual_view_support=actual_view,
        ready_for_smoke=bool(ready),
        ready_for_d5=bool(ready),
        blocking_reasons=blocking,
    )


def audit_all_datasets() -> dict[str, DatasetAuditResult]:
    return {name: audit_dataset(contract) for name, contract in DATASET_CONTRACTS.items()}


def write_audit_reports(results: dict[str, DatasetAuditResult] | None = None, out_dir: str | Path = "reports") -> dict[str, str]:
    results = audit_all_datasets() if results is None else results
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {name: result.to_dict() for name, result in results.items()}
    json_path = out / "dataset_audit_report.json"
    md_path = out / "dataset_audit_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_audit_markdown(results), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def write_dataset_specific_audit_report(result: DatasetAuditResult, out_dir: str | Path = "reports") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "cesnet_audit_report" if result.name == "cesnet_tls_year22" else f"{result.name}_audit_report"
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    payload = result.to_dict()
    payload.update(
        {
            "class_imbalance_ratio": _imbalance_ratio(result.label_distribution),
            "selected_class_policy": "postfilter" if result.name == "cesnet_tls_year22" else None,
            "active_views": [view for view, active in result.expected_view_support.items() if active],
            "source_verified": DATASET_CONTRACTS[result.name].source_verified if result.name in DATASET_CONTRACTS else None,
            "replacement_for": DATASET_CONTRACTS[result.name].replacement_for if result.name in DATASET_CONTRACTS else None,
            "ready_for_mini_matrix": bool(result.ready_for_smoke),
            "ready_for_d5_component": bool(result.ready_for_d5),
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_specific_audit_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def write_readiness_reports(
    audits: dict[str, DatasetAuditResult] | None = None,
    out_dir: str | Path = "reports",
) -> dict[str, str]:
    audits = audit_all_datasets() if audits is None else audits
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    readiness = build_readiness(audits)
    json_path = out / "realdata_readiness_report.json"
    md_path = out / "realdata_readiness_report.md"
    json_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    md_path.write_text(_readiness_markdown(readiness), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_readiness(audits: dict[str, DatasetAuditResult]) -> dict[str, Any]:
    cicids = audits["cicids2017"]
    maltls = audits["maltls22"]
    cesnet = audits.get("cesnet_tls_year22")
    optc = audits["optc"]
    d5_allowed = bool(cicids.ready_for_d5 and cesnet and cesnet.ready_for_d5)
    next_actions = []
    for name, audit in audits.items():
        if name == "optc":
            if not audit.ready_for_d5:
                next_actions.append("OpTC unavailable; keep it out of formal experiments or provide real events.csv.")
            continue
        if name == "maltls22":
            if not audit.ready_for_d5:
                next_actions.append("MALTLS-22 remains unevaluated unless source verification changes.")
            continue
        if not audit.ready_for_d5:
            next_actions.append(f"Resolve {name}: {'; '.join(audit.blocking_reasons)}")
    if not d5_allowed:
        next_actions.append("Do not run D5 until CICIDS and CESNET-TLS-Year22 components pass.")

    return {
        "stage": "realdata-acquisition-audit",
        "submission_ready": False,
        "d5_allowed": d5_allowed,
        "d6_d7_allowed": d5_allowed,
        "datasets": {
            "cicids2017": _readiness_dataset(cicids),
            "maltls22": {
                **_readiness_dataset(maltls),
                "source_verified": DATASET_CONTRACTS["maltls22"].source_verified,
                "replacement_required_if_unavailable": True,
                "evaluated": False,
            },
            "cesnet_tls_year22": _readiness_dataset(cesnet) if cesnet is not None else {
                "available": False,
                "audit_passed": False,
                "ready_for_smoke": False,
                "ready_for_d5": False,
                "blocking_reasons": ["contract not audited"],
            },
            "optc": {
                "available": optc.exists and optc.expected_files_present,
                "audit_passed": optc.ready_for_d5,
                "ready_for_case_study": optc.ready_for_d5,
                "formal_experiment": False,
                "future_case_study_only": True,
                "blocking_reasons": optc.blocking_reasons,
            },
        },
        "next_actions": next_actions,
    }


def _readiness_dataset(result: DatasetAuditResult) -> dict[str, Any]:
    if result is None:
        return {"available": False, "audit_passed": False, "ready_for_smoke": False, "ready_for_d5": False, "blocking_reasons": ["missing audit"]}
    return {
        "available": result.exists and bool(result.files_used),
        "audit_passed": result.ready_for_d5,
        "ready_for_smoke": result.ready_for_smoke,
        "ready_for_d5": result.ready_for_d5,
        "ready_for_d5_component": result.ready_for_d5,
        "blocking_reasons": result.blocking_reasons,
    }


def _files_for_contract(root: Path, contract: DatasetContract) -> tuple[list[Path], list[str]]:
    if contract.expected_files is not None:
        files = [root / name for name in contract.expected_files if (root / name).exists()]
        missing = [name for name in contract.expected_files if not (root / name).exists()]
        return files, missing
    if not root.exists():
        return [], []
    patterns = ("*.csv", "*.csv.gz", "*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.rglob(pattern)))
    return files, []


def _read_frames(files: list[Path]) -> pd.DataFrame:
    frames = [_read_table(path) for path in files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_hash(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(file_hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest()


def _resolve_label_column(frame: pd.DataFrame, contract: DatasetContract) -> str | None:
    if contract.label_column:
        return str(contract.label_column).strip()
    candidates = contract.required_any_columns.get("label", [])
    for candidate in candidates:
        match = _matching_column(frame.columns, candidate)
        if match:
            return match
    return None


def _has_any_column(columns, candidates: list[str]) -> bool:
    return any(_matching_column(columns, candidate) is not None for candidate in candidates)


def _matching_column(columns, candidate: str) -> str | None:
    candidate_clean = str(candidate).strip()
    candidate_lower = candidate_clean.lower()
    for col in columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()
        if col_clean == candidate_clean or col_lower == candidate_lower or candidate_lower in col_lower:
            return col_clean
    return None


def _append_required_any_blockers(contract: DatasetContract, status: dict[str, bool], blocking: list[str]) -> None:
    if contract.name == "cicids2017":
        core = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
        if not any(status.get(key, False) for key in core):
            blocking.append("IP/port/protocol columns are all missing")
        return
    missing = [key for key, ok in status.items() if not ok]
    if missing:
        blocking.append(f"required column groups missing: {', '.join(missing)}")


def _required_any_ready(contract: DatasetContract, status: dict[str, bool]) -> bool:
    if not status:
        return True
    if contract.name == "cicids2017":
        core = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
        return any(status.get(key, False) for key in core)
    return all(status.values())


def _actual_view_support(contract: DatasetContract, any_status: dict[str, bool], columns: set[str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    expected = contract.expected_view_support
    for view, should_exist in expected.items():
        if not should_exist:
            actual[view] = "not_expected"
            continue
        if view == "temporal":
            actual[view] = "available" if any_status.get("timestamp", "timestamp" in columns) else "derived_limited"
        elif view == "process":
            actual[view] = "available" if {"process_id", "parent_process_id"}.issubset(columns) else "missing"
        elif view == "threat_intel":
            actual[view] = "available" if {"alert_type", "risk_score"}.issubset(columns) else "missing"
        elif view == "ip":
            actual[view] = "available" if (
                any_status.get("src_ip", "src_ip" in columns)
                or any_status.get("dst_ip", "dst_ip" in columns)
                or {"src_ip", "dst_ip"}.intersection(columns)
                or any_status.get("tls_or_flow_features", False)
            ) else "missing"
        else:
            actual[view] = "available" if (
                any_status.get("src_ip", False)
                or "host_id" in columns
                or "host" in columns
            ) else "missing"
    return actual


def _numeric_feature_count(frame: pd.DataFrame, label_col: str | None) -> int:
    if frame.empty:
        return 0
    numeric_cols = list(frame.select_dtypes(include=[np.number]).columns)
    return len([col for col in numeric_cols if col != label_col])


def _audit_markdown(results: dict[str, DatasetAuditResult]) -> str:
    lines = ["# Dataset Audit Report", ""]
    for name, result in results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Root: `{result.root}`",
                f"- Available files: {len(result.files_used)}",
                f"- Rows / columns: {result.num_rows} / {result.num_columns}",
                f"- Classes: {result.class_count}",
                f"- Dataset hash: `{result.dataset_hash}`",
                f"- Ready for smoke: {result.ready_for_smoke}",
                f"- Ready for D5: {result.ready_for_d5}",
                f"- View support: {json.dumps(result.actual_view_support, sort_keys=True)}",
                "- Blocking reasons:",
            ]
        )
        if result.blocking_reasons:
            lines.extend([f"  - {reason}" for reason in result.blocking_reasons])
        else:
            lines.append("  - none")
        lines.append("")
    return "\n".join(lines)


def _readiness_markdown(readiness: dict[str, Any]) -> str:
    lines = [
        "# Real-Data Readiness Report",
        "",
        f"- D5 allowed: {readiness['d5_allowed']}",
        f"- D6/D7 allowed: {readiness['d6_d7_allowed']}",
        f"- Submission ready: {readiness['submission_ready']}",
        "",
        "## Datasets",
    ]
    for name, info in readiness["datasets"].items():
        lines.extend(
            [
                "",
                f"### {name}",
                f"- Available: {info.get('available')}",
                f"- Audit passed: {info.get('audit_passed')}",
                f"- Ready for smoke: {info.get('ready_for_smoke', info.get('ready_for_case_study'))}",
                f"- Ready for D5/case: {info.get('ready_for_d5', info.get('ready_for_case_study'))}",
                "- Blocking reasons:",
            ]
        )
        reasons = info.get("blocking_reasons") or []
        lines.extend([f"  - {reason}" for reason in reasons] or ["  - none"])
    lines.extend(["", "## Next Actions"])
    lines.extend([f"- {item}" for item in readiness["next_actions"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONTRACTS), default=None)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    if args.dataset:
        result = audit_dataset(DATASET_CONTRACTS[args.dataset])
        audits = audit_all_datasets()
        audits[args.dataset] = result
        write_audit_reports(audits, args.out)
        write_readiness_reports(audits, args.out)
        write_dataset_specific_audit_report(result, args.out)
        return
    audits = audit_all_datasets()
    write_audit_reports(audits, args.out)
    write_readiness_reports(audits, args.out)


def _imbalance_ratio(counts: dict[str, int]) -> float:
    values = np.asarray(list(counts.values()), dtype=float)
    if values.size == 0 or values.min() <= 0:
        return 0.0
    return float(values.max() / values.min())


def _specific_audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['name']} Audit Report",
        "",
        f"- Root: `{payload['root']}`",
        f"- Exists: {payload['exists']}",
        f"- Rows / columns: {payload['num_rows']} / {payload['num_columns']}",
        f"- Classes: {payload['class_count']}",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        f"- Source verified: {payload.get('source_verified')}",
        f"- Replacement for: {payload.get('replacement_for')}",
        f"- Selected class policy: {payload.get('selected_class_policy')}",
        f"- Active views: {', '.join(payload.get('active_views') or [])}",
        f"- Ready for smoke: {payload['ready_for_smoke']}",
        f"- Ready for mini-matrix: {payload.get('ready_for_mini_matrix')}",
        f"- Ready for D5 component: {payload.get('ready_for_d5_component')}",
        "",
        "## Blocking Reasons",
    ]
    lines.extend([f"- {reason}" for reason in payload.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
