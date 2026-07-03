"""Audit external storage before large real-dataset downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(data_root: str | Path, required_gb: float = 80.0, reports: str | Path = "reports") -> dict:
    root = Path(data_root)
    volume = root.drive or root.anchor
    exists = root.exists() or root.parent.exists()
    blocking_reason = None
    free_gb = None
    if not exists:
        blocking_reason = f"volume or parent path does not exist: {root.parent}"
    else:
        usage_target = root if root.exists() else root.parent
        free_gb = shutil.disk_usage(usage_target).free / 1024**3
        if free_gb < required_gb:
            blocking_reason = f"free space {free_gb:.2f}GB below required {required_gb:.2f}GB"
    report = {
        "requested_data_root": str(root),
        "volume": volume,
        "free_gb": free_gb,
        "required_gb": required_gb,
        "download_allowed": blocking_reason is None,
        "blocking_reason": blocking_reason,
    }
    write_reports(report, reports)
    return report


def write_reports(report: dict, reports: str | Path = "reports") -> None:
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "storage_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Storage Audit Report",
        "",
        f"- Requested data root: `{report['requested_data_root']}`",
        f"- Volume: `{report['volume']}`",
        f"- Free GB: {report['free_gb']}",
        f"- Required GB: {report['required_gb']}",
        f"- Download allowed: {report['download_allowed']}",
        f"- Blocking reason: {report['blocking_reason'] or 'none'}",
        "",
    ]
    (out / "storage_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--required-gb", type=float, default=80.0)
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args(argv)
    report = run(args.data_root, args.required_gb, args.reports)
    print(json.dumps(report, indent=2))
    return 0 if report["download_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
