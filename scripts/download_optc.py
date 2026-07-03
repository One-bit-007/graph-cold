"""Prepare OpTC provenance events for the enterprise case gate."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import audit_all_datasets, audit_dataset, write_audit_reports, write_readiness_reports
from src.data.contracts import OPTC_CONTRACT


OPTC_SOURCE = "https://github.com/FiveDirections/OpTC-data"


def run(
    mode: str,
    out: str | Path = "data/optc",
    events: str | Path | None = None,
    confirm_large_download: bool = False,
) -> dict:
    out_path = Path(out)
    report = _base_report(mode, out_path, confirm_large_download)
    if mode == "instructions":
        pass
    elif mode == "manifest":
        _write_manifest(out_path)
        report["manifest_written"] = True
    elif mode == "local-events":
        if events is None or not Path(events).exists():
            report["error"] = f"events.csv not found: {events}"
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(events), out_path / "events.csv")
            report["events_copied"] = True
    elif mode == "gdrive":
        report["full_download_attempted"] = bool(confirm_large_download)
        report["error"] = "Full OpTC download is not automated here; use source instructions and convert to events.csv."
    else:
        raise ValueError(f"Unsupported OpTC mode: {mode}")
    report.update(_audit(out_path))
    _write_report(report)
    return report


def _base_report(mode: str, out: Path, confirm_large_download: bool) -> dict:
    return {
        "dataset": "optc",
        "mode": mode,
        "source": OPTC_SOURCE,
        "out": str(out),
        "events_csv_present": False,
        "audit_invoked": True,
        "audit_passed": False,
        "full_download_attempted": False,
        "large_download_confirmed": confirm_large_download,
        "manual_action_required": True,
        "manifest_written": False,
        "events_copied": False,
        "manual_instructions": [
            f"Review OpTC data source: {OPTC_SOURCE}",
            "Obtain the authorized OpTC release through the official distribution channel.",
            "Convert eCAR/Bro/JSON provenance records into data/optc/events.csv.",
            "Required columns: host_id, process_id, parent_process_id, src_ip, dst_ip, timestamp, event_type, alert_type, label, risk_score.",
            "Run: python scripts/download_optc.py --mode local-events --events path/to/events.csv --out data/optc",
        ],
    }


def _write_manifest(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "optc",
        "required_output": "events.csv",
        "required_columns": OPTC_CONTRACT.required_columns,
        "source": OPTC_SOURCE,
        "notes": "Manifest only; no raw OpTC files downloaded.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _audit(out: Path) -> dict:
    contract = replace(OPTC_CONTRACT, root=str(out))
    audit = audit_dataset(contract)
    audits = audit_all_datasets()
    write_audit_reports(audits)
    write_readiness_reports(audits)
    return {
        "events_csv_present": (out / "events.csv").exists(),
        "file_hashes": audit.file_hashes,
        "dataset_hash": audit.dataset_hash,
        "audit_passed": audit.ready_for_d5,
        "blocking_reasons": audit.blocking_reasons,
    }


def _write_report(report: dict) -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "optc_download_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# OpTC Preparation Report",
        "",
        f"- Mode: {report['mode']}",
        f"- Events CSV present: {report['events_csv_present']}",
        f"- Audit passed: {report['audit_passed']}",
        f"- Full download attempted: {report['full_download_attempted']}",
        f"- Dataset hash: `{report.get('dataset_hash')}`",
        "",
        "## Instructions",
        *[f"- {item}" for item in report["manual_instructions"]],
        "",
    ]
    if report.get("error"):
        md.extend(["## Error", "", str(report["error"]), ""])
    (reports / "optc_download_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["instructions", "manifest", "local-events", "gdrive"], default="instructions")
    parser.add_argument("--events")
    parser.add_argument("--out", default="data/optc")
    parser.add_argument("--confirm-large-download", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.mode, args.out, args.events, args.confirm_large_download)
    print(json.dumps(report, indent=2))
    if args.mode in {"instructions", "manifest"}:
        return 0
    return 0 if report["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
