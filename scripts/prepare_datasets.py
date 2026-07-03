"""Unified dataset preparation entry point.

This script dispatches to the dataset-specific preparation helpers, refreshes
audit/readiness reports, and never invokes the D5 full matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import download_cicids2017, download_optc, download_tls_alternative
from src.data.audit import audit_all_datasets, write_audit_reports, write_readiness_reports


def run(args) -> dict:
    if args.dataset == "cicids2017":
        result = download_cicids2017.run(args.mode, args.out or "data/cicids2017", args.zip_path)
    elif args.dataset == "tls_alternative":
        result = download_tls_alternative.run(
            args.candidate,
            args.mode,
            args.out or f"data/tls_alternative/{args.candidate}",
            args.confirm_large_download,
            getattr(args, "archive", None),
            getattr(args, "data_root", None),
            getattr(args, "download_cache", None),
            getattr(args, "min_free_gb", None),
        )
    elif args.dataset == "optc":
        result = download_optc.run(args.mode, args.out or "data/optc", args.events, args.confirm_large_download)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    audits = audit_all_datasets()
    write_audit_reports(audits)
    write_readiness_reports(audits)
    result["readiness_refreshed"] = True
    result["d5_full_matrix_invoked"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cicids2017", "tls_alternative", "optc"], required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--out")
    parser.add_argument("--zip", dest="zip_path")
    parser.add_argument("--archive")
    parser.add_argument("--data-root")
    parser.add_argument("--download-cache")
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--candidate", default="cesnet_tls_year22")
    parser.add_argument("--events")
    parser.add_argument("--confirm-large-download", action="store_true")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    if args.mode in {"instructions", "manifest"}:
        return 0
    return 0 if bool(result.get("audit_passed") or result.get("download_success")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
