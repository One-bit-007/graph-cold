"""Prepare public TLS replacement candidates without renaming them as MALTLS-22."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import audit_all_datasets, audit_dataset, write_audit_reports, write_dataset_specific_audit_report, write_readiness_reports
from src.data.contracts import CESNET_TLS_YEAR22_CONTRACT
from src.data.paths import get_download_cache, resolve_dataset_path


CANDIDATES = {
    "cesnet_tls_year22": {
        "label": "CESNET-TLS-Year22",
        "url": "https://zenodo.org/records/10608607",
        "datazoo": "https://cesnet.github.io/cesnet-datazoo/",
        "large": True,
        "notes": "Year-spanning TLS flow/service dataset; large public research corpus.",
    },
    "cesnet_tls22": {
        "label": "CESNET-TLS22",
        "url": "https://www.liberouter.org/technology-v2/tools-services-datasets/datasets/cesnet-tls22/",
        "large": True,
        "notes": "Earlier CESNET TLS traffic classification dataset; separate from CESNET-TLS-Year22.",
    },
    "ustc_tfc2016": {
        "label": "USTC-TFC2016",
        "url": "https://github.com/yungshenglu/USTC-TFC2016",
        "large": False,
        "notes": "Traffic classification dataset; requires separate schema contract before use.",
    },
    "malicious_tls": {
        "label": "Malicious_TLS",
        "url": "https://github.com/ojroques/tls-malware-detection",
        "large": False,
        "notes": "TLS malware detection project/data references; verify license and labels before use.",
    },
}


def run(
    candidate: str,
    mode: str,
    out: str | Path,
    confirm_large_download: bool = False,
    archive: str | Path | None = None,
    data_root: str | Path | None = None,
    download_cache: str | Path | None = None,
    min_free_gb: float | None = None,
) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown TLS alternative candidate: {candidate}")
    info = CANDIDATES[candidate]
    out_path = Path(out) if out else resolve_dataset_path(candidate, data_root)
    cache_path = get_download_cache(data_root, download_cache) / candidate
    attempted = mode in {"auto", "datazoo"} and (confirm_large_download or not info["large"])
    report = {
        "dataset": "tls_alternative",
        "candidate": candidate,
        "must_be_reported_as": info["label"],
        "not_maltls22": True,
        "mode": mode,
        "download_attempted": attempted,
        "download_success": False,
        "large_download_confirmed": confirm_large_download,
        "download_source": info["url"],
        "datazoo_source": info.get("datazoo"),
        "manual_action_required": True,
        "out": str(out_path),
        "data_root": str(data_root) if data_root else None,
        "download_cache": str(cache_path),
        "min_free_gb": min_free_gb,
        "audit_passed": False,
        "notes": info["notes"],
    }
    if mode in {"auto", "datazoo"} and info["large"] and not confirm_large_download:
        report["error"] = "Large TLS download requires --confirm-large-download."
    elif mode == "auto":
        report.update(_prepare_zenodo_auto(info, out_path, cache_path, min_free_gb))
    elif mode == "datazoo":
        report.update(_prepare_datazoo(info, out_path))
    elif mode == "local-archive":
        report.update(_prepare_local_archive(archive, out_path))
    elif mode != "instructions":
        raise ValueError("TLS alternative supports modes: instructions, auto, datazoo, local-archive")
    if candidate == "cesnet_tls_year22":
        report.update(_audit_cesnet(out_path))
    _write_tls_reports(report)
    audits = audit_all_datasets()
    write_audit_reports(audits)
    write_readiness_reports(audits)
    return report


def _prepare_datazoo(info: dict, out_path: Path) -> dict:
    """Record guarded DataZoo status without inventing files."""
    try:
        import cesnet_datazoo  # type: ignore  # noqa: F401
    except Exception as exc:
        return {
            "download_success": False,
            "manual_action_required": True,
            "datazoo_available": False,
            "error": f"CESNET DataZoo package is not available in this environment: {exc}",
            "manual_instructions": _manual_instructions(info, out_path),
        }
    return {
        "download_success": False,
        "manual_action_required": True,
        "datazoo_available": True,
        "error": "DataZoo package is installed, but this project requires an explicit export path/schema mapping before audit.",
        "manual_instructions": _manual_instructions(info, out_path),
    }


def _prepare_zenodo_auto(info: dict, out_path: Path, cache_path: Path, min_free_gb: float | None) -> dict:
    try:
        manifest = _zenodo_manifest(info["url"])
    except Exception as exc:
        return {
            "download_success": False,
            "manual_action_required": True,
            "error": f"Could not read Zenodo file manifest: {exc}",
            "manual_instructions": _manual_instructions(info, out_path),
        }
    total_size = sum(int(file.get("size", 0)) for file in manifest)
    free = shutil.disk_usage(out_path.resolve().anchor or ".").free
    minimum_free = 0 if min_free_gb is None else int(min_free_gb * 1024**3)
    needed = total_size + max(5 * 1024**3, minimum_free)
    if total_size and free < needed:
        return {
            "download_success": False,
            "manual_action_required": True,
            "zenodo_files": manifest,
            "total_size_bytes": total_size,
            "free_space_bytes": free,
            "error": (
                "Insufficient free disk space for CESNET-TLS-Year22. "
                f"Need at least {needed} bytes including extraction margin; free space is {free} bytes."
            ),
            "manual_instructions": _manual_instructions(info, out_path),
        }
    out_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    downloaded = []
    archive_hashes = {}
    try:
        for file in manifest:
            target = cache_path / file["key"]
            _download_file(file["url"], target, expected_size=int(file.get("size", 0)))
            archive_hashes[str(target)] = _sha256(target)
            downloaded.append(str(target))
            if zipfile.is_zipfile(target):
                _extract_archive(target, out_path)
            elif target.suffix.lower() in {".csv", ".parquet"} or "".join(target.suffixes).lower().endswith(".csv.gz"):
                shutil.copy2(target, out_path / target.name)
    except Exception as exc:
        return {
            "download_success": False,
            "manual_action_required": True,
            "zenodo_files": manifest,
            "archive_files": downloaded,
            "archive_hashes": archive_hashes,
            "error": f"Zenodo download or extraction failed: {exc}",
            "manual_instructions": _manual_instructions(info, out_path),
        }
    return {
        "download_success": True,
        "manual_action_required": False,
        "zenodo_files": manifest,
        "archive_files": downloaded,
        "archive_hashes": archive_hashes,
        "files_present": [str(path) for path in sorted(out_path.rglob("*")) if path.is_file()][:200],
        "actual_data_path": str(out_path),
    }


def _zenodo_manifest(record_url: str) -> list[dict]:
    record_id = record_url.rstrip("/").split("/")[-1]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "Graph-CoLD dataset audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    files = []
    for item in payload.get("files", []):
        files.append(
            {
                "key": item.get("key"),
                "size": int(item.get("size", 0)),
                "url": item.get("links", {}).get("self"),
            }
        )
    return files


def _download_file(url: str, target: Path, expected_size: int = 0) -> None:
    if target.exists() and expected_size and target.stat().st_size == expected_size:
        return
    partial = target.with_suffix(target.suffix + ".part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "60",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        subprocess.run(cmd, check=True)
        if expected_size and partial.stat().st_size != expected_size:
            raise IOError(f"downloaded size mismatch for {target.name}: {partial.stat().st_size} != {expected_size}")
        partial.replace(target)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Graph-CoLD dataset audit"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(target)


def _extract_archive(archive_path: Path, out_path: Path) -> None:
    out_resolved = out_path.resolve()
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as handle:
            for member in handle.infolist():
                target = (out_path / member.filename).resolve()
                if not str(target).startswith(str(out_resolved)):
                    raise ValueError(f"Unsafe archive member path: {member.filename}")
            handle.extractall(out_path)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as handle:
            for member in handle.getmembers():
                target = (out_path / member.name).resolve()
                if not str(target).startswith(str(out_resolved)):
                    raise ValueError(f"Unsafe archive member path: {member.name}")
            handle.extractall(out_path)


def _prepare_local_archive(archive: str | Path | None, out_path: Path) -> dict:
    if archive is None:
        return {"download_success": False, "error": "--archive is required for local-archive mode."}
    archive_path = Path(archive)
    if not archive_path.exists():
        return {"download_success": False, "error": f"Archive not found: {archive_path}"}
    out_path.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        _extract_archive(archive_path, out_path)
    elif tarfile.is_tarfile(archive_path):
        _extract_archive(archive_path, out_path)
    elif archive_path.is_file():
        shutil.copy2(archive_path, out_path / archive_path.name)
    return {
        "download_success": True,
        "manual_action_required": False,
        "archive": str(archive_path),
        "archive_hashes": {str(archive_path): _sha256(archive_path)} if archive_path.is_file() else {},
        "actual_data_path": str(out_path),
        "files_present": [str(path) for path in sorted(out_path.rglob("*")) if path.is_file()][:50],
    }


def _manual_instructions(info: dict, out_path: Path) -> list[str]:
    return [
        f"Download CESNET-TLS-Year22 from {info['url']} or export it through {info.get('datazoo')}.",
        f"Place CSV/Parquet/DataZoo-exported files under {out_path}.",
        "If using a local archive, run: python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode local-archive --archive path/to/archive --out data/tls_alternative/cesnet_tls_year22",
        "Then run: python -m src.data.audit --dataset cesnet_tls_year22",
    ]


def _audit_cesnet(out_path: Path) -> dict:
    audit = audit_dataset(replace(CESNET_TLS_YEAR22_CONTRACT, root=str(out_path)))
    if out_path == Path(CESNET_TLS_YEAR22_CONTRACT.root):
        write_dataset_specific_audit_report(audit)
    return {
        "audit_passed": audit.ready_for_d5,
        "dataset_hash": audit.dataset_hash,
        "files": audit.files_used,
        "rows": audit.num_rows,
        "classes": audit.class_count,
        "blocking_reasons": audit.blocking_reasons,
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_tls_reports(report: dict) -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if report["candidate"] == "cesnet_tls_year22":
        (reports / "cesnet_download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (reports / "cesnet_download_report.md").write_text(_download_markdown(report), encoding="utf-8")
    (reports / "tls_replacement_decision.json").write_text(json.dumps(_decision(report), indent=2), encoding="utf-8")
    (reports / "tls_replacement_decision.md").write_text(_decision_markdown(report), encoding="utf-8")


def _decision(report: dict) -> dict:
    return {
        "maltls22": {
            "source_verified": False,
            "available": False,
            "ready_for_d5": False,
        },
        "recommended_replacement": report["candidate"],
        "replacement_download_source": report["download_source"],
        "replacement_download_attempted": report["download_attempted"],
        "replacement_download_success": report["download_success"],
        "large_download_confirmed": report["large_download_confirmed"],
        "must_be_reported_as": report["must_be_reported_as"],
        "allowed_for_d5": False,
        "error": report.get("error"),
        "dataset_hash": report.get("dataset_hash"),
        "files": report.get("files", []),
        "user_confirmation_required": True,
        "notes": [
            "The replacement is not MALTLS-22.",
            "If selected, the paper must name the replacement dataset explicitly.",
            "A separate audited contract is required before experiments.",
        ],
    }


def _decision_markdown(report: dict) -> str:
    return "\n".join(
        [
            "# TLS Replacement Decision",
            "",
            "- MALTLS-22 source verified: false",
            "- MALTLS-22 allowed for D5: false",
            f"- Recommended replacement candidate: {report['must_be_reported_as']}",
            f"- Download source: {report['download_source']}",
            f"- DataZoo source: {report.get('datazoo_source')}",
            "- Replacement differs from MALTLS-22 and must not be renamed.",
            f"- Download attempted: {report['download_attempted']}",
            f"- Large download confirmed: {report['large_download_confirmed']}",
            f"- Error: {report.get('error') or 'none'}",
            "- Allowed for D5 now: false",
            "- User confirmation required: true",
            "",
            "If this replacement is later selected and passes audit, the manuscript must report the dataset as "
            f"`CICIDS-2017 + {report['must_be_reported_as']}`.",
            "",
        ]
    )


def _download_markdown(report: dict) -> str:
    lines = [
        "# CESNET-TLS-Year22 Download Report",
        "",
        f"- Mode: {report['mode']}",
        f"- Download attempted: {report['download_attempted']}",
        f"- Download success: {report['download_success']}",
        f"- Large download confirmed: {report['large_download_confirmed']}",
        f"- Source: {report['download_source']}",
        f"- DataZoo source: {report.get('datazoo_source')}",
        f"- Output path: `{report['out']}`",
        f"- Dataset hash: `{report.get('dataset_hash')}`",
        f"- Rows/classes: {report.get('rows')} / {report.get('classes')}",
        f"- Error: {report.get('error') or 'none'}",
        "",
        "## Files",
    ]
    for item in report.get("zenodo_files", []):
        lines.append(f"- {item.get('key')}: {item.get('size')} bytes")
    for item in report.get("files", []):
        lines.append(f"- `{item}`")
    if not report.get("zenodo_files") and not report.get("files"):
        lines.append("- none")
    lines.extend(["", "## Manual Instructions"])
    lines.extend([f"- {item}" for item in report.get("manual_instructions", [])] or ["- none"])
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {item}" for item in report.get("blocking_reasons", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="cesnet_tls_year22")
    parser.add_argument("--mode", choices=["instructions", "auto", "datazoo", "local-archive"], default="instructions")
    parser.add_argument("--out", default="data/tls_alternative/cesnet_tls_year22")
    parser.add_argument("--archive")
    parser.add_argument("--data-root")
    parser.add_argument("--download-cache")
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--confirm-large-download", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        args.candidate,
        args.mode,
        args.out,
        args.confirm_large_download,
        args.archive,
        args.data_root,
        args.download_cache,
        args.min_free_gb,
    )
    print(json.dumps(report, indent=2))
    return 0 if args.mode in {"instructions", "local-archive"} and not report.get("error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
