"""Download or prepare CICIDS-2017 CSV files.

Auto mode only uses the official UNB/CIC pages. If a direct CSV zip URL cannot
be discovered, the script writes manual instructions and exits non-zero.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import audit_all_datasets, audit_dataset, write_audit_reports, write_readiness_reports
from src.data.contracts import CICIDS2017_CONTRACT


OFFICIAL_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
DOWNLOAD_FORM = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/"
KEYWORDS = ("MachineLearningCSV", "GeneratedLabelledFlows", "CSV", "ISCX")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.links.append(href)


def run(mode: str, out: str | Path, zip_path: str | Path | None = None) -> dict:
    out_path = Path(out)
    report = _base_report(mode)
    try:
        if mode == "instructions":
            report.update(_manual_state())
        elif mode == "local-zip":
            if zip_path is None or not Path(zip_path).exists():
                report.update(_manual_state())
                report["error"] = f"zip file not found: {zip_path}"
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                _extract_canonical_csvs(Path(zip_path), out_path)
                report["download_success"] = True
                report["manual_action_required"] = False
        elif mode == "auto":
            report["download_attempted"] = True
            url = discover_download_url()
            report["download_url_discovered"] = url
            if not url:
                report.update(_manual_state())
                report["error"] = "Official page did not expose a direct MachineLearningCSV zip URL."
            else:
                archive = _download_zip(url, ROOT / "data" / "_downloads" / "cicids2017")
                out_path.mkdir(parents=True, exist_ok=True)
                _extract_canonical_csvs(archive, out_path)
                report["download_success"] = True
                report["manual_action_required"] = False
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    finally:
        report.update(_presence_and_audit(out_path))
        _write_download_report(report)
    return report


def discover_download_url() -> str | None:
    candidates: list[str] = []
    for page in (OFFICIAL_PAGE, DOWNLOAD_FORM):
        try:
            html = _read_url(page)
        except Exception:
            continue
        parser = LinkParser()
        parser.feed(html)
        for link in parser.links:
            absolute = urllib.parse.urljoin(page, link)
            if any(keyword.lower() in absolute.lower() for keyword in KEYWORDS):
                candidates.append(absolute)
    direct = [url for url in candidates if url.lower().endswith((".zip", ".csv", ".csv.gz"))]
    return direct[0] if direct else None


def _read_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Graph-CoLD dataset audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _download_zip(url: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(url).path).name or "MachineLearningCSV.zip"
    archive = download_dir / filename
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return archive


def _extract_canonical_csvs(zip_path: Path, out: Path) -> None:
    expected = CICIDS2017_CONTRACT.expected_files or []
    by_key = {_norm_name(name): name for name in expected}
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".csv"):
                continue
            basename = Path(member.filename).name
            canonical = by_key.get(_norm_name(basename))
            if canonical is None:
                continue
            target = out / canonical
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            seen.add(canonical)
    missing = sorted(set(expected) - seen)
    if missing:
        raise FileNotFoundError(f"CICIDS-2017 zip did not contain expected CSV files: {', '.join(missing)}")


def _norm_name(name: str) -> str:
    return "".join(char.lower() for char in Path(name).name if char.isalnum())


def _presence_and_audit(out: Path) -> dict:
    expected = CICIDS2017_CONTRACT.expected_files or []
    present = [name for name in expected if (out / name).exists()]
    missing = [name for name in expected if not (out / name).exists()]
    contract = replace(CICIDS2017_CONTRACT, root=str(out))
    audit = audit_dataset(contract)
    audits = audit_all_datasets()
    write_audit_reports(audits)
    write_readiness_reports(audits)
    return {
        "files_expected": expected,
        "files_present": present,
        "missing_files": missing,
        "file_hashes": audit.file_hashes,
        "dataset_hash": audit.dataset_hash,
        "audit_invoked": True,
        "audit_passed": audit.ready_for_d5,
    }


def _base_report(mode: str) -> dict:
    return {
        "dataset": "cicids2017",
        "mode": mode,
        "download_attempted": mode == "auto",
        "download_success": False,
        "official_source_used": True,
        "third_party_mirror_used": False,
        "download_url_discovered": None,
        "manual_action_required": mode != "local-zip",
        "manual_instructions": _instructions(),
        "files_expected": [],
        "files_present": [],
        "missing_files": [],
        "file_hashes": {},
        "dataset_hash": None,
        "audit_invoked": False,
        "audit_passed": False,
    }


def _manual_state() -> dict:
    return {"download_success": False, "manual_action_required": True, "manual_instructions": _instructions()}


def _instructions() -> list[str]:
    return [
        f"Open the official CICIDS-2017 page: {OFFICIAL_PAGE}",
        "Use the official Download this dataset link and complete the CIC form if required.",
        "Download MachineLearningCSV.zip, not the PCAP bundle.",
        "Run: python scripts/download_cicids2017.py --mode local-zip --zip path/to/MachineLearningCSV.zip --out data/cicids2017",
    ]


def _write_download_report(report: dict) -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "dataset_download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Dataset Download Report",
        "",
        f"- Dataset: {report['dataset']}",
        f"- Mode: {report['mode']}",
        f"- Download attempted: {report['download_attempted']}",
        f"- Download success: {report['download_success']}",
        f"- Official source used: {report['official_source_used']}",
        f"- Third-party mirror used: {report['third_party_mirror_used']}",
        f"- Manual action required: {report['manual_action_required']}",
        f"- Files present: {len(report['files_present'])}/{len(report['files_expected'])}",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- Audit passed: {report['audit_passed']}",
        "",
        "## Manual Instructions",
        *[f"- {item}" for item in report["manual_instructions"]],
        "",
    ]
    if report.get("error"):
        lines.extend(["## Error", "", str(report["error"]), ""])
    (reports / "dataset_download_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cicids2017")
    parser.add_argument("--zip", dest="zip_path")
    parser.add_argument("--mode", choices=["auto", "local-zip", "instructions"], default="instructions")
    args = parser.parse_args(argv)
    report = run(args.mode, args.out, args.zip_path)
    print(json.dumps(report, indent=2))
    if args.mode == "instructions":
        return 0
    return 0 if report["download_success"] and not report["missing_files"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
