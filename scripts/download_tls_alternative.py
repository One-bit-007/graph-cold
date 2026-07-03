"""Prepare public TLS replacement candidates without renaming them as MALTLS-22."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import write_readiness_reports


CANDIDATES = {
    "cesnet_tls_year22": {
        "label": "CESNET-TLS-Year22",
        "url": "https://zenodo.org/records/7969895",
        "large": True,
        "notes": "Year-spanning TLS metadata dataset; large public research corpus.",
    },
    "cesnet_tls22": {
        "label": "CESNET-TLS-Year22",
        "url": "https://zenodo.org/records/7969895",
        "large": True,
        "notes": "Alias for CESNET-TLS-Year22.",
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


def run(candidate: str, mode: str, out: str | Path, confirm_large_download: bool = False) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown TLS alternative candidate: {candidate}")
    info = CANDIDATES[candidate]
    out_path = Path(out)
    attempted = mode == "auto" and (confirm_large_download or not info["large"])
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
        "manual_action_required": True,
        "out": str(out_path),
        "audit_passed": False,
        "notes": info["notes"],
    }
    if mode == "auto" and info["large"] and not confirm_large_download:
        report["error"] = "Large TLS download requires --confirm-large-download."
    elif mode == "auto":
        report["error"] = "Auto download is intentionally conservative; follow source instructions and audit local files."
        _probe_source(info["url"])
    elif mode != "instructions":
        raise ValueError("TLS alternative supports modes: instructions, auto")
    _write_tls_reports(report)
    write_readiness_reports()
    return report


def _probe_source(url: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Graph-CoLD dataset audit"})
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read(1024)


def _write_tls_reports(report: dict) -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
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
            "- Replacement differs from MALTLS-22 and must not be renamed.",
            f"- Download attempted: {report['download_attempted']}",
            f"- Large download confirmed: {report['large_download_confirmed']}",
            "- Allowed for D5 now: false",
            "- User confirmation required: true",
            "",
            "If this replacement is later selected and passes audit, the manuscript must report the dataset as "
            f"`CICIDS-2017 + {report['must_be_reported_as']}`.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="cesnet_tls_year22")
    parser.add_argument("--mode", choices=["instructions", "auto"], default="instructions")
    parser.add_argument("--out", default="data/tls_alternative/cesnet_tls_year22")
    parser.add_argument("--confirm-large-download", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.candidate, args.mode, args.out, args.confirm_large_download)
    print(json.dumps(report, indent=2))
    return 0 if args.mode == "instructions" else 2


if __name__ == "__main__":
    raise SystemExit(main())

