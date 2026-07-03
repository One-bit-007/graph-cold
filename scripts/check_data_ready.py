"""Print the current real-data readiness gate state."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    reports = ROOT / "reports"
    readiness = _load_json(reports / "realdata_readiness_report.json")
    audit = _load_json(reports / "dataset_audit_report.json")
    tls = _load_json(reports / "tls_replacement_decision.json")

    cicids = readiness.get("datasets", {}).get("cicids2017", {})
    maltls = readiness.get("datasets", {}).get("maltls22", {})
    cesnet = readiness.get("datasets", {}).get("cesnet_tls_year22", {})
    optc = readiness.get("datasets", {}).get("optc", {})
    tls_ready = "not selected"
    if tls:
        tls_ready = "ready" if tls.get("allowed_for_d5") else "blocked"

    lines = [
        f"CICIDS-2017: {'ready' if cicids.get('ready_for_d5') else 'blocked'}",
        f"CESNET-TLS-Year22: {'ready' if cesnet.get('ready_for_d5') or cesnet.get('ready_for_d5_component') else 'blocked'}",
        f"MALTLS-22: {'source verified' if maltls.get('source_verified') else 'source unverified / blocked'}",
        f"TLS alternative: {tls_ready}",
        f"OpTC: {'case-ready' if optc.get('ready_for_case_study') else 'future case / unavailable'}",
        f"D5 allowed: {str(readiness.get('d5_allowed', False)).lower()}",
        f"Smoke allowed: {str(bool(cicids.get('ready_for_smoke') or maltls.get('ready_for_smoke'))).lower()}",
        "Submission ready: false",
    ]
    print("\n".join(lines))
    if not audit:
        print("Audit report missing: run python -m src.data.audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
