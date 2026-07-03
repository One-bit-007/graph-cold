"""Second-dataset selection gate for D5 readiness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISPLAY_NAMES = {
    "cicids2017": "CICIDS-2017",
    "cesnet_tls_year22": "CESNET-TLS-Year22",
    "unsw_nb15": "UNSW-NB15",
    "ustc_tfc2016": "USTC-TFC2016",
}


def select_second_dataset(readiness: dict[str, Any] | None = None, reports: str | Path = "reports") -> dict[str, Any]:
    reports_path = Path(reports)
    if readiness is None:
        readiness = _load_readiness(reports_path)
    datasets = readiness.get("datasets", {})
    cicids_ready = _is_ready(datasets.get("cicids2017", {}))
    candidates = ["cesnet_tls_year22", "unsw_nb15", "ustc_tfc2016"]
    selected = None
    rejected: dict[str, list[str]] = {}
    if cicids_ready:
        for candidate in candidates:
            info = datasets.get(candidate, {})
            if _is_ready(info) and not bool(info.get("candidate_only", False)):
                selected = candidate
                break
            rejected[candidate] = _reasons(candidate, info)
    else:
        rejected["cicids2017"] = _reasons("cicids2017", datasets.get("cicids2017", {}))
        for candidate in candidates:
            rejected[candidate] = _reasons(candidate, datasets.get(candidate, {}))

    d5_allowed = bool(cicids_ready and selected)
    scope = ["cicids2017", selected] if d5_allowed and selected else []
    if selected:
        rejected.pop(selected, None)
    manuscript_names = [DISPLAY_NAMES[name] for name in scope]
    report = {
        "stage": "second-dataset-selection",
        "d5_allowed": d5_allowed,
        "selected_second_dataset": selected,
        "d5_scope": scope,
        "manuscript_dataset_names": manuscript_names,
        "rejected_candidates": rejected,
        "paper_claims_changed": bool(selected and selected != "cesnet_tls_year22"),
        "blocking_reasons": [] if d5_allowed else _blocking_summary(cicids_ready, rejected),
        "maltls22_allowed": False,
        "optc_formal_experiment": False,
    }
    return report


def write_selection_gate(reports: str | Path = "reports", readiness: dict[str, Any] | None = None) -> dict[str, Any]:
    report = select_second_dataset(readiness, reports)
    out = Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "second_dataset_selection_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "second_dataset_selection_gate.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _load_readiness(reports: Path) -> dict[str, Any]:
    primary = reports / "realdata_readiness_report.json"
    secondary = reports / "two_dataset_readiness_report.json"
    if primary.exists():
        return json.loads(primary.read_text(encoding="utf-8"))
    if secondary.exists():
        legacy = json.loads(secondary.read_text(encoding="utf-8"))
        return _legacy_to_readiness(legacy)
    return {"datasets": {}}


def _legacy_to_readiness(legacy: dict[str, Any]) -> dict[str, Any]:
    datasets = {
        "cicids2017": {
            "ready_for_d5_component": bool(legacy.get("cicids2017", {}).get("ready_for_d5_component", False)),
            "blocking_reasons": legacy.get("cicids2017", {}).get("blocking_reasons", []),
        },
        "cesnet_tls_year22": {
            "ready_for_d5_component": bool(legacy.get("cesnet_tls_year22", {}).get("ready_for_d5_component", False)),
            "blocking_reasons": legacy.get("cesnet_tls_year22", {}).get("blocking_reasons", []),
        },
    }
    return {"datasets": datasets}


def _is_ready(info: dict[str, Any]) -> bool:
    return bool(
        info.get("ready_for_d5_component")
        or info.get("ready_for_d5")
        or info.get("audit_passed")
    )


def _reasons(candidate: str, info: dict[str, Any]) -> list[str]:
    reasons = list(info.get("blocking_reasons") or [])
    if candidate == "ustc_tfc2016":
        reasons.append("candidate_only; not selected unless user confirms download and audit")
    if not _is_ready(info):
        reasons.append("not ready for D5 component")
    if candidate == "maltls22":
        reasons.append("source unverified")
    return reasons or ["not selected by priority order"]


def _blocking_summary(cicids_ready: bool, rejected: dict[str, list[str]]) -> list[str]:
    reasons = []
    if not cicids_ready:
        reasons.append("CICIDS-2017 is not ready")
    if rejected:
        reasons.append("No verified second dataset is ready")
    return reasons


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Second Dataset Selection Gate",
        "",
        f"- D5 allowed: {report['d5_allowed']}",
        f"- Selected second dataset: {report['selected_second_dataset'] or 'none'}",
        f"- D5 scope: {', '.join(report['d5_scope']) if report['d5_scope'] else 'none'}",
        f"- Manuscript dataset names: {', '.join(report['manuscript_dataset_names']) if report['manuscript_dataset_names'] else 'none'}",
        f"- Paper claims changed: {report['paper_claims_changed']}",
        "- MALTLS-22 allowed: false",
        "- OpTC formal experiment: false",
        "",
        "## Rejected Candidates",
    ]
    if report["rejected_candidates"]:
        for name, reasons in report["rejected_candidates"].items():
            lines.append(f"- {name}: {'; '.join(reasons)}")
    else:
        lines.append("- none")
    lines.extend(["", "## Blocking Reasons"])
    lines.extend([f"- {reason}" for reason in report["blocking_reasons"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    print(json.dumps(write_selection_gate(args.reports), indent=2))


if __name__ == "__main__":
    main()
