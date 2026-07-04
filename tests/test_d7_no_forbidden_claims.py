from pathlib import Path

from src.paper.d7_assemble import run_d7_assembly


def _ensure_outputs() -> None:
    if not Path("paper/elsevier/graph_cold_cas_realdata.tex").exists():
        run_d7_assembly()


def _text_paths():
    _ensure_outputs()
    paths = [
        Path("paper/elsevier/graph_cold_cas_realdata.tex"),
        Path("paper/elsevier/cover_letter_draft.md"),
        Path("reproducibility/README_realdata.md"),
        Path("reports/d7/d7_readiness_reconciliation.md"),
        Path("reports/d7/reviewer_simulation_final.md"),
        Path("reports/d7/rebuttal_preparation_pack.md"),
        Path("reports/d7/reference_gap_report.md"),
    ]
    return [path for path in paths if path.exists()]


def test_d7_text_outputs_do_not_report_generated_or_placeholder_results():
    forbidden = [
        "synthetic result",
        "synthetic dataset",
        "deterministic soc",
        "draft placeholder",
        "to be refreshed later",
        "MALTLS-22 results",
        "OpTC results",
    ]
    for path in _text_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for term in forbidden:
            assert term.lower() not in lowered, f"{term} found in {path}"


def test_d7_manuscript_avoids_overclaiming_language():
    forbidden_claims = [
        "state-of-the-art dominance",
        "near-perfect",
        "massive gain",
        "causal proof",
        "beats all baselines",
        "full archive result",
    ]
    manuscript = Path("paper/elsevier/graph_cold_cas_realdata.tex").read_text(encoding="utf-8", errors="ignore")
    lowered = manuscript.lower()
    for term in forbidden_claims:
        assert term not in lowered

    assert "submission_ready\": true" not in Path("reports/d7/d7_final_audit.json").read_text(
        encoding="utf-8", errors="ignore"
    )
