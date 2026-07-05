import json
from pathlib import Path

from pypdf import PdfReader

from src.paper.d8_harden import run_d8_hardening


MANUSCRIPT = Path("paper/elsevier/graph_cold_cas_realdata.tex")


def _ensure_d8() -> str:
    if not Path("reports/d8/d8_hardening_audit.json").exists():
        run_d8_hardening()
    return MANUSCRIPT.read_text(encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_d8_manuscript_has_submission_grade_structure():
    text = _ensure_d8()
    normalized = _norm(text)
    for required in [
        r"\section{Introduction}",
        r"\section{Related Work}",
        r"\section{Problem Formulation and Design Goals}",
        r"\section{Method}",
        r"\section{Experimental Design}",
        r"\section{Results}",
        r"\section{Discussion}",
        r"\section{Threats to Validity}",
        r"\section{Limitations}",
        r"\section{Conclusion}",
    ]:
        assert required in text
    assert "This paper makes four contributions" in normalized
    for rq in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]:
        assert rq in text
    assert "not a full Co-Teaching reproduction" in normalized
    assert "not a full-archive evaluation" in normalized or "not a full archive" in normalized


def test_d8_manuscript_keeps_scope_and_results_traceable():
    text = _ensure_d8()
    normalized = _norm(text)
    assert "results/table\\_main\\_expanded.csv" in text
    assert "CICIDS-2017 postfilter11" in normalized
    assert "CESNET-TLS-Year22" in normalized
    assert "MALTLS-22 is not evaluated" in normalized
    assert "OpTC is not evaluated as a formal enterprise case" in normalized
    assert "Graph-CoLD improves Macro-F1 over the aligned CoLD baseline by 1.83 percentage points" in normalized
    assert "ERR\\_final from 0.8953" in text


def test_d8_audit_and_pdf_are_ready_for_v1_review():
    _ensure_d8()
    audit = json.loads(Path("reports/d8/d8_hardening_audit.json").read_text(encoding="utf-8"))
    assert audit["initial_draft_v1_ready"] is True
    assert audit["final_submission_ready"] is False
    for key in [
        "real_data_only",
        "results_unchanged",
        "uses_table_main_expanded",
        "no_maltls22_results",
        "no_optc_results",
        "references_corrected",
        "threats_to_validity_included",
        "research_questions_included",
    ]:
        assert audit[key] is True, key

    pdf = Path("paper/elsevier/graph_cold_cas_realdata.pdf")
    assert pdf.exists()
    reader = PdfReader(str(pdf))
    assert len(reader.pages) >= 12


def test_d8_references_correct_known_d7_metadata():
    _ensure_d8()
    refs = Path("paper/elsevier/references.bib").read_text(encoding="utf-8")
    assert "Collaborative Label Denoising Framework for Network Intrusion Detection" in refs
    assert "Yang, Shuo" in refs
    assert "10.1038/s41597-024-03927-4" in refs
    assert "10.5281/zenodo.10608607" in refs
    for term in ["TODO", "TBD", "fake", "unknown"]:
        assert term.lower() not in refs.lower()


def test_d8_no_forbidden_overclaiming_or_generated_result_language():
    text = _ensure_d8()
    lowered = text.lower()
    forbidden = [
        "state-of-the-art",
        "near-perfect",
        "beats all baselines",
        "synthetic result",
        "synthetic dataset",
        "draft placeholder",
        "opTC results".lower(),
        "MALTLS-22 results".lower(),
    ]
    for term in forbidden:
        assert term not in lowered


def test_d8_reproducibility_entrypoint_exists():
    _ensure_d8()
    script = Path("reproducibility/run_d8_manuscript.ps1")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "python -m src.paper.d8_harden" in text
    assert "build_elsevier.ps1" in text
    assert "run_d5_experiments" not in text
