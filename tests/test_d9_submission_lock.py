import json
from pathlib import Path

from pypdf import PdfReader

from src.paper.d9_submission_lock import run_d9_submission_lock


AUDIT = Path("reports/d9/d9_submission_lock_audit.json")
PACKAGE = Path("submission/cas_candidate_d9")


def _ensure_d9() -> dict:
    if not AUDIT.exists():
        run_d9_submission_lock()
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_d9_audit_keeps_submission_ready_false():
    audit = _ensure_d9()
    assert audit["stage"] == "D9"
    assert audit["candidate_package_ready"] is True
    assert audit["submission_ready"] is False
    assert audit["human_confirmation_required"] is True
    assert audit["results_unchanged"] is True
    assert audit["model_code_unchanged_in_worktree"] is True
    assert audit["d8_submission_ready_left_false"] is True


def test_d9_formal_scope_and_result_hashes_are_locked():
    audit = _ensure_d9()
    assert audit["formal_scope_ok"] is True
    assert audit["excluded_scope_declared"] is True
    assert audit["excluded_scope_not_reported"] is True
    assert audit["co_teaching_lite_not_full"] is True
    assert audit["cesnet_not_maltls"] is True
    assert audit["cesnet_subset_declared"] is True
    assert audit["results_hashes"]["results/table_main_expanded.csv"] == (
        "c7d998d6c918ecfbcb9cc56bd494dcec73b3fa6826b2046fb53e2ca2109519cd"
    )
    assert audit["results_hashes"]["results/table_baseline_expansion.csv"] == (
        "b74a3552b9a11b87ee847df2fa5490197fcb4c4fbe59973c7ec3593945b9d158"
    )


def test_d9_package_contains_author_review_and_submission_materials():
    _ensure_d9()
    required = [
        PACKAGE / "author/graph_cold_cas_realdata.pdf",
        PACKAGE / "author/graph_cold_cas_realdata.tex",
        PACKAGE / "review/graph_cold_cas_realdata.pdf",
        PACKAGE / "review/graph_cold_cas_realdata.tex",
        PACKAGE / "submission_materials/highlights.md",
        PACKAGE / "submission_materials/declaration_of_competing_interest.md",
        PACKAGE / "submission_materials/funding_statement.md",
        PACKAGE / "submission_materials/credit_author_statement.md",
        PACKAGE / "submission_materials/data_availability_statement.md",
        PACKAGE / "source_trace/source_hash_manifest.json",
        PACKAGE / "package_manifest.json",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path
    assert len(PdfReader(str(PACKAGE / "author/graph_cold_cas_realdata.pdf")).pages) >= 12
    assert len(PdfReader(str(PACKAGE / "review/graph_cold_cas_realdata.pdf")).pages) >= 12


def test_d9_review_version_is_anonymized_without_scope_drift():
    _ensure_d9()
    review = (PACKAGE / "review/graph_cold_cas_realdata.tex").read_text(encoding="utf-8")
    assert "Anonymous Authors" in review
    assert "Graph-CoLD Project Team" not in review
    assert "Acknowledgements are omitted for anonymous review" in review
    assert "CESNET-TLS-Year22 postfilter25" in review
    assert "MALTLS-22 is not evaluated" in review
    assert "OpTC is not evaluated as a formal enterprise case" in review
    assert "not a full Co-Teaching implementation" in review
    assert "not a full Co-Teaching reproduction" in review


def test_d9_reproducibility_scripts_are_stub_safe_and_do_not_run_new_experiments():
    _ensure_d9()
    for name in [
        "run_d6_paper_assets.ps1",
        "run_d7_build.ps1",
        "run_d8_manuscript.ps1",
        "run_d9_submission_package.ps1",
    ]:
        text = Path("reproducibility", name).read_text(encoding="utf-8")
        assert "GRAPH_COLD_PYTHON" in text
        assert ".cache\\codex-runtimes" in text
        if name != "run_d6_paper_assets.ps1":
            assert "run_d5_experiments" not in text
    d9 = Path("reproducibility/run_d9_submission_package.ps1").read_text(encoding="utf-8")
    assert "python -m src.paper.d9_submission_lock" in d9
