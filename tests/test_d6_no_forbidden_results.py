from pathlib import Path

from src.paper.d6_prep import run_d6_realdata_prep


def _ensure_d6_outputs():
    if not Path("reports/d6/d6_generation_manifest.json").exists():
        run_d6_realdata_prep()


def _generated_text_files():
    roots = [Path("tables"), Path("reports/d6"), Path("paper/sections")]
    for root in roots:
        for suffix in ("*.csv", "*.md", "*.json", "*.tex"):
            yield from root.glob(suffix)


def test_generated_d6_outputs_do_not_use_forbidden_result_terms():
    _ensure_d6_outputs()
    forbidden = ("synthetic", "fallback", "emulation")
    for path in _generated_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden:
            assert term not in text, f"{term} found in {path}"


def test_generated_d6_outputs_avoid_overclaiming_and_final_pdf():
    _ensure_d6_outputs()
    forbidden_claims = ("dominates", "massive gain", "state-of-the-art", "near-perfect", "causal proof")
    for path in _generated_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden_claims:
            assert term not in text, f"{term} found in {path}"

    assert not Path("paper/graph_cold_cas_submission.pdf").exists()
    assert not Path("paper/graph_cold_cas_submission.tex").exists()
