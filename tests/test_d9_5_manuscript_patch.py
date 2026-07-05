import json
from pathlib import Path

import pytest
from pypdf import PdfReader


def test_reinforced_manuscript_if_present_uses_fine_style_name_and_keeps_submission_false():
    audit_path = Path("reports/d9_5/d9_5_final_audit.json")
    tex_path = Path("paper/elsevier/graph_cold_cas_realdata_reinforced.tex")
    pdf_path = Path("paper/elsevier/graph_cold_cas_realdata_reinforced.pdf")
    if not audit_path.exists() or not tex_path.exists():
        pytest.skip("D9.5 manuscript patch has not been generated in this checkout.")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    text = tex_path.read_text(encoding="utf-8")

    assert audit["submission_ready"] is False
    assert audit["fine_not_misnamed"] is True
    assert "Decoupling" in text
    if audit.get("fine_style_smoke_passed"):
        assert "FINE-style" in text
        assert "not claimed as a full FINE reproduction" in text or "not an official reproduction" in text
    else:
        assert "Graph-CoLD_vs_FINE-style" not in text
    assert "Graph-CoLD_vs_FINE-style" not in text
    assert pdf_path.exists()
    assert len(PdfReader(str(pdf_path)).pages) >= 10
