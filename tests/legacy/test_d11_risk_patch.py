import json
from pathlib import Path

from pypdf import PdfReader


def test_d11_mandatory_risk_patch_present_in_tex_and_pdf_compiles():
    tex_path = Path("paper/elsevier/final_candidate/graph_cold_cas_realdata_d11.tex")
    pdf_path = Path("paper/elsevier/final_candidate/graph_cold_cas_realdata_d11.pdf")
    text = tex_path.read_text(encoding="utf-8")

    assert "FINE-style was implemented as a representation/eigenvector filtering baseline" in text
    assert "CESNET-TLS-Year22 exhibits a ceiling effect in Macro-F1" in text
    assert "Decoupling relies on prediction disagreement as the update signal" in text
    assert "ERR is not a detection-accuracy score" in text
    assert "The comparison covers implemented and real-data smoke-passed baselines" in text
    assert "Declaration of generative AI and AI-assisted technologies" in text

    assert pdf_path.exists()
    assert len(PdfReader(str(pdf_path)).pages) >= 10


def test_d11_final_audit_confirms_patch_and_keeps_submission_false():
    audit = json.loads(Path("reports/d11/d11_final_audit.json").read_text(encoding="utf-8"))
    assert audit["mandatory_risk_patch_applied"] is True
    assert audit["fine_style_exclusion_explained"] is True
    assert audit["cesnet_ceiling_effect_explained"] is True
    assert audit["decoupling_limitation_explained"] is True
    assert audit["err_one_clarified"] is True
    assert audit["pdf_compiles"] is True
    assert audit["submission_ready"] is False
    assert audit["human_confirmation_required"] is True
    assert audit["missing_requested_inputs"] == []
