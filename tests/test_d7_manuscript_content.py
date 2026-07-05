from pathlib import Path

from src.paper.d7_assemble import run_d7_assembly


MANUSCRIPT = Path("paper/elsevier/graph_cold_cas_realdata.tex")


def _ensure_d7_outputs() -> str:
    if not MANUSCRIPT.exists():
        run_d7_assembly()
    return MANUSCRIPT.read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_d7_manuscript_has_required_structure_and_datasets():
    text = _ensure_d7_outputs()
    normalized = _normalized(text)

    heading_groups = [
        [r"\section{Introduction}"],
        [r"\section{Related Work}"],
        [r"\section{Problem Formulation and Motivation}", r"\section{Problem Formulation and Design Goals}"],
        [r"\section{Methodology}", r"\section{Method}"],
        [r"\section{Experimental Setup}", r"\section{Experimental Design}"],
        [r"\section{Results and Analysis}", r"\section{Results}"],
        [r"\section{Discussion}"],
        [r"\section{Limitations}"],
        [r"\section{Conclusion}"],
    ]
    for headings in heading_groups:
        assert any(heading in text for heading in headings)

    assert "CICIDS-2017 postfilter11" in normalized
    assert "CESNET-TLS-Year22 postfilter25" in normalized
    assert "deterministic audit-window subset" in normalized
    assert "not a full-archive evaluation" in normalized
    assert "MALTLS-22 is not evaluated" in normalized
    assert "OpTC is not evaluated as a formal enterprise case" in normalized


def test_d7_manuscript_contains_label_space_graph_cdm_and_weight_formula():
    text = _ensure_d7_outputs()
    normalized = _normalized(text)

    assert (
        "label-space consistency diagnostic function" in normalized
        or "label-space Graph-CDM diagnostic" in normalized
        or "label-space consistency diagnostic" in normalized
    )
    assert r"\operatorname{GraphCDM}(v)" in text
    assert r"\lambda_1D_{pred}(v)+\lambda_2D_{neigh}(v)+" in text
    assert r"D_{pred}(v)" in text
    assert "observed noisy training label" in normalized
    assert r"D_{neigh}(v)=KL" in text
    assert r"D_{view}(v)=1-\max_c" in text
    assert r"w(v)=\sigma" in text
    assert r"P(v)=\alpha_1" in text


def test_d7_manuscript_declares_baseline_scope_without_overclaiming():
    text = _ensure_d7_outputs()
    normalized = _normalized(text)

    assert "Co-Teaching-lite" in text
    assert (
        "lightweight smoke-passed approximation" in normalized
        or "lightweight implemented approximation" in normalized
    )
    assert "not a full Co-Teaching implementation" in normalized
    for method in ["FINE", "MCRe", "MORSE", "Flash", "Argus", "Decoupling", "full Co-Teaching"]:
        assert method in text
    assert "excluded from formal comparison" in normalized
