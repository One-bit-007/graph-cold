from pathlib import Path


FORBIDDEN_TERMS = [
    "dominates",
    "state-of-the-art",
    "massive",
    "near-perfect",
    "universal",
    "guaranteed",
    "causal proof",
]


def test_d11_final_manuscript_and_package_do_not_use_forbidden_overclaiming_terms():
    paths = [
        Path("paper/elsevier/final_candidate/graph_cold_cas_realdata_d11.tex"),
        Path("submission/cas_candidate_d11/manuscript_author_version_d11.tex"),
        Path("submission/cas_candidate_d11/manuscript_anonymous_review_d11.tex"),
        Path("submission/cas_candidate_d11/cover_letter_d11.md"),
        Path("submission/cas_candidate_d11/highlights_d11.md"),
        Path("submission/cas_candidate_d11/reviewer_risk_pack_d11.md"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"{term!r} found in {path}"


def test_d11_final_manuscript_has_no_local_windows_paths():
    text = Path("paper/elsevier/final_candidate/graph_cold_cas_realdata_d11.tex").read_text(encoding="utf-8")
    assert "C:\\" not in text
    assert "E:\\" not in text
