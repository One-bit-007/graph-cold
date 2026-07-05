import json
from pathlib import Path

from pypdf import PdfReader


def test_d11_candidate_package_manifest_and_required_files():
    package = Path("submission/cas_candidate_d11")
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["candidate_package_ready"] is True
    assert manifest["submission_ready"] is False
    assert manifest["human_confirmation_required"] is True
    assert manifest["raw_data_packaged"] is False
    assert manifest["aria2_logs_packaged"] is False

    required = {
        "manuscript_author_version_d11.tex",
        "manuscript_author_version_d11.pdf",
        "manuscript_anonymous_review_d11.tex",
        "manuscript_anonymous_review_d11.pdf",
        "cover_letter_d11.md",
        "highlights_d11.md",
        "data_availability_statement_d11.md",
        "generative_ai_statement_d11.md",
        "reviewer_risk_pack_d11.md",
        "rebuttal_prewrite_full.md",
        "package_manifest.json",
    }
    assert required.issubset(set(manifest["included_files"]))
    for name in required:
        assert (package / name).exists()

    assert len(PdfReader(str(package / "manuscript_author_version_d11.pdf")).pages) >= 10
    assert len(PdfReader(str(package / "manuscript_anonymous_review_d11.pdf")).pages) >= 10


def test_d11_candidate_package_excludes_raw_data_and_aria2_logs():
    package = Path("submission/cas_candidate_d11")
    names = [str(path.relative_to(package)).replace("\\", "/").lower() for path in package.rglob("*") if path.is_file()]
    assert not any(name.startswith("data/") for name in names)
    assert not any(name.endswith((".zip", ".pcap", ".pcapng", ".7z", ".gz")) for name in names)
    assert not any("aria2" in name for name in names)
