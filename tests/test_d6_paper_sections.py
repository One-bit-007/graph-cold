from pathlib import Path

from src.paper.d6_prep import run_d6_realdata_prep


def _ensure_d6_outputs():
    if not Path("reports/d6/d6_generation_manifest.json").exists():
        run_d6_realdata_prep()


def _section_text() -> str:
    files = [
        "paper/sections/experiments_realdata.tex",
        "paper/sections/results_realdata.tex",
        "paper/sections/discussion_realdata.tex",
        "paper/sections/limitations_realdata.tex",
    ]
    return "\n".join(Path(path).read_text(encoding="utf-8") for path in files)


def test_paper_sections_exist_and_use_correct_dataset_names():
    _ensure_d6_outputs()
    for path in [
        "paper/sections/experiments_realdata.tex",
        "paper/sections/results_realdata.tex",
        "paper/sections/discussion_realdata.tex",
        "paper/sections/limitations_realdata.tex",
    ]:
        assert Path(path).exists(), path

    text = _section_text()
    assert "CICIDS-2017" in text
    assert "CESNET-TLS-Year22" in text
    assert "MALTLS-22 is not included" in text
    assert "OpTC is not included as a formal experiment" in text
    assert "Co-Teaching" in text
    assert "MCRe" in text
    assert "MORSE" in text


def test_paper_sections_report_sample_policy_and_do_not_claim_unavailable_results():
    _ensure_d6_outputs()
    text = _section_text()

    assert "sample policy" in text
    assert "full-archive evaluation" in text
    assert "MALTLS-22 results" not in text
    assert "OpTC Table" not in text
    assert "OpTC figure" not in text
    assert not Path("paper/graph_cold_cas_submission.pdf").exists()
